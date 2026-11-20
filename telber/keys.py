"""Sustained-note performance product built from keyboard/pedal telemetry.

This module sits *above* the generic record machinery: the wire format of one
message is an ordinary compiled directory (``schemas/keys_v1.json``), ingested
into the fact repository exactly like any other product.  Here the imported
field facts are compiled into an immutable **performance product**:

* every key press creates an independent **voice identity**; re-pressing a
  pitch that is still held on the same channel is legal, and a key release is
  always paired with the *earliest still-held* press of that channel+pitch;
* the sustain pedal is tracked per channel: a key released while the pedal is
  down keeps sounding until the pedal is lifted, and lifting the pedal ends
  only voices whose key is already released (a voice still held by the finger
  is unaffected);
* messages on the same beat take effect one by one in message-sequence order,
  so the product keeps the exact within-beat order, the raw press position and
  the position at which each voice should stop sounding;
* a key release with nothing held to pair with is reported as an *orphan
  release* together with the source message it came from;
* voices still unfinished when the recording cuts off keep an **undefined
  right end** (``end is None``).

From the one product, finger-held intervals (press → key release) and natural
sounding intervals (press → should-stop position) are queried separately.  The
product is immutable and JSON round-trippable; the player in
:mod:`telber.player` reads it (from object or file) and can never change it.
"""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, NamedTuple

from .errors import KeysError

SCHEMA_NAME = "keys.event"
PRODUCT_KIND = "keys.performance"
PRODUCT_VERSION = 1

# Message actions (the ``action`` INTEGER of a keys.event record).
KEY_DOWN = 0
KEY_UP = 1
PEDAL_DOWN = 2
PEDAL_UP = 3
ACTIONS = (KEY_DOWN, KEY_UP, PEDAL_DOWN, PEDAL_UP)
ACTION_NAMES = {KEY_DOWN: "key_down", KEY_UP: "key_up", PEDAL_DOWN: "pedal_down", PEDAL_UP: "pedal_up"}

# Voice end kinds recorded in the product.
END_RELEASE = "release"  # key released with the pedal up: stops at the release
END_PEDAL = "pedal"  # released under a held pedal: stops when the pedal lifts


class Pos(NamedTuple):
    """A musical position: integer beat plus the message sequence within it."""

    beat: int
    seq: int

    def as_list(self) -> list[int]:
        return [self.beat, self.seq]


def _pos(obj: Any, where: str) -> Pos:
    if (
        not isinstance(obj, (list, tuple))
        or len(obj) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in obj)
    ):
        raise KeysError(f"{where} must be a [beat, seq] pair of integers", code="invalid_product")
    return Pos(obj[0], obj[1])


@dataclass(frozen=True)
class Message:
    """One keyboard/pedal message with its repository provenance.

    ``record`` is the record ordinal inside the import; ``start``/``end`` are
    the absolute stream offsets of the record's encoding.
    """

    beat: int
    seq: int
    channel: int
    action: int
    pitch: int | None
    record: int = 0
    start: int | None = None
    end: int | None = None

    @property
    def pos(self) -> Pos:
        return Pos(self.beat, self.seq)


@dataclass(frozen=True)
class Voice:
    """One independent voice identity created by a key press.

    ``press`` is the raw press position.  ``release`` is the finger-release
    position (``None`` while the key is still held at the recording cutoff).
    ``end`` is the natural should-stop position (``None`` = right end
    undefined: the recording cut off before the voice ended normally).
    """

    voice_id: int
    channel: int
    pitch: int
    press: Pos
    press_record: int
    release: Pos | None
    release_record: int | None
    end: Pos | None
    end_record: int | None
    end_kind: str | None  # END_RELEASE | END_PEDAL | None (undefined right end)

    def finger_interval(self) -> tuple[Pos, Pos | None]:
        return self.press, self.release

    def sounding_interval(self) -> tuple[Pos, Pos | None]:
        return self.press, self.end

    def as_dict(self) -> dict:
        return {
            "voice_id": self.voice_id,
            "channel": self.channel,
            "pitch": self.pitch,
            "press": self.press.as_list(),
            "press_record": self.press_record,
            "release": None if self.release is None else self.release.as_list(),
            "release_record": self.release_record,
            "end": None if self.end is None else self.end.as_list(),
            "end_record": self.end_record,
            "end_kind": self.end_kind,
        }

    @staticmethod
    def from_dict(obj: Any) -> "Voice":
        if not isinstance(obj, dict):
            raise KeysError("voice entry must be an object", code="invalid_product")
        try:
            return Voice(
                voice_id=int(obj["voice_id"]),
                channel=int(obj["channel"]),
                pitch=int(obj["pitch"]),
                press=_pos(obj["press"], "voice.press"),
                press_record=int(obj["press_record"]),
                release=None if obj["release"] is None else _pos(obj["release"], "voice.release"),
                release_record=obj["release_record"],
                end=None if obj["end"] is None else _pos(obj["end"], "voice.end"),
                end_record=obj["end_record"],
                end_kind=obj["end_kind"],
            )
        except KeyError as exc:
            raise KeysError(f"voice entry is missing {exc}", code="invalid_product") from exc


@dataclass(frozen=True)
class OrphanRelease:
    """A key-release message that had no held press to pair with.

    The source of the message is reported: its record ordinal and the absolute
    stream offsets of its encoding.
    """

    channel: int
    pitch: int
    pos: Pos
    record: int
    start: int | None
    end: int | None

    def as_dict(self) -> dict:
        return {
            "channel": self.channel,
            "pitch": self.pitch,
            "pos": self.pos.as_list(),
            "record": self.record,
            "start": self.start,
            "end": self.end,
        }

    @staticmethod
    def from_dict(obj: Any) -> "OrphanRelease":
        if not isinstance(obj, dict):
            raise KeysError("orphan release entry must be an object", code="invalid_product")
        try:
            return OrphanRelease(
                channel=int(obj["channel"]),
                pitch=int(obj["pitch"]),
                pos=_pos(obj["pos"], "orphan.pos"),
                record=int(obj["record"]),
                start=obj["start"],
                end=obj["end"],
            )
        except KeyError as exc:
            raise KeysError(f"orphan release entry is missing {exc}", code="invalid_product") from exc


@dataclass(frozen=True)
class SourceInfo:
    """Where a performance product was derived from."""

    schema_product: str | None = None
    import_id: int | None = None
    message_count: int = 0

    def as_dict(self) -> dict:
        return {
            "schema_product": self.schema_product,
            "import_id": self.import_id,
            "message_count": self.message_count,
        }


@dataclass(frozen=True)
class Performance:
    """The immutable sustained-note performance product.

    Finger-held intervals and natural sounding intervals are both derivable
    from the same product (and the same serialised file) via
    :meth:`finger_intervals` and :meth:`sounding_intervals`.
    """

    voices: tuple[Voice, ...]
    orphan_releases: tuple[OrphanRelease, ...]
    source: SourceInfo = SourceInfo()

    # -- interval queries ---------------------------------------------------
    def finger_intervals(self) -> list[dict]:
        """Press → key-release interval of every voice (``end=None`` if the
        key was still held when the recording cut off)."""
        return [
            {
                "voice": v.voice_id,
                "channel": v.channel,
                "pitch": v.pitch,
                "start": v.press,
                "end": v.release,
            }
            for v in self.voices
        ]

    def sounding_intervals(self) -> list[dict]:
        """Press → natural should-stop interval of every voice
        (``end=None`` marks an undefined right end at the recording cutoff)."""
        return [
            {
                "voice": v.voice_id,
                "channel": v.channel,
                "pitch": v.pitch,
                "start": v.press,
                "end": v.end,
            }
            for v in self.voices
        ]

    # -- serialisation ------------------------------------------------------
    def as_dict(self) -> dict:
        return {
            "kind": PRODUCT_KIND,
            "version": PRODUCT_VERSION,
            "source": self.source.as_dict(),
            "voices": [v.as_dict() for v in self.voices],
            "orphan_releases": [o.as_dict() for o in self.orphan_releases],
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"

    def save(self, path: str | Path) -> None:
        Path(path).write_text(self.to_json(), encoding="utf-8")

    @staticmethod
    def from_dict(obj: Any) -> "Performance":
        if not isinstance(obj, dict):
            raise KeysError("performance product must be a JSON object", code="invalid_product")
        if obj.get("kind") != PRODUCT_KIND:
            raise KeysError(
                f"not a {PRODUCT_KIND!r} document (kind={obj.get('kind')!r})",
                code="invalid_product",
            )
        if obj.get("version") != PRODUCT_VERSION:
            raise KeysError(
                f"unsupported performance product version {obj.get('version')!r}",
                code="invalid_product",
            )
        source_raw = obj.get("source") or {}
        source = SourceInfo(
            schema_product=source_raw.get("schema_product"),
            import_id=source_raw.get("import_id"),
            message_count=int(source_raw.get("message_count", 0)),
        )
        voices = tuple(Voice.from_dict(v) for v in obj.get("voices", []))
        orphans = tuple(OrphanRelease.from_dict(o) for o in obj.get("orphan_releases", []))
        return Performance(voices=voices, orphan_releases=orphans, source=source)

    @staticmethod
    def from_json(text: str | bytes) -> "Performance":
        try:
            obj = json.loads(text)
        except json.JSONDecodeError as exc:
            raise KeysError(f"performance product is not valid JSON: {exc}", code="invalid_product") from exc
        return Performance.from_dict(obj)


def load_performance(path: str | Path) -> Performance:
    """Read a performance product back from its file (the replay entry point
    uses this so the player never touches the telemetry repository)."""
    return Performance.from_json(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Building the product


def load_messages(repo, import_id: int) -> list[Message]:
    """Reconstruct the ordered message list from a repository import.

    Messages are ordered by ``(beat, seq, record ordinal)``: the beat and its
    within-beat sequence number define the musical order; the record ordinal
    keeps a deterministic order for messages sharing both.
    """
    info = repo.import_info(import_id)
    product = repo.get_product(info.product_id)
    if product.schema_name != SCHEMA_NAME:
        raise KeysError(
            f"import {import_id} is bound to schema {product.schema_name!r}, "
            f"not {SCHEMA_NAME!r}; cannot build a performance from it",
            code="wrong_product",
        )
    records = {r["ordinal"]: r for r in repo.records(import_id)}
    fields: dict[int, dict[str, int]] = {}
    for row in repo.relation_rows(import_id, present_only=True):
        fields.setdefault(row["record_ordinal"], {})[row["path"]] = row["int_value"]
    messages: list[Message] = []
    for ordinal in sorted(fields):
        f = fields[ordinal]
        rec = records[ordinal]
        messages.append(
            Message(
                beat=f["beat"],
                seq=f["seq"],
                channel=f["channel"],
                action=f["action"],
                pitch=f.get("pitch"),
                record=ordinal,
                start=rec["start_offset"],
                end=rec["end_offset"],
            )
        )
    messages.sort(key=lambda m: (m.beat, m.seq, m.record))
    return messages


def performance_from_import(repo, import_id: int) -> Performance:
    """Build the performance product directly from a repository import."""
    info = repo.import_info(import_id)
    messages = load_messages(repo, import_id)
    return build_performance(
        messages,
        source=SourceInfo(
            schema_product=info.product_id,
            import_id=import_id,
            message_count=len(messages),
        ),
    )


class _VoiceBuilder:
    __slots__ = (
        "voice_id",
        "channel",
        "pitch",
        "press",
        "press_record",
        "release",
        "release_record",
        "end",
        "end_record",
        "end_kind",
    )

    def __init__(self, voice_id: int, msg: Message) -> None:
        self.voice_id = voice_id
        self.channel = msg.channel
        self.pitch = msg.pitch
        self.press = msg.pos
        self.press_record = msg.record
        self.release: Pos | None = None
        self.release_record: int | None = None
        self.end: Pos | None = None
        self.end_record: int | None = None
        self.end_kind: str | None = None

    def freeze(self) -> Voice:
        return Voice(
            voice_id=self.voice_id,
            channel=self.channel,
            pitch=self.pitch,  # type: ignore[arg-type]
            press=self.press,
            press_record=self.press_record,
            release=self.release,
            release_record=self.release_record,
            end=self.end,
            end_record=self.end_record,
            end_kind=self.end_kind,
        )


def _check_message(msg: Message) -> None:
    where = f"message at beat {msg.beat} seq {msg.seq} (record {msg.record})"
    if msg.beat < 0 or msg.seq < 0:
        raise KeysError(f"{where}: negative beat/seq position", offset=msg.start, code="bad_position")
    if msg.action not in ACTIONS:
        raise KeysError(
            f"{where}: unknown action {msg.action}; expected one of {ACTIONS}",
            offset=msg.start,
            code="unknown_action",
        )
    if msg.action in (KEY_DOWN, KEY_UP) and msg.pitch is None:
        raise KeysError(
            f"{where}: {ACTION_NAMES[msg.action]} requires a pitch",
            offset=msg.start,
            code="missing_pitch",
        )
    if msg.action in (PEDAL_DOWN, PEDAL_UP) and msg.pitch is not None:
        raise KeysError(
            f"{where}: {ACTION_NAMES[msg.action]} must not carry a pitch",
            offset=msg.start,
            code="unexpected_pitch",
        )


def build_performance(
    messages: Iterable[Message], *, source: SourceInfo | None = None
) -> Performance:
    """Compile ordered keyboard/pedal messages into the performance product.

    Messages are applied one by one in ``(beat, seq)`` order, so same-beat
    key/pedal combinations take effect in message-sequence order.  The input
    is re-sorted defensively; pass :func:`load_messages` output or any
    equivalent sequence.
    """
    ordered = sorted(messages, key=lambda m: (m.beat, m.seq, m.record))
    for msg in ordered:
        _check_message(msg)

    voices: list[_VoiceBuilder] = []
    orphans: list[OrphanRelease] = []
    held: dict[tuple[int, int], deque[_VoiceBuilder]] = {}
    sustained: dict[int, list[_VoiceBuilder]] = {}
    pedal_down: set[int] = set()

    for msg in ordered:
        pos = msg.pos
        if msg.action == KEY_DOWN:
            voice = _VoiceBuilder(len(voices), msg)
            voices.append(voice)
            held.setdefault((msg.channel, msg.pitch), deque()).append(voice)
        elif msg.action == KEY_UP:
            queue = held.get((msg.channel, msg.pitch))
            if not queue:
                # Nothing held for this channel+pitch: report the extra
                # release together with the message it came from.
                orphans.append(
                    OrphanRelease(
                        channel=msg.channel,
                        pitch=msg.pitch,  # type: ignore[arg-type]
                        pos=pos,
                        record=msg.record,
                        start=msg.start,
                        end=msg.end,
                    )
                )
                continue
            voice = queue.popleft()  # the earliest still-held press
            voice.release = pos
            voice.release_record = msg.record
            if msg.channel in pedal_down:
                # The pedal holds the voice; it ends when the pedal lifts.
                sustained.setdefault(msg.channel, []).append(voice)
            else:
                voice.end = pos
                voice.end_record = msg.record
                voice.end_kind = END_RELEASE
        elif msg.action == PEDAL_DOWN:
            pedal_down.add(msg.channel)
        else:  # PEDAL_UP
            pedal_down.discard(msg.channel)
            # Lifting the pedal ends only voices whose key is already
            # released; a voice still held by the finger keeps sounding.
            for voice in sustained.pop(msg.channel, []):
                if voice.end is None:
                    voice.end = pos
                    voice.end_record = msg.record
                    voice.end_kind = END_PEDAL

    # Voices with ``end is None`` were still sounding (held or sustained) when
    # the recording cut off: their right end stays undefined.
    return Performance(
        voices=tuple(v.freeze() for v in voices),
        orphan_releases=tuple(orphans),
        source=source if source is not None else SourceInfo(message_count=len(ordered)),
    )
