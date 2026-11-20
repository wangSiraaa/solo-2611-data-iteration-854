"""Finite-voice player: replays a performance product against a voice budget.

The player is a *separate reader* of the product: it takes an immutable
:class:`telber.keys.Performance` (built earlier, possibly re-loaded from its
file) plus playback parameters — the total number of voices (slots) and a
per-channel priority — and derives a :class:`ReplayResult`.  Adjusting the
player or its parameters never changes the performance product.

Rules implemented here:

* a voice's natural stop releases its slot at the stop's actual event
  position (which already accounts for the sustain pedal);
* when a new voice arrives and no slot is free, the new voice is compared
  *together with* the current holders: the lowest priority loses, and ties go
  against the earliest-started voice.  If the new voice itself loses, it
  never sounds (``no_voice``);
* every voice is actually played at most once: a truncated tail never
  revives, even if a slot frees up later;
* a voice still sounding when the recording cut off keeps an undefined right
  end (``open``).

The result reports, per voice, the actually-sounding interval and the reason
it ended (natural stop / truncated / never given a voice / open), and answers
"which voices are truly audible at this beat position".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .errors import PlayerError
from .keys import Performance, Pos

# Playback statuses (the "reason" categories of the replay report).
NATURAL = "natural"  # ended at its natural stop
TRUNCATED = "truncated"  # evicted by a higher-priority arrival
NO_VOICE = "no_voice"  # never given a slot
OPEN = "open"  # still sounding at the recording cutoff (right end undefined)


@dataclass(frozen=True)
class Playback:
    """The actually-sounding interval of one performance voice."""

    voice_id: int
    channel: int
    pitch: int
    press: Pos
    start: Pos | None  # first audible position (None if never sounded)
    end: Pos | None  # stops being audible here (None if open/never sounded)
    status: str  # NATURAL | TRUNCATED | NO_VOICE | OPEN
    reason: str
    evicted_by: int | None = None  # voice id that took the slot (truncated)
    competitors: tuple[int, ...] = ()  # holders that outranked it (no_voice)

    def audible_at(self, pos: Pos) -> bool:
        """Half-open interval test: audible for ``start <= pos < end``; an
        undefined right end stays audible from ``start`` on."""
        if self.start is None or pos < self.start:
            return False
        return self.end is None or pos < self.end

    def as_dict(self) -> dict:
        return {
            "voice_id": self.voice_id,
            "channel": self.channel,
            "pitch": self.pitch,
            "press": self.press.as_list(),
            "start": None if self.start is None else self.start.as_list(),
            "end": None if self.end is None else self.end.as_list(),
            "status": self.status,
            "reason": self.reason,
            "evicted_by": self.evicted_by,
            "competitors": list(self.competitors),
        }


@dataclass(frozen=True)
class ReplayResult:
    """One replay of a performance under a voice budget and priorities."""

    voices_total: int
    channel_priority: tuple[tuple[int, int], ...]
    playbacks: tuple[Playback, ...]

    def audible_at(self, pos: Pos | tuple[int, int]) -> list[Playback]:
        """The voices truly audible at the given (beat, seq) position."""
        pos = Pos(*pos)
        return [p for p in self.playbacks if p.audible_at(pos)]

    def playback(self, voice_id: int) -> Playback:
        return self.playbacks[voice_id]

    def as_dict(self) -> dict:
        return {
            "voices_total": self.voices_total,
            "channel_priority": {str(c): p for c, p in self.channel_priority},
            "playbacks": [p.as_dict() for p in self.playbacks],
        }


class _SlotState:
    """Mutable per-voice replay state (the product itself stays frozen)."""

    __slots__ = ("start", "end", "status", "reason", "evicted_by", "competitors", "holding")

    def __init__(self) -> None:
        self.start: Pos | None = None
        self.end: Pos | None = None
        self.status: str | None = None
        self.reason = ""
        self.evicted_by: int | None = None
        self.competitors: tuple[int, ...] = ()
        self.holding = False


def replay(
    performance: Performance,
    voices_total: int,
    channel_priority: Mapping[int, int] | None = None,
) -> ReplayResult:
    """Replay ``performance`` against ``voices_total`` slots.

    ``channel_priority`` maps channel number to a priority; higher wins and
    unlisted channels default to 0.  The performance product is only read,
    never mutated.
    """
    if isinstance(voices_total, bool) or not isinstance(voices_total, int) or voices_total < 1:
        raise PlayerError(
            f"voices_total must be a positive integer, got {voices_total!r}",
            code="bad_voice_count",
        )
    priority = dict(channel_priority or {})
    prio_of = lambda channel: priority.get(channel, 0)  # noqa: E731

    voices = performance.voices
    states = [_SlotState() for _ in voices]

    arrivals: dict[Pos, list[int]] = {}
    stops: dict[Pos, list[int]] = {}
    for i, v in enumerate(voices):
        arrivals.setdefault(v.press, []).append(i)
        if v.end is not None:
            stops.setdefault(v.end, []).append(i)

    holders: list[int] = []  # voice ids currently occupying a slot

    def release(i: int, pos: Pos) -> None:
        st = states[i]
        st.holding = False
        holders.remove(i)
        st.end = pos
        st.status = NATURAL
        st.reason = f"ended at its natural stop {tuple(pos)}"

    for pos in sorted(set(arrivals) | set(stops)):
        # 1. Natural stops release their slots at their actual event position.
        deferred: list[int] = []
        for i in stops.get(pos, []):
            if states[i].holding:
                release(i, pos)
            elif states[i].status is None:
                # The voice has not even started (a zero-length voice): apply
                # its stop right after the arrivals at this same position.
                deferred.append(i)
            # Otherwise the voice already lost its slot: a truncated or
            # never-played tail is not revived and needs no further stop.

        # 2. Arrivals compete for the remaining free slots, in voice order.
        for i in arrivals.get(pos, []):
            st = states[i]
            if len(holders) < voices_total:
                st.holding = True
                holders.append(i)
                st.start = pos
                continue
            # No free slot: the new voice is compared together with the
            # current holders; the lowest priority loses, ties go against the
            # earliest-started voice.  The new voice's start counts as now.
            candidates = holders + [i]
            loser = min(
                candidates,
                key=lambda j: (
                    prio_of(voices[j].channel),
                    states[j].start if states[j].start is not None else pos,
                    j,
                ),
            )
            if loser == i:
                st.status = NO_VOICE
                st.competitors = tuple(holders)
                st.reason = (
                    f"no free slot at {tuple(pos)}; priority "
                    f"{prio_of(voices[i].channel)} is the lowest among "
                    f"holders {holders}"
                )
            else:
                old = states[loser]
                old.holding = False
                holders.remove(loser)
                old.end = pos
                old.status = TRUNCATED
                old.evicted_by = i
                old.reason = (
                    f"truncated at {tuple(pos)}: voice {i} (channel "
                    f"{voices[i].channel}) took the slot"
                )
                st.holding = True
                holders.append(i)
                st.start = pos

        # 3. Zero-length voices (press and stop at the same position).
        for i in deferred:
            if states[i].holding:
                release(i, pos)
            elif states[i].status is None:
                states[i].status = NO_VOICE
                states[i].reason = f"no free slot at {tuple(pos)}"

    # Voices still holding a slot were cut off by the end of the recording.
    for i in holders:
        st = states[i]
        st.status = OPEN
        st.reason = "still sounding at the recording cutoff; right end undefined"

    playbacks = tuple(
        Playback(
            voice_id=v.voice_id,
            channel=v.channel,
            pitch=v.pitch,
            press=v.press,
            start=st.start,
            end=st.end,
            status=st.status if st.status is not None else NO_VOICE,
            reason=st.reason,
            evicted_by=st.evicted_by,
            competitors=st.competitors,
        )
        for v, st in zip(voices, states)
    )
    return ReplayResult(
        voices_total=voices_total,
        channel_priority=tuple(sorted(priority.items())),
        playbacks=playbacks,
    )
