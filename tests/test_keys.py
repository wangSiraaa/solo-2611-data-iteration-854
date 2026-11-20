"""Keyboard/pedal product and finite-voice player: regression checks.

Ground-truth streams are built with the independent byte builder
(:mod:`telber.build`), not the library encoder, so the wire format is checked
against an independent construction.  The player scenarios follow the
specification:

* single voice, channel A prioritised: G never sounds;
* single voice, channel B prioritised: C is truncated at beat 3 and stays
  silent after G ends (a truncated tail never revives);
* pedal lifted just before the G press: G sounds until beat 4.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from telber import build as B
from telber.cli import main as cli_main
from telber.codec import decode_record, encode_record
from telber.errors import KeysError, PlayerError
from telber.keydemo import run as keydemo_run
from telber.keys import (
    KEY_DOWN,
    KEY_UP,
    PEDAL_DOWN,
    PEDAL_UP,
    Message,
    Performance,
    Pos,
    build_performance,
    load_messages,
    load_performance,
    performance_from_import,
)
from telber.player import NATURAL, NO_VOICE, OPEN, TRUNCATED, replay
from telber.repository import Repository
from telber.schema import compile_product

ROOT = Path(__file__).resolve().parent.parent
KEYS_SCHEMA_TEXT = (ROOT / "schemas" / "keys_v1.json").read_text(encoding="utf-8")
TELEMETRY_SCHEMA_TEXT = (ROOT / "schemas" / "telemetry_v1.json").read_text(encoding="utf-8")

CH_A, CH_B = 0, 1
C4, G4 = 60, 67

# The synthetic recording: beat 0 A down C, beat 1 pedal, beat 2 up C,
# beat 3 B down G, beat 4 up G, beat 5 pedal up.
MAIN_MESSAGES = [
    (0, 0, CH_A, KEY_DOWN, C4),
    (1, 0, CH_A, PEDAL_DOWN, None),
    (2, 0, CH_A, KEY_UP, C4),
    (3, 0, CH_B, KEY_DOWN, G4),
    (4, 0, CH_B, KEY_UP, G4),
    (5, 0, CH_A, PEDAL_UP, None),
]

# Variant: pedal lifted at beat 3 just before the G press.
VARIANT_MESSAGES = [
    (0, 0, CH_A, KEY_DOWN, C4),
    (1, 0, CH_A, PEDAL_DOWN, None),
    (2, 0, CH_A, KEY_UP, C4),
    (3, 0, CH_A, PEDAL_UP, None),
    (3, 1, CH_B, KEY_DOWN, G4),
    (4, 0, CH_B, KEY_UP, G4),
]


def key_msg(beat, seq, channel, action, pitch=None, *, indefinite=False):
    """Independently built wire bytes of one keys.event record."""
    body = B.integer(beat, cls="context", number=0)
    body += B.integer(seq, cls="context", number=1)
    body += B.integer(channel, cls="context", number=2)
    body += B.integer(action, cls="context", number=3)
    if pitch is not None:
        body += B.integer(pitch, cls="context", number=4)
    return B.tlv(B.tag("application", True, 40), body, indefinite=indefinite)


def stream_of(messages, **kw):
    return b"".join(key_msg(*m, **kw) for m in messages)


@pytest.fixture
def keys_product():
    return compile_product(KEYS_SCHEMA_TEXT)


@pytest.fixture
def main_performance(keys_product, tmp_path):
    """The main synthetic recording, ingested and compiled to a product."""
    stream = stream_of(MAIN_MESSAGES)
    with Repository(tmp_path / "keys.sqlite") as repo:
        repo.publish_product(KEYS_SCHEMA_TEXT)
        import_id = repo.create_import(keys_product.product_id, source_name="main")
        repo.ingest_bytes(import_id, stream, block_size=1)
        return performance_from_import(repo, import_id)


def make_performance(messages):
    return build_performance(
        [
            Message(beat=b, seq=s, channel=c, action=a, pitch=p, record=i)
            for i, (b, s, c, a, p) in enumerate(messages)
        ]
    )


# ---------------------------------------------------------------------------
# Wire format


def test_schema_compiles_with_distinct_root(keys_product):
    telemetry = compile_product(TELEMETRY_SCHEMA_TEXT)
    assert keys_product.schema_name == "keys.event"
    assert keys_product.root_tag != telemetry.root_tag
    assert keys_product.root_tag.cls == "application"
    assert keys_product.product_id != telemetry.product_id
    assert keys_product.field_paths() == ["beat", "seq", "channel", "action", "pitch"]


def test_wire_roundtrip_against_independent_bytes(keys_product):
    data = key_msg(3, 1, CH_B, KEY_DOWN, G4)
    record = decode_record(data, keys_product)
    values = record.value_map()
    assert values["beat"].value == 3
    assert values["seq"].value == 1
    assert values["channel"].value == CH_B
    assert values["action"].value == KEY_DOWN
    assert values["pitch"].value == G4
    # The library encoder reproduces the independent bytes exactly.
    assert (
        encode_record(
            {"beat": 3, "seq": 1, "channel": CH_B, "action": KEY_DOWN, "pitch": G4},
            keys_product,
        )
        == data
    )
    # A pedal message carries no pitch; indefinite outer layers decode too.
    pedal = key_msg(5, 0, CH_A, PEDAL_UP, indefinite=True)
    values = decode_record(pedal, keys_product).value_map()
    assert values["action"].value == PEDAL_UP
    assert "pitch" not in values
    assert encode_record(
        {"beat": 5, "seq": 0, "channel": CH_A, "action": PEDAL_UP}, keys_product
    ) == key_msg(5, 0, CH_A, PEDAL_UP)


def test_ingest_and_load_messages(keys_product, tmp_path):
    stream = stream_of(MAIN_MESSAGES)
    with Repository(tmp_path / "keys.sqlite") as repo:
        repo.publish_product(KEYS_SCHEMA_TEXT)
        import_id = repo.create_import(keys_product.product_id, source_name="main")
        info = repo.ingest_bytes(import_id, stream, block_size=1)
        assert info.status == "complete" and info.record_count == 6

        messages = load_messages(repo, import_id)
        assert [(m.beat, m.seq, m.channel, m.action, m.pitch) for m in messages] == [
            (b, s, c, a, p) for b, s, c, a, p in MAIN_MESSAGES
        ]
        # Provenance: each message's absolute offsets slice its exact record.
        for i, m in enumerate(messages):
            assert m.record == i
            assert stream[m.start : m.end] == key_msg(*MAIN_MESSAGES[i])

        # Block size is I/O granularity only.
        other_id = repo.create_import(keys_product.product_id, source_name="bs7")
        repo.ingest_bytes(other_id, stream, block_size=7)
        assert load_messages(repo, other_id) == messages


def test_load_messages_rejects_non_keys_import(tmp_path):
    with Repository(tmp_path / "mixed.sqlite") as repo:
        telemetry = repo.publish_product(TELEMETRY_SCHEMA_TEXT)
        import_id = repo.create_import(telemetry.product_id, source_name="telemetry")
        repo.ingest_bytes(
            import_id,
            encode_record(
                {"station_id": 1, "observed_at": 2, "sensor": {"kind": 0}}, telemetry
            ),
        )
        with pytest.raises(KeysError) as excinfo:
            load_messages(repo, import_id)
        assert excinfo.value.code == "wrong_product"


# ---------------------------------------------------------------------------
# Performance product semantics


def test_performance_main_scenario(main_performance):
    perf = main_performance
    assert len(perf.voices) == 2
    c, g = perf.voices
    # C: finger interval ends at the key release; the pedal extends the
    # natural sounding interval to the pedal lift.
    assert c.finger_interval() == (Pos(0, 0), Pos(2, 0))
    assert c.sounding_interval() == (Pos(0, 0), Pos(5, 0))
    assert c.end_kind == "pedal"
    # G: no pedal on channel B, so it stops at its key release.
    assert g.finger_interval() == (Pos(3, 0), Pos(4, 0))
    assert g.sounding_interval() == (Pos(3, 0), Pos(4, 0))
    assert g.end_kind == "release"
    assert perf.orphan_releases == ()
    # Both interval kinds come from the same product.
    assert [r["end"] for r in perf.finger_intervals()] == [Pos(2, 0), Pos(4, 0)]
    assert [r["end"] for r in perf.sounding_intervals()] == [Pos(5, 0), Pos(4, 0)]


def test_same_beat_messages_apply_in_sequence_order():
    # Key release at (2, 0) happens while the pedal is down; the pedal lift
    # at (2, 1) ends the voice there.
    perf = make_performance(
        [
            (0, 0, CH_A, KEY_DOWN, C4),
            (1, 0, CH_A, PEDAL_DOWN, None),
            (2, 0, CH_A, KEY_UP, C4),
            (2, 1, CH_A, PEDAL_UP, None),
        ]
    )
    assert perf.voices[0].sounding_interval() == (Pos(0, 0), Pos(2, 1))
    assert perf.voices[0].end_kind == "pedal"

    # Same positions, opposite order: the release at (2, 0) happens before
    # the pedal goes down, so the voice stops at the release.
    perf = make_performance(
        [
            (0, 0, CH_A, KEY_DOWN, C4),
            (2, 0, CH_A, KEY_UP, C4),
            (2, 1, CH_A, PEDAL_DOWN, None),
            (3, 0, CH_A, PEDAL_UP, None),
        ]
    )
    assert perf.voices[0].sounding_interval() == (Pos(0, 0), Pos(2, 0))
    assert perf.voices[0].end_kind == "release"


def test_pedal_lift_only_ends_released_voices():
    # v0 released under the pedal; v1 still held when the pedal lifts: v1
    # keeps sounding and ends at its own later release.
    perf = make_performance(
        [
            (0, 0, CH_A, KEY_DOWN, C4),
            (0, 1, CH_A, KEY_DOWN, G4),
            (1, 0, CH_A, PEDAL_DOWN, None),
            (2, 0, CH_A, KEY_UP, C4),
            (3, 0, CH_A, PEDAL_UP, None),
            (4, 0, CH_A, KEY_UP, G4),
        ]
    )
    v0, v1 = perf.voices
    assert v0.sounding_interval() == (Pos(0, 0), Pos(3, 0))
    assert v0.end_kind == "pedal"
    assert v1.sounding_interval() == (Pos(0, 1), Pos(4, 0))
    assert v1.end_kind == "release"


def test_repress_identity_with_interleaved_releases(keys_product, tmp_path):
    # Same channel, same pitch, pressed again while still held; releases
    # interleave with a third press.  Releases pair FIFO with held presses.
    # The check goes through the actual wire messages, not hand-built state.
    messages = [
        (0, 0, CH_A, KEY_DOWN, C4),
        (1, 0, CH_A, KEY_DOWN, C4),
        (2, 0, CH_A, KEY_UP, C4),
        (2, 1, CH_A, KEY_DOWN, C4),
        (3, 0, CH_A, KEY_UP, C4),
        (4, 0, CH_A, KEY_UP, C4),
    ]
    with Repository(tmp_path / "keys.sqlite") as repo:
        repo.publish_product(KEYS_SCHEMA_TEXT)
        import_id = repo.create_import(keys_product.product_id, source_name="identity")
        repo.ingest_bytes(import_id, stream_of(messages))
        perf = performance_from_import(repo, import_id)

    assert len(perf.voices) == 3
    v0, v1, v2 = perf.voices
    # Three independent identities for the same pitch.
    assert {v.voice_id for v in perf.voices} == {0, 1, 2}
    assert {v.pitch for v in perf.voices} == {C4}
    # Identity-aware pairing: a pitch-set comparison could not distinguish
    # these voices, but their intervals are all different.
    assert v0.finger_interval() == (Pos(0, 0), Pos(2, 0))
    assert v1.finger_interval() == (Pos(1, 0), Pos(3, 0))
    assert v2.finger_interval() == (Pos(2, 1), Pos(4, 0))
    intervals = {v.voice_id: v.finger_interval() for v in perf.voices}
    assert len(set(intervals.values())) == 3
    # And the pairing is observable in the replay: with two slots, which
    # identity is audible at each position follows the FIFO-paired ends.
    result = replay(perf, 2, {})
    assert [p.voice_id for p in result.audible_at(Pos(2, 0))] == [1]
    assert [p.voice_id for p in result.audible_at(Pos(2, 1))] == [1, 2]
    assert [p.voice_id for p in result.audible_at(Pos(3, 0))] == [2]


def test_orphan_release_reported_with_source(keys_product, tmp_path):
    messages = [
        (0, 0, CH_A, KEY_DOWN, C4),
        (1, 0, CH_A, KEY_UP, C4),
        (2, 0, CH_A, KEY_UP, C4),  # nothing held any more: extra release
        (3, 0, CH_B, KEY_UP, G4),  # never held at all: extra release
    ]
    stream = stream_of(messages)
    with Repository(tmp_path / "keys.sqlite") as repo:
        repo.publish_product(KEYS_SCHEMA_TEXT)
        import_id = repo.create_import(keys_product.product_id, source_name="orphan")
        repo.ingest_bytes(import_id, stream)
        perf = performance_from_import(repo, import_id)

    assert len(perf.voices) == 1
    assert perf.voices[0].sounding_interval() == (Pos(0, 0), Pos(1, 0))
    assert len(perf.orphan_releases) == 2
    first, second = perf.orphan_releases
    # The extra release reports the message it came from: record ordinal and
    # absolute stream offsets.
    assert (first.channel, first.pitch, first.pos) == (CH_A, C4, Pos(2, 0))
    assert first.record == 2
    assert stream[first.start : first.end] == key_msg(*messages[2])
    assert (second.channel, second.pitch, second.pos) == (CH_B, G4, Pos(3, 0))
    assert second.record == 3
    assert stream[second.start : second.end] == key_msg(*messages[3])


def test_cutoff_marks_right_end_undefined():
    perf = make_performance(
        [
            (0, 0, CH_A, KEY_DOWN, C4),  # still held when the recording stops
            (1, 0, CH_B, KEY_DOWN, G4),
            (2, 0, CH_B, PEDAL_DOWN, None),
            (3, 0, CH_B, KEY_UP, G4),  # sustained, pedal never lifted
        ]
    )
    held, sustained = perf.voices
    assert held.finger_interval() == (Pos(0, 0), None)
    assert held.sounding_interval() == (Pos(0, 0), None)
    assert sustained.finger_interval() == (Pos(1, 0), Pos(3, 0))
    assert sustained.sounding_interval() == (Pos(1, 0), None)
    assert held.end_kind is None and sustained.end_kind is None

    result = replay(perf, 2, {})
    assert [p.status for p in result.playbacks] == [OPEN, OPEN]
    assert [p.voice_id for p in result.audible_at(Pos(99, 0))] == [0, 1]


def test_message_validation():
    def build(messages):
        return build_performance(
            [Message(beat=b, seq=s, channel=c, action=a, pitch=p, record=i)
             for i, (b, s, c, a, p) in enumerate(messages)]
        )

    with pytest.raises(KeysError) as excinfo:
        build([(0, 0, CH_A, 9, C4)])
    assert excinfo.value.code == "unknown_action"

    with pytest.raises(KeysError) as excinfo:
        build([(0, 0, CH_A, KEY_DOWN, None)])
    assert excinfo.value.code == "missing_pitch"

    with pytest.raises(KeysError) as excinfo:
        build([(0, 0, CH_A, PEDAL_DOWN, C4)])
    assert excinfo.value.code == "unexpected_pitch"

    with pytest.raises(KeysError) as excinfo:
        build([(-1, 0, CH_A, KEY_DOWN, C4)])
    assert excinfo.value.code == "bad_position"


# ---------------------------------------------------------------------------
# Player


def test_replay_single_voice_a_priority_g_silent(main_performance):
    result = replay(main_performance, 1, {CH_A: 2, CH_B: 1})
    c, g = result.playbacks
    assert (c.status, c.start, c.end) == (NATURAL, Pos(0, 0), Pos(5, 0))
    assert (g.status, g.start, g.end) == (NO_VOICE, None, None)
    assert g.competitors == (0,)
    # G is never audible; C sounds from beat 0 up to (not including) beat 5.
    assert [p.voice_id for p in result.audible_at(Pos(3, 0))] == [0]
    assert [p.voice_id for p in result.audible_at(Pos(4, 0))] == [0]
    assert result.audible_at(Pos(5, 0)) == []


def test_replay_single_voice_b_priority_c_truncated_no_revival(main_performance):
    result = replay(main_performance, 1, {CH_A: 1, CH_B: 2})
    c, g = result.playbacks
    assert (c.status, c.start, c.end) == (TRUNCATED, Pos(0, 0), Pos(3, 0))
    assert c.evicted_by == 1
    assert (g.status, g.start, g.end) == (NATURAL, Pos(3, 0), Pos(4, 0))
    # The slot frees at beat 4, but the truncated C tail never revives.
    assert [p.voice_id for p in result.audible_at(Pos(3, 0))] == [1]
    assert result.audible_at(Pos(4, 0)) == []
    assert result.audible_at(Pos(5, 0)) == []


def test_replay_variant_pedal_up_before_g(keys_product, tmp_path):
    stream = stream_of(VARIANT_MESSAGES)
    with Repository(tmp_path / "keys.sqlite") as repo:
        repo.publish_product(KEYS_SCHEMA_TEXT)
        import_id = repo.create_import(keys_product.product_id, source_name="variant")
        repo.ingest_bytes(import_id, stream)
        perf = performance_from_import(repo, import_id)

    # Same-beat order: the pedal lift at (3, 0) ends C before G's press at
    # (3, 1), so the natural stop frees the slot at its actual event.
    assert perf.voices[0].sounding_interval() == (Pos(0, 0), Pos(3, 0))
    result = replay(perf, 1, {CH_A: 2, CH_B: 1})
    c, g = result.playbacks
    assert (c.status, c.end) == (NATURAL, Pos(3, 0))
    assert (g.status, g.start, g.end) == (NATURAL, Pos(3, 1), Pos(4, 0))
    # G sounds until beat 4.
    assert [p.voice_id for p in result.audible_at(Pos(3, 1))] == [1]
    assert result.audible_at(Pos(3, 0)) == []
    assert result.audible_at(Pos(4, 0)) == []


def test_replay_equal_priority_evicts_earliest_started():
    perf = make_performance(
        [
            (0, 0, CH_A, KEY_DOWN, C4),
            (1, 0, CH_B, KEY_DOWN, G4),
            (2, 0, CH_B, KEY_UP, G4),
            (5, 0, CH_A, KEY_UP, C4),
        ]
    )
    result = replay(perf, 1, {CH_A: 1, CH_B: 1})
    c, g = result.playbacks
    # Same priority: the earliest-started voice loses the slot.
    assert (c.status, c.end) == (TRUNCATED, Pos(1, 0))
    assert (g.status, g.start, g.end) == (NATURAL, Pos(1, 0), Pos(2, 0))


def test_replay_two_voices_both_sound():
    perf = make_performance(
        [
            (0, 0, CH_A, KEY_DOWN, C4),
            (1, 0, CH_B, KEY_DOWN, G4),
            (2, 0, CH_A, KEY_UP, C4),
            (3, 0, CH_B, KEY_UP, G4),
        ]
    )
    result = replay(perf, 2, {CH_A: 1, CH_B: 2})
    assert [p.status for p in result.playbacks] == [NATURAL, NATURAL]
    assert [p.voice_id for p in result.audible_at(Pos(1, 0))] == [0, 1]
    assert [p.voice_id for p in result.audible_at(Pos(2, 0))] == [1]


def test_replay_rejects_bad_voice_count(main_performance):
    with pytest.raises(PlayerError):
        replay(main_performance, 0, {})


def test_player_cannot_change_the_product(main_performance, tmp_path):
    before = main_performance.to_json()
    replay(main_performance, 1, {CH_A: 2, CH_B: 1})
    replay(main_performance, 1, {CH_A: 1, CH_B: 2})
    replay(main_performance, 3, {})
    assert main_performance.to_json() == before
    with pytest.raises(dataclasses.FrozenInstanceError):
        main_performance.voices = ()  # type: ignore[misc]

    # Two different player settings over the same product file give two
    # different results; the file itself is untouched.
    path = tmp_path / "performance.json"
    main_performance.save(path)
    reloaded = load_performance(path)
    assert reloaded == main_performance
    a = replay(reloaded, 1, {CH_A: 2, CH_B: 1})
    b = replay(reloaded, 1, {CH_A: 1, CH_B: 2})
    assert a.playbacks[1].status == NO_VOICE
    assert b.playbacks[0].status == TRUNCATED
    assert path.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# Product serialisation


def test_product_json_roundtrip(main_performance, tmp_path):
    perf = main_performance
    assert Performance.from_json(perf.to_json()) == perf
    path = tmp_path / "performance.json"
    perf.save(path)
    assert load_performance(path) == perf
    # The file carries both interval kinds plus the orphan-release report.
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["kind"] == "keys.performance"
    assert doc["voices"][0]["end"] == [5, 0]
    assert doc["voices"][0]["release"] == [2, 0]
    assert doc["source"]["message_count"] == 6

    with pytest.raises(KeysError):
        Performance.from_json('{"kind": "other"}')
    with pytest.raises(KeysError):
        Performance.from_json("not json")


def test_coexistence_with_telemetry_product(keys_product, tmp_path):
    """The new record class lives next to the ordinary telemetry messages;
    the existing usage of the telemetry product is unchanged."""
    with Repository(tmp_path / "mixed.sqlite") as repo:
        telemetry = repo.publish_product(TELEMETRY_SCHEMA_TEXT)
        keys = repo.publish_product(KEYS_SCHEMA_TEXT)
        assert telemetry.product_id != keys.product_id

        tel_import = repo.create_import(telemetry.product_id, source_name="telemetry")
        repo.ingest_bytes(
            tel_import,
            encode_record(
                {"station_id": -1, "observed_at": 7, "sensor": {"kind": 1}}, telemetry
            ),
        )
        keys_import = repo.create_import(keys.product_id, source_name="keys")
        repo.ingest_bytes(keys_import, stream_of(MAIN_MESSAGES))

        rows = {r["path"]: r for r in repo.relation_rows(tel_import)}
        assert rows["station_id"]["int_value"] == -1
        assert rows["sensor.kind"]["int_value"] == 1

        perf = performance_from_import(repo, keys_import)
        assert len(perf.voices) == 2
        with pytest.raises(KeysError) as excinfo:
            performance_from_import(repo, tel_import)
        assert excinfo.value.code == "wrong_product"


# ---------------------------------------------------------------------------
# Demo and CLI entry points


def test_keydemo_truth(tmp_path):
    """The demo's report is checked against independently derived truth."""
    report = keydemo_run(tmp_path)
    assert (tmp_path / "keys.bin").read_bytes() == stream_of(MAIN_MESSAGES)
    assert (tmp_path / "keys_variant.bin").read_bytes() == stream_of(VARIANT_MESSAGES)
    perf = load_performance(tmp_path / "performance.json")
    assert [v.sounding_interval() for v in perf.voices] == [
        (Pos(0, 0), Pos(5, 0)),
        (Pos(3, 0), Pos(4, 0)),
    ]
    replays = report["replays"]
    assert replays["main_A_priority"][0]["status"] == "natural"
    assert replays["main_A_priority"][1]["status"] == "no_voice"
    assert replays["main_B_priority"][0]["status"] == "truncated"
    assert replays["main_B_priority"][0]["audible"] == [[0, 0], [3, 0]]
    assert replays["main_B_priority"][1]["audible"] == [[3, 0], [4, 0]]
    assert replays["variant_A_priority"][1]["audible"] == [[3, 1], [4, 0]]
    assert report["audible_checks"] == {
        "main_A_priority@beat3": [0],
        "main_B_priority@beat3": [1],
        "main_B_priority@beat4": [],
        "variant_A_priority@beat3_seq1": [1],
    }
    saved = json.loads((tmp_path / "replay.json").read_text(encoding="utf-8"))
    assert saved["main_B_priority"]["playbacks"][0]["status"] == "truncated"


def test_cli_keys_flow(keys_product, tmp_path, capsys):
    """From the existing encoding, through import, to the replay entry that
    reads the product file on its own."""
    product_json = tmp_path / "keys_product.json"
    stream_path = tmp_path / "keys.bin"
    db = tmp_path / "keys.sqlite"
    perf_path = tmp_path / "performance.json"

    assert cli_main(["compile", str(ROOT / "schemas" / "keys_v1.json"), "-o", str(product_json)]) == 0
    stream_path.write_bytes(stream_of(MAIN_MESSAGES))
    assert cli_main(["ingest", str(product_json), str(stream_path), "--db", str(db)]) == 0

    assert cli_main(["keys-export", str(db), "--import-id", "1", "-o", str(perf_path)]) == 0
    capsys.readouterr()  # drain compile/ingest/export chatter
    perf = load_performance(perf_path)
    assert [v.sounding_interval() for v in perf.voices] == [
        (Pos(0, 0), Pos(5, 0)),
        (Pos(3, 0), Pos(4, 0)),
    ]

    assert cli_main(["keys-report", str(perf_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert [r["end"] for r in report["finger_intervals"]] == [[2, 0], [4, 0]]
    assert [r["end"] for r in report["sounding_intervals"]] == [[5, 0], [4, 0]]

    assert (
        cli_main(
            [
                "keys-replay",
                str(perf_path),
                "--voices",
                "1",
                "--priority",
                "0=1",
                "--priority",
                "1=2",
                "--at",
                "3:0",
                "--at",
                "4:0",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["playbacks"][0]["status"] == "truncated"
    assert out["playbacks"][1]["status"] == "natural"
    assert out["audible_at"] == {"3:0": [1], "4:0": []}


def test_cli_keys_errors(keys_product, tmp_path, capsys):
    """The replay/export entries report Telber errors on stderr with exit 2."""
    db = tmp_path / "mixed.sqlite"
    with Repository(db) as repo:
        telemetry = repo.publish_product(TELEMETRY_SCHEMA_TEXT)
        import_id = repo.create_import(telemetry.product_id, source_name="telemetry")
        repo.ingest_bytes(
            import_id,
            encode_record(
                {"station_id": 1, "observed_at": 2, "sensor": {"kind": 0}}, telemetry
            ),
        )
    assert cli_main(["keys-export", str(db), "--import-id", str(import_id)]) == 2
    err = json.loads(capsys.readouterr().err)
    assert err["code"] == "wrong_product"

    bad = tmp_path / "bad.json"
    bad.write_text('{"kind": "other"}', encoding="utf-8")
    assert cli_main(["keys-report", str(bad)]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "invalid_product"

    perf_path = tmp_path / "performance.json"
    make_performance([(0, 0, CH_A, KEY_DOWN, C4)]).save(perf_path)
    assert cli_main(["keys-replay", str(perf_path), "--voices", "0"]) == 2
    assert json.loads(capsys.readouterr().err)["code"] == "bad_voice_count"
