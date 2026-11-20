#!/usr/bin/env python3
"""End-to-end keyboard/pedal demo: encode a synthetic recording with the
existing codec, ingest it, derive the sustained-note performance product, and
replay it under a finite-voice player.

Run from the repository root::

    python3 -m telber.keydemo            # in-memory, prints a report
    python3 tools/keys_demo.py --out DIR # writes keys.bin, keys.sqlite,
                                         # performance.json, replay.json

The synthetic recording (channel A=0, B=1; pitches C4=60, G4=67)::

    beat 0  channel A  key down C
    beat 1  channel A  pedal down
    beat 2  channel A  key up C      (pedal holds C sounding)
    beat 3  channel B  key down G
    beat 4  channel B  key up G
    beat 5  channel A  pedal up      (C finally stops)

With a single voice and channel A prioritised, G never gets a slot.  With
channel B prioritised, C is truncated at beat 3 and stays silent after G
ends.  A variant recording moves the pedal lift to beat 3 *before* the G
press (same-beat messages act in sequence order), so G sounds until beat 4.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telber.keys import (  # noqa: E402
    KEY_DOWN,
    KEY_UP,
    PEDAL_DOWN,
    PEDAL_UP,
    Pos,
    load_performance,
    performance_from_import,
)
from telber.player import replay  # noqa: E402
from telber.repository import Repository  # noqa: E402
from telber.service import CompileService  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_PATH = ROOT / "schemas" / "keys_v1.json"

CH_A, CH_B = 0, 1
C4, G4 = 60, 67
CHANNEL_NAMES = {CH_A: "A", CH_B: "B"}
NOTE_NAMES = {C4: "C4", G4: "G4"}

# The main synthetic recording: (beat, seq, channel, action, pitch).
MAIN_MESSAGES = [
    (0, 0, CH_A, KEY_DOWN, C4),
    (1, 0, CH_A, PEDAL_DOWN, None),
    (2, 0, CH_A, KEY_UP, C4),
    (3, 0, CH_B, KEY_DOWN, G4),
    (4, 0, CH_B, KEY_UP, G4),
    (5, 0, CH_A, PEDAL_UP, None),
]

# Variant: the pedal lift moves to beat 3 just before the G press.
VARIANT_MESSAGES = [
    (0, 0, CH_A, KEY_DOWN, C4),
    (1, 0, CH_A, PEDAL_DOWN, None),
    (2, 0, CH_A, KEY_UP, C4),
    (3, 0, CH_A, PEDAL_UP, None),
    (3, 1, CH_B, KEY_DOWN, G4),
    (4, 0, CH_B, KEY_UP, G4),
]


def encode_stream(service, product, messages) -> bytes:
    """Encode messages with the existing record codec (the same path
    ``telber.cli encode`` uses) and concatenate them into one stream."""
    parts = []
    for beat, seq, channel, action, pitch in messages:
        values = {"beat": beat, "seq": seq, "channel": channel, "action": action}
        if pitch is not None:
            values["pitch"] = pitch
        parts.append(service.encode(product, values))
    return b"".join(parts)


def _intervals_view(performance, intervals):
    return [
        {
            "voice": row["voice"],
            "channel": CHANNEL_NAMES.get(row["channel"], row["channel"]),
            "pitch": NOTE_NAMES.get(row["pitch"], row["pitch"]),
            "start": list(row["start"]),
            "end": None if row["end"] is None else list(row["end"]),
        }
        for row in intervals
    ]


def _replay_view(result):
    out = []
    for p in result.playbacks:
        out.append(
            {
                "voice": p.voice_id,
                "channel": CHANNEL_NAMES.get(p.channel, p.channel),
                "pitch": NOTE_NAMES.get(p.pitch, p.pitch),
                "audible": [
                    None if p.start is None else list(p.start),
                    None if p.end is None else list(p.end),
                ],
                "status": p.status,
                "reason": p.reason,
            }
        )
    return out


def run(out: Path | None = None) -> dict:
    schema_text = SCHEMA_PATH.read_text(encoding="utf-8")
    service = CompileService()
    product = service.compile(schema_text)

    main_stream = encode_stream(service, product, MAIN_MESSAGES)
    variant_stream = encode_stream(service, product, VARIANT_MESSAGES)

    db_path = ":memory:" if out is None else str(out / "keys.sqlite")
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / "product.json").write_text(service.artifact(product), encoding="utf-8")
        (out / "keys.bin").write_bytes(main_stream)
        (out / "keys_variant.bin").write_bytes(variant_stream)

    report: dict = {
        "product_id": product.product_id,
        "schema": product.schema_name,
        "schema_version": product.version,
    }

    with Repository(db_path) as repo:
        repo.publish_product(schema_text)
        performances = {}
        for name, stream in (("main", main_stream), ("variant", variant_stream)):
            import_id = repo.create_import(product.product_id, source_name=f"demo/{name}.bin")
            info = repo.ingest_bytes(import_id, stream, block_size=5)
            assert info.status == "complete"
            performances[name] = performance_from_import(repo, import_id)

    # The replay entry reads the product back from its file, separately from
    # the repository; the player never touches the telemetry store.
    with ExitStack() as stack:
        product_dir = out
        if product_dir is None:
            product_dir = Path(stack.enter_context(tempfile.TemporaryDirectory()))
        perf_path = product_dir / "performance.json"
        variant_path = product_dir / "performance_variant.json"
        performances["main"].save(perf_path)
        performances["variant"].save(variant_path)
        main_perf = load_performance(perf_path)
        variant_perf = load_performance(variant_path)

    report["performance"] = {
        "finger_intervals": _intervals_view(main_perf, main_perf.finger_intervals()),
        "sounding_intervals": _intervals_view(main_perf, main_perf.sounding_intervals()),
        "orphan_releases": [o.as_dict() for o in main_perf.orphan_releases],
    }

    one_voice = 1
    replays = {
        # Single voice, channel A prioritised: G never gets the slot.
        "main_A_priority": replay(main_perf, one_voice, {CH_A: 2, CH_B: 1}),
        # Single voice, channel B prioritised: C is truncated at beat 3 and
        # stays silent after G ends (a truncated tail never revives).
        "main_B_priority": replay(main_perf, one_voice, {CH_A: 1, CH_B: 2}),
        # Pedal lifted just before the G press: the slot frees at (3, 0) and
        # G sounds until its key release at beat 4.
        "variant_A_priority": replay(variant_perf, one_voice, {CH_A: 2, CH_B: 1}),
    }
    report["replays"] = {name: _replay_view(r) for name, r in replays.items()}
    report["audible_checks"] = {
        "main_A_priority@beat3": [p.voice_id for p in replays["main_A_priority"].audible_at(Pos(3, 0))],
        "main_B_priority@beat3": [p.voice_id for p in replays["main_B_priority"].audible_at(Pos(3, 0))],
        "main_B_priority@beat4": [p.voice_id for p in replays["main_B_priority"].audible_at(Pos(4, 0))],
        "variant_A_priority@beat3_seq1": [
            p.voice_id for p in replays["variant_A_priority"].audible_at(Pos(3, 1))
        ],
    }
    if out is not None:
        (out / "replay.json").write_text(
            json.dumps({name: r.as_dict() for name, r in replays.items()}, indent=2),
            encoding="utf-8",
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="write artifacts here")
    args = parser.parse_args(argv)
    report = run(args.out)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
