#!/usr/bin/env python3
"""Small command-line interface over the library.

Examples::

    python3 -m telber.cli compile schemas/telemetry_v1.json -o product.json
    python3 -m telber.cli check product.json record.bin
    python3 -m telber.cli encode product.json values.json -o record.bin
    python3 -m telber.cli ingest product.json telemetry.bin -o telemetry.sqlite
    python3 -m telber.cli report telemetry.sqlite

Keyboard/pedal performance products::

    python3 -m telber.cli ingest keys_product.json keys.bin --db keys.sqlite
    python3 -m telber.cli keys-export keys.sqlite --import-id 1 -o performance.json
    python3 -m telber.cli keys-report performance.json
    python3 -m telber.cli keys-replay performance.json --voices 1 --priority 0=2 --priority 1=1 --at 3:0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from telber.errors import TelberError  # noqa: E402
from telber.keys import Pos, load_performance, performance_from_import  # noqa: E402
from telber.player import replay  # noqa: E402
from telber.repository import Repository  # noqa: E402
from telber.service import CompileService  # noqa: E402


def _load_product(path: str):
    return CompileService().load_artifact(Path(path).read_text(encoding="utf-8"))


def cmd_compile(args) -> int:
    service = CompileService()
    product = service.compile(Path(args.schema).read_text(encoding="utf-8"))
    text = service.artifact(product)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"product {product.product_id} -> {args.out}")
    else:
        print(text, end="")
    return 0


def cmd_check(args) -> int:
    service = CompileService()
    product = _load_product(args.product)
    data = Path(args.record).read_bytes()
    try:
        result = service.check(product, data)
    except TelberError as exc:
        print(json.dumps(exc.as_dict(), indent=2), file=sys.stderr)
        return 2
    print(json.dumps(result.as_dict(), indent=2, default=str))
    return 0


def _expand_hex(values):
    """JSON has no bytes type; accept ``{"$hex": "00ff"}`` for OCTET STRINGs."""
    if isinstance(values, dict):
        if set(values) == {"$hex"} and isinstance(values["$hex"], str):
            return bytes.fromhex(values["$hex"])
        return {k: _expand_hex(v) for k, v in values.items()}
    if isinstance(values, list):
        return [_expand_hex(v) for v in values]
    return values


def cmd_encode(args) -> int:
    service = CompileService()
    product = _load_product(args.product)
    values = _expand_hex(json.loads(Path(args.values).read_text(encoding="utf-8")))
    data = service.encode(product, values)
    if args.indefinite:
        data = service.encode_indefinite(product, values)
    if args.out:
        Path(args.out).write_bytes(data)
        print(f"{len(data)} bytes -> {args.out}")
    else:
        sys.stdout.buffer.write(data)
    return 0


def cmd_ingest(args) -> int:
    product = _load_product(args.product)
    with Repository(args.db) as repo:
        repo.publish_product(product.schema_dict())
        import_id = repo.create_import(product.product_id, source_name=args.source)
        try:
            info = repo.ingest_file(import_id, args.source, block_size=args.block_size)
        except TelberError as exc:
            print(json.dumps(exc.as_dict(), indent=2), file=sys.stderr)
            print(f"import {getattr(exc, 'info', None)}", file=sys.stderr)
            return 2
    print(json.dumps(info.as_dict(), indent=2))
    return 0


def cmd_report(args) -> int:
    out = []
    with Repository(args.db) as repo:
        import_ids = [
            int(r["import_id"])
            for r in repo.conn.execute("SELECT import_id FROM imports ORDER BY import_id")
        ]
        if args.import_id is not None:
            import_ids = [i for i in import_ids if i == args.import_id]
        for import_id in import_ids:
            entry = {"info": repo.import_info(import_id).as_dict()}
            if args.relation:
                entry["records"] = repo.records(import_id)
                entry["relation"] = repo.relation_rows(import_id)
                entry["source_ranges"] = repo.source_ranges(import_id)
            out.append(entry)
    print(json.dumps(out, indent=2, default=str))
    return 0


def cmd_keys_export(args) -> int:
    """Build the sustained-note performance product from a keys.event import."""
    try:
        with Repository(args.db) as repo:
            performance = performance_from_import(repo, args.import_id)
    except TelberError as exc:
        print(json.dumps(exc.as_dict(), indent=2), file=sys.stderr)
        return 2
    text = performance.to_json()
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"performance product ({len(performance.voices)} voices) -> {args.out}")
    else:
        print(text, end="")
    return 0


def _interval_rows(performance, intervals):
    return [
        {
            "voice": row["voice"],
            "channel": row["channel"],
            "pitch": row["pitch"],
            "start": list(row["start"]),
            "end": None if row["end"] is None else list(row["end"]),
        }
        for row in intervals
    ]


def cmd_keys_report(args) -> int:
    """Show both interval kinds of a performance product read from its file."""
    try:
        performance = load_performance(args.performance)
    except TelberError as exc:
        print(json.dumps(exc.as_dict(), indent=2), file=sys.stderr)
        return 2
    out = {
        "voices": len(performance.voices),
        "finger_intervals": _interval_rows(performance, performance.finger_intervals()),
        "sounding_intervals": _interval_rows(performance, performance.sounding_intervals()),
        "orphan_releases": [o.as_dict() for o in performance.orphan_releases],
    }
    print(json.dumps(out, indent=2))
    return 0


def _parse_priority(text: str) -> tuple[int, int]:
    try:
        channel, prio = text.split("=", 1)
        return int(channel), int(prio)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"priority must look like CHANNEL=PRIO, got {text!r}"
        ) from None


def _parse_pos(text: str) -> Pos:
    sep = ":" if ":" in text else ","
    try:
        beat, seq = text.split(sep, 1)
        return Pos(int(beat), int(seq))
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"position must look like BEAT:SEQ, got {text!r}"
        ) from None


def cmd_keys_replay(args) -> int:
    """Replay a performance product file under a finite-voice player."""
    try:
        performance = load_performance(args.performance)
        result = replay(performance, args.voices, dict(args.priority))
    except TelberError as exc:
        print(json.dumps(exc.as_dict(), indent=2), file=sys.stderr)
        return 2
    out = result.as_dict()
    if args.at:
        out["audible_at"] = {
            f"{pos.beat}:{pos.seq}": [p.voice_id for p in result.audible_at(pos)]
            for pos in args.at
        }
    print(json.dumps(out, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="telber", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("compile", help="compile a JSON schema into a product artifact")
    p.add_argument("schema")
    p.add_argument("-o", "--out")
    p.set_defaults(func=cmd_compile)

    p = sub.add_parser("check", help="decode one record under a product")
    p.add_argument("product")
    p.add_argument("record")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("encode", help="encode JSON values into one record")
    p.add_argument("product")
    p.add_argument("values")
    p.add_argument("-o", "--out")
    p.add_argument("--indefinite", action="store_true")
    p.set_defaults(func=cmd_encode)

    p = sub.add_parser("ingest", help="ingest a stream into a SQLite fact repository")
    p.add_argument("product")
    p.add_argument("source")
    p.add_argument("--db", default="telemetry.sqlite")
    p.add_argument("--block-size", type=int, default=65536)
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("report", help="list imports in a repository")
    p.add_argument("db")
    p.add_argument("--import-id", type=int, default=None)
    p.add_argument("--relation", action="store_true", help="include record/field facts")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser(
        "keys-export",
        help="derive the sustained-note performance product from a keys.event import",
    )
    p.add_argument("db")
    p.add_argument("--import-id", type=int, required=True)
    p.add_argument("-o", "--out")
    p.set_defaults(func=cmd_keys_export)

    p = sub.add_parser(
        "keys-report",
        help="show finger-held and natural sounding intervals of a performance product",
    )
    p.add_argument("performance")
    p.set_defaults(func=cmd_keys_report)

    p = sub.add_parser(
        "keys-replay",
        help="replay a performance product file under a finite-voice player",
    )
    p.add_argument("performance")
    p.add_argument("--voices", type=int, required=True, help="total number of voice slots")
    p.add_argument(
        "--priority",
        type=_parse_priority,
        action="append",
        default=[],
        metavar="CHANNEL=PRIO",
        help="per-channel priority (higher wins); repeatable, unlisted channels are 0",
    )
    p.add_argument(
        "--at",
        type=_parse_pos,
        action="append",
        default=[],
        metavar="BEAT:SEQ",
        help="also list the voices audible at this position; repeatable",
    )
    p.set_defaults(func=cmd_keys_replay)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
