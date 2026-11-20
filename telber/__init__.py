"""telber — a bounded BER-subset type directory, codec and telemetry repository.

Modules
-------
``ber``         tag/length wire primitives with minimal-encoding enforcement
``schema``      JSON schema directory compiler (immutable, content-addressed product)
``codec``       record decoder/encoder with absolute byte provenance
``stream``      bounded streaming record delimitation over the original octets
``repository``  SQLite product registry + per-record field/relation fact store
``keys``        keyboard/pedal messages compiled into a sustained-note product
``player``      finite-voice replay of a performance product
``errors``      error hierarchy carrying absolute offsets

See ``README.md`` for the exact supported encoding subset and the design rules.
"""

from .ber import Tag
from .codec import Record, Value, decode_record, encode_record, encode_record_indefinite
from .errors import (
    BerError,
    DecodeError,
    EncodeError,
    ImportStateError,
    KeysError,
    PlayerError,
    ProductRevisionError,
    SchemaError,
    TelberError,
    TruncatedError,
)
from .keys import (
    KEY_DOWN,
    KEY_UP,
    PEDAL_DOWN,
    PEDAL_UP,
    Message,
    OrphanRelease,
    Performance,
    Pos,
    Voice,
    build_performance,
    load_messages,
    load_performance,
    performance_from_import,
)
from .player import Playback, ReplayResult, replay
from .repository import ImportInfo, Repository
from .schema import Field, Product, compile_product
from .stream import RecordScanner, StreamRecord, scan_records

__all__ = [
    "Tag",
    "Field",
    "Product",
    "compile_product",
    "Record",
    "Value",
    "decode_record",
    "encode_record",
    "encode_record_indefinite",
    "RecordScanner",
    "StreamRecord",
    "scan_records",
    "Repository",
    "ImportInfo",
    "Pos",
    "Message",
    "Voice",
    "OrphanRelease",
    "Performance",
    "build_performance",
    "load_messages",
    "load_performance",
    "performance_from_import",
    "KEY_DOWN",
    "KEY_UP",
    "PEDAL_DOWN",
    "PEDAL_UP",
    "Playback",
    "ReplayResult",
    "replay",
    "TelberError",
    "BerError",
    "TruncatedError",
    "DecodeError",
    "SchemaError",
    "EncodeError",
    "ProductRevisionError",
    "ImportStateError",
    "KeysError",
    "PlayerError",
]
