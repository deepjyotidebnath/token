"""
Append-only JSONL storage for trace records.

JSONL (not SQLite) is used deliberately: it's human-readable, greppable,
diffable, safe under concurrent appends (each write is one line), and
needs zero setup. For very high call volumes, swap this module for a
SQLite or database-backed implementation without changing the Tracer API.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

DEFAULT_LOG_PATH = os.environ.get("TOKENTRACER_LOG", "tokentracer.jsonl")


@dataclass
class TraceRecord:
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_usd: float
    tag: str | None = None
    duration_ms: float | None = None
    metadata: dict[str, Any] | None = None


def append_record(record: TraceRecord, path: str | Path = DEFAULT_LOG_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")


def read_records(path: str | Path = DEFAULT_LOG_PATH) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def clear_log(path: str | Path = DEFAULT_LOG_PATH) -> None:
    path = Path(path)
    if path.exists():
        path.unlink()
