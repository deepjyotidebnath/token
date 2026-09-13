"""
Unit tests for tokentracer. Pure stdlib — no network, no API keys needed.
Run with: pytest tests/
"""
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tokentracer.pricing import cost_for
from tokentracer.storage import clear_log, read_records
from tokentracer.tracer import log_usage, trace, trace_call


def _tmp_path() -> str:
    return tempfile.mktemp(suffix=".jsonl")


def test_cost_for_known_model():
    cost = cost_for("claude-sonnet-5", input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 3.00 + 15.00


def test_cost_for_unknown_model_is_zero_not_crash():
    cost = cost_for("some-made-up-model", input_tokens=1000, output_tokens=1000)
    assert cost == 0.0


def test_log_usage_writes_record():
    path = _tmp_path()
    record = log_usage(model="claude-haiku-4-5-20251001", input_tokens=100, output_tokens=50, tag="test", path=path)

    assert record.total_tokens == 150
    assert record.cost_usd > 0

    records = list(read_records(path))
    assert len(records) == 1
    assert records[0]["tag"] == "test"
    clear_log(path)


def test_trace_call_context_manager():
    path = _tmp_path()
    with trace_call(tag="ctx", path=path) as t:
        t.report(model="claude-sonnet-5", input_tokens=10, output_tokens=20)

    records = list(read_records(path))
    assert len(records) == 1
    assert records[0]["input_tokens"] == 10
    assert records[0]["output_tokens"] == 20
    clear_log(path)


def test_trace_call_warns_if_unreported(capsys):
    path = _tmp_path()
    with trace_call(tag="forgot", path=path):
        pass  # never called t.report(...)

    captured = capsys.readouterr()
    assert "warning" in captured.out.lower()

    records = list(read_records(path))
    assert len(records) == 0


def test_trace_decorator_with_anthropic_shaped_response():
    @dataclass
    class Usage:
        input_tokens: int
        output_tokens: int

    @dataclass
    class Response:
        model: str
        usage: Usage

    path = _tmp_path()

    @trace(tag="decorated", path=path)
    def fake_call(prompt: str) -> Response:
        return Response(model="claude-sonnet-5", usage=Usage(input_tokens=5, output_tokens=10))

    result = fake_call("hello")
    assert result.model == "claude-sonnet-5"  # decorator doesn't swallow the return value

    records = list(read_records(path))
    assert len(records) == 1
    assert records[0]["model"] == "claude-sonnet-5"
    assert records[0]["total_tokens"] == 15
    clear_log(path)


def test_trace_decorator_handles_unrecognized_response_gracefully(capsys):
    path = _tmp_path()

    @trace(tag="bad", path=path)
    def fake_call() -> str:
        return "not a response object"

    result = fake_call()
    assert result == "not a response object"  # still returns normally

    captured = capsys.readouterr()
    assert "warning" in captured.out.lower()

    records = list(read_records(path))
    assert len(records) == 0


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
