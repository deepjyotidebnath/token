"""
Core tracing API.

Three ways to use it:

1. Decorator — wrap any function that returns an Anthropic-style response
   (an object/dict with `.model`, `.usage.input_tokens`, `.usage.output_tokens`):

       @trace(tag="summarize")
       def call_claude(prompt):
           return client.messages.create(...)

2. Context manager — for finer control or non-Anthropic clients, report
   token counts yourself:

       with trace_call(tag="summarize") as t:
           response = client.messages.create(...)
           t.record(model=response.model,
                     input_tokens=response.usage.input_tokens,
                     output_tokens=response.usage.output_tokens)

3. Direct logging — no wrapping at all:

       log_usage(model="claude-sonnet-5", input_tokens=120, output_tokens=340, tag="summarize")
"""
from __future__ import annotations

import functools
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, TypeVar

from tokentracer.pricing import cost_for
from tokentracer.storage import DEFAULT_LOG_PATH, TraceRecord, append_record

F = TypeVar("F", bound=Callable[..., Any])


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _extract_usage(response: Any) -> tuple[str, int, int]:
    """
    Best-effort extraction of (model, input_tokens, output_tokens) from
    common response shapes: Anthropic SDK objects, OpenAI SDK objects, or
    plain dicts with an equivalent structure.
    """
    # Anthropic SDK object: response.model, response.usage.input_tokens/output_tokens
    if hasattr(response, "usage") and hasattr(response, "model"):
        usage = response.usage
        if hasattr(usage, "input_tokens"):
            return response.model, usage.input_tokens, usage.output_tokens
        # OpenAI SDK object: response.usage.prompt_tokens/completion_tokens
        if hasattr(usage, "prompt_tokens"):
            return response.model, usage.prompt_tokens, usage.completion_tokens

    # Plain dict shape
    if isinstance(response, dict):
        model = response.get("model", "unknown")
        usage = response.get("usage", {})
        in_tok = usage.get("input_tokens", usage.get("prompt_tokens", 0))
        out_tok = usage.get("output_tokens", usage.get("completion_tokens", 0))
        return model, in_tok, out_tok

    raise ValueError(
        "Could not extract token usage from response. Use trace_call() or "
        "log_usage() to report tokens manually for unsupported response shapes."
    )


def log_usage(
    model: str,
    input_tokens: int,
    output_tokens: int,
    tag: str | None = None,
    duration_ms: float | None = None,
    metadata: dict[str, Any] | None = None,
    path: str = DEFAULT_LOG_PATH,
) -> TraceRecord:
    """Write a single trace record directly. Returns the record for inspection."""
    record = TraceRecord(
        timestamp=_now_iso(),
        model=model,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cost_usd=round(cost_for(model, input_tokens, output_tokens), 6),
        tag=tag,
        duration_ms=duration_ms,
        metadata=metadata,
    )
    append_record(record, path=path)
    return record


class _TraceHandle:
    """Yielded by trace_call(); lets you report usage from inside the `with` block."""

    def __init__(self, tag: str | None, path: str):
        self.tag = tag
        self.path = path
        self._start = time.perf_counter()
        self.record: TraceRecord | None = None

    def report(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        metadata: dict[str, Any] | None = None,
    ) -> TraceRecord:
        duration_ms = (time.perf_counter() - self._start) * 1000
        self.record = log_usage(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            tag=self.tag,
            duration_ms=duration_ms,
            metadata=metadata,
            path=self.path,
        )
        return self.record

    # alias — reads slightly better at call sites
    record_usage = report


@contextmanager
def trace_call(tag: str | None = None, path: str = DEFAULT_LOG_PATH) -> Iterator[_TraceHandle]:
    handle = _TraceHandle(tag=tag, path=path)
    yield handle
    if handle.record is None:
        print(f"[tokentracer] warning: trace_call(tag={tag!r}) exited without "
              f"calling .report(...) — nothing was logged.")


def trace(tag: str | None = None, path: str = DEFAULT_LOG_PATH) -> Callable[[F], F]:
    """
    Decorator for functions that return an LLM SDK response object (or dict)
    containing model + token usage. Logs automatically after the call.
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            response = func(*args, **kwargs)
            duration_ms = (time.perf_counter() - start) * 1000
            try:
                model, in_tok, out_tok = _extract_usage(response)
                log_usage(
                    model=model,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    tag=tag or func.__name__,
                    duration_ms=duration_ms,
                    path=path,
                )
            except ValueError as e:
                print(f"[tokentracer] warning: {e}")
            return response
        return wrapper  # type: ignore[return-value]
    return decorator
