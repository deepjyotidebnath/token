"""
Example usage of TokenTracer — runs with no API key required, using a
mock response object shaped like an Anthropic SDK response. Swap
`fake_claude_call` for a real `client.messages.create(...)` call to trace
actual API usage.

Run: python example.py
Then: python -m tokentracer.cli report
"""
from dataclasses import dataclass

from tokentracer import trace, trace_call, log_usage


@dataclass
class FakeUsage:
    input_tokens: int
    output_tokens: int


@dataclass
class FakeResponse:
    model: str
    usage: FakeUsage


# --- 1. Decorator style ---
@trace(tag="summarize")
def fake_claude_call(prompt: str) -> FakeResponse:
    # In real code this would be: client.messages.create(model=..., messages=[...])
    return FakeResponse(model="claude-sonnet-5", usage=FakeUsage(input_tokens=len(prompt.split()), output_tokens=48))


# --- 2. Context manager style ---
def fake_call_with_manual_report(prompt: str) -> str:
    with trace_call(tag="classify") as t:
        # ... call your LLM here ...
        response = FakeResponse(model="claude-haiku-4-5-20251001", usage=FakeUsage(input_tokens=30, output_tokens=5))
        t.report(model=response.model,
                 input_tokens=response.usage.input_tokens,
                 output_tokens=response.usage.output_tokens)
    return "positive"


if __name__ == "__main__":
    fake_claude_call("Summarize this quarterly report for the exec team please")
    fake_claude_call("Summarize this shorter memo")
    fake_call_with_manual_report("Is this review positive or negative?")

    # --- 3. Direct logging, no wrapping ---
    log_usage(model="claude-opus-5", input_tokens=500, output_tokens=1200, tag="deep_research")

    print("Logged 4 trace records to tokentracer.jsonl")
    print("Run: python -m tokentracer.cli report --by tag")
