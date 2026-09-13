"""
Per-model token pricing (USD per 1M tokens) and cost calculation.

Prices are approximate and change over time — treat MODEL_PRICING as a
config table you update, not a source of truth. Unknown models fall back
to a $0 rate with a warning rather than crashing, so tracing never blocks
on a missing price entry.
"""
from __future__ import annotations

# USD per 1,000,000 tokens
MODEL_PRICING: dict[str, dict[str, float]] = {
    "claude-opus-5":            {"input": 15.00, "output": 75.00},
    "claude-sonnet-5":          {"input": 3.00,  "output": 15.00},
    "claude-fable-5-1":         {"input": 3.00,  "output": 15.00},
    "claude-haiku-4-5-20251001": {"input": 0.80,  "output": 4.00},
    "gpt-4o":                   {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":              {"input": 0.15,  "output": 0.60},
}

_warned_models: set[str] = set()


def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    """Return the USD cost of a call given its token counts."""
    rates = MODEL_PRICING.get(model)
    if rates is None:
        if model not in _warned_models:
            _warned_models.add(model)
            print(f"[tokentracer] warning: no pricing entry for model '{model}', "
                  f"costing this call as $0.00. Add it to MODEL_PRICING.")
        rates = {"input": 0.0, "output": 0.0}
    return (input_tokens / 1_000_000) * rates["input"] + (output_tokens / 1_000_000) * rates["output"]
