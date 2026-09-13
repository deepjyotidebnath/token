"""
TokenTracer — a tiny, dependency-free library for tracing LLM token usage
and cost across your app.

    from tokentracer import trace, trace_call, log_usage

    @trace(tag="summarize")
    def call_claude(prompt):
        return client.messages.create(...)
"""
from tokentracer.tracer import trace, trace_call, log_usage
from tokentracer.storage import DEFAULT_LOG_PATH, read_records, clear_log
from tokentracer.pricing import MODEL_PRICING, cost_for

__all__ = [
    "trace",
    "trace_call",
    "log_usage",
    "DEFAULT_LOG_PATH",
    "read_records",
    "clear_log",
    "MODEL_PRICING",
    "cost_for",
]

__version__ = "0.1.0"
