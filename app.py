"""
TokenTracer — LLM Token Usage Analyzer & Cost Optimizer
=========================================================
A self-contained Flask app that:
  - Counts tokens for a given prompt/response across multiple LLM providers
  - Estimates cost per model and highlights the cheapest option
  - Logs usage history to SQLite for trend analysis
  - Runs a lightweight heuristic "optimizer" that flags waste
    (redundant whitespace, repeated phrases, oversized context,
    verbose system prompts, non-cached repeated prompts) and
    estimates potential savings.

Run:
    pip install -r requirements.txt
    python app.py
    -> http://127.0.0.1:5000
"""

from __future__ import annotations

import json
import re
import sqlite3
import time
from collections import Counter
from dataclasses import dataclass, asdict
from pathlib import Path

from flask import Flask, g, jsonify, render_template, request

try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - fallback if tiktoken unavailable
    _ENC = None

APP_DIR = Path(__file__).parent
DB_PATH = APP_DIR / "tokentracer.db"

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Pricing table (USD per 1,000,000 tokens). Illustrative rates — update as
# providers change pricing. Values: (input_price, output_price)
# ---------------------------------------------------------------------------
PRICING = {
    "gpt-4o":            {"provider": "OpenAI",    "in": 5.00,  "out": 15.00, "context": 128_000},
    "gpt-4o-mini":       {"provider": "OpenAI",    "in": 0.15,  "out": 0.60,  "context": 128_000},
    "gpt-4-turbo":       {"provider": "OpenAI",    "in": 10.00, "out": 30.00, "context": 128_000},
    "gpt-3.5-turbo":     {"provider": "OpenAI",    "in": 0.50,  "out": 1.50,  "context": 16_000},
    "claude-opus-4":     {"provider": "Anthropic", "in": 15.00, "out": 75.00, "context": 200_000},
    "claude-sonnet-4":   {"provider": "Anthropic", "in": 3.00,  "out": 15.00, "context": 200_000},
    "claude-haiku-4":    {"provider": "Anthropic", "in": 0.80,  "out": 4.00,  "context": 200_000},
    "gemini-1.5-pro":    {"provider": "Google",    "in": 3.50,  "out": 10.50, "context": 1_000_000},
    "gemini-1.5-flash":  {"provider": "Google",    "in": 0.075, "out": 0.30,  "context": 1_000_000},
    "llama-3.1-70b":     {"provider": "Meta/OSS",  "in": 0.59,  "out": 0.79,  "context": 128_000},
    "mixtral-8x7b":      {"provider": "Mistral/OSS","in": 0.24, "out": 0.24,  "context": 32_000},
}

DEFAULT_MODEL = "gpt-4o"

# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base) with a heuristic fallback."""
    if not text:
        return 0
    if _ENC is not None:
        return len(_ENC.encode(text))
    # Fallback heuristic: ~4 chars/token for English text
    return max(1, round(len(text) / 4))


def cost_for(model: str, input_tokens: int, output_tokens: int) -> float:
    p = PRICING.get(model, PRICING[DEFAULT_MODEL])
    return (input_tokens / 1_000_000) * p["in"] + (output_tokens / 1_000_000) * p["out"]


def cost_breakdown(input_tokens: int, output_tokens: int) -> list[dict]:
    rows = []
    for model, p in PRICING.items():
        c = cost_for(model, input_tokens, output_tokens)
        rows.append({
            "model": model,
            "provider": p["provider"],
            "cost": round(c, 6),
            "context_limit": p["context"],
        })
    rows.sort(key=lambda r: r["cost"])
    return rows


# ---------------------------------------------------------------------------
# Optimization heuristics
# ---------------------------------------------------------------------------

@dataclass
class Suggestion:
    severity: str   # "high" | "medium" | "low"
    title: str
    detail: str
    est_savings_pct: float  # estimated % reduction in token count this fix could yield


def analyze_optimizations(prompt: str, model: str, history_repeats: int) -> list[Suggestion]:
    suggestions: list[Suggestion] = []
    tokens = count_tokens(prompt)

    # 1. Redundant whitespace / formatting bloat
    ws_ratio = len(re.findall(r"[ \t]{2,}|\n{3,}", prompt)) / max(1, len(prompt) / 200)
    if ws_ratio > 0.5:
        suggestions.append(Suggestion(
            "low", "Trim redundant whitespace",
            "Multiple runs of extra spaces/blank lines detected. Collapsing them "
            "reduces tokens without changing meaning.",
            est_savings_pct=1.5,
        ))

    # 2. Repeated phrases/sentences (copy-paste bloat, boilerplate repeated in context)
    sentences = re.split(r"(?<=[.!?])\s+", prompt)
    dupe_count = sum(c - 1 for c in Counter(s.strip().lower() for s in sentences if len(s.strip()) > 15).values() if c > 1)
    if dupe_count > 0:
        pct = min(25.0, dupe_count * 2.5)
        suggestions.append(Suggestion(
            "medium", "Repeated content detected",
            f"{dupe_count} near-duplicate sentence(s) found. Deduplicating repeated "
            "instructions or context chunks cuts tokens sent on every call.",
            est_savings_pct=pct,
        ))

    # 3. Oversized prompt relative to model context window
    ctx = PRICING.get(model, PRICING[DEFAULT_MODEL])["context"]
    usage_pct = tokens / ctx * 100
    if usage_pct > 60:
        suggestions.append(Suggestion(
            "high", "Prompt nearing context limit",
            f"Using {usage_pct:.1f}% of {model}'s {ctx:,}-token context window. "
            "Consider chunking, summarizing older context, or switching to a "
            "larger-context model to avoid truncation and rising per-call cost.",
            est_savings_pct=0.0,
        ))

    # 4. Verbose system-prompt / instruction pattern (very long prompt for a
    #    simple-looking request — heuristic: many instruction keywords + long length)
    instruction_words = len(re.findall(r"\b(please|make sure|remember|note that|important|always|never)\b", prompt, re.I))
    if instruction_words >= 5 and tokens > 300:
        suggestions.append(Suggestion(
            "medium", "Verbose instruction style",
            f"{instruction_words} emphasis/instruction phrases found in a "
            f"{tokens}-token prompt. Tightening system instructions (bullet lists "
            "over repeated emphasis) typically shrinks prompts 10-20%.",
            est_savings_pct=12.0,
        ))

    # 5. Model-cost mismatch — flag if an expensive model is used for a short,
    #    simple-looking prompt that a cheaper model could likely handle.
    if model in ("gpt-4o", "gpt-4-turbo", "claude-opus-4") and tokens < 200:
        cheaper = "gpt-4o-mini" if "gpt" in model else "claude-haiku-4"
        current_cost = cost_for(model, tokens, tokens // 2)
        cheaper_cost = cost_for(cheaper, tokens, tokens // 2)
        if current_cost > 0 and cheaper_cost < current_cost * 0.3:
            savings = (1 - cheaper_cost / current_cost) * 100
            suggestions.append(Suggestion(
                "high", f"Consider {cheaper} for this request",
                f"Short, low-complexity prompt ({tokens} tokens) on a premium model. "
                f"Routing simple requests to {cheaper} could cut this call's cost by "
                f"~{savings:.0f}%.",
                est_savings_pct=0.0,
            ))

    # 6. Repeated identical prompts without caching
    if history_repeats >= 2:
        suggestions.append(Suggestion(
            "high", "Enable prompt/response caching",
            f"This exact prompt has been sent {history_repeats} times before. "
            "Caching identical or near-identical requests (or using provider-side "
            "prompt caching) avoids paying for repeated processing entirely.",
            est_savings_pct=0.0,
        ))

    if not suggestions:
        suggestions.append(Suggestion(
            "low", "Looks efficient",
            "No obvious waste patterns detected in this prompt.",
            est_savings_pct=0.0,
        ))

    return suggestions


# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(_exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS usage_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            model TEXT NOT NULL,
            prompt_hash TEXT NOT NULL,
            input_tokens INTEGER NOT NULL,
            output_tokens INTEGER NOT NULL,
            cost REAL NOT NULL,
            label TEXT
        )
    """)
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html", models=list(PRICING.keys()))


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    completion = data.get("completion", "")
    model = data.get("model", DEFAULT_MODEL)

    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(completion)

    prompt_hash = str(hash(prompt.strip().lower()))
    db = get_db()
    repeats = db.execute(
        "SELECT COUNT(*) c FROM usage_log WHERE prompt_hash = ?", (prompt_hash,)
    ).fetchone()["c"]

    breakdown = cost_breakdown(input_tokens, output_tokens)
    suggestions = analyze_optimizations(prompt, model, repeats)
    cheapest = breakdown[0]
    current = next((r for r in breakdown if r["model"] == model), breakdown[0])

    potential_savings = round(current["cost"] - cheapest["cost"], 6)

    return jsonify({
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "current_model": model,
        "current_cost": current["cost"],
        "cheapest_model": cheapest["model"],
        "cheapest_cost": cheapest["cost"],
        "potential_savings_vs_cheapest": max(0.0, potential_savings),
        "breakdown": breakdown,
        "suggestions": [asdict(s) for s in suggestions],
        "prompt_hash": prompt_hash,
        "repeats_seen": repeats,
    })


@app.route("/api/log", methods=["POST"])
def api_log():
    data = request.get_json(force=True)
    prompt = data.get("prompt", "")
    completion = data.get("completion", "")
    model = data.get("model", DEFAULT_MODEL)
    label = data.get("label", "")

    input_tokens = count_tokens(prompt)
    output_tokens = count_tokens(completion)
    cost = cost_for(model, input_tokens, output_tokens)
    prompt_hash = str(hash(prompt.strip().lower()))

    db = get_db()
    db.execute(
        "INSERT INTO usage_log (ts, model, prompt_hash, input_tokens, output_tokens, cost, label) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (time.time(), model, prompt_hash, input_tokens, output_tokens, cost, label),
    )
    db.commit()
    return jsonify({"status": "logged", "cost": round(cost, 6)})


@app.route("/api/history")
def api_history():
    db = get_db()
    rows = db.execute(
        "SELECT ts, model, input_tokens, output_tokens, cost, label FROM usage_log "
        "ORDER BY ts DESC LIMIT 200"
    ).fetchall()
    return jsonify([dict(r) for r in rows])


@app.route("/api/stats")
def api_stats():
    db = get_db()
    total = db.execute(
        "SELECT COUNT(*) n, COALESCE(SUM(cost),0) cost, "
        "COALESCE(SUM(input_tokens+output_tokens),0) tokens FROM usage_log"
    ).fetchone()
    by_model = db.execute(
        "SELECT model, COUNT(*) n, COALESCE(SUM(cost),0) cost FROM usage_log "
        "GROUP BY model ORDER BY cost DESC"
    ).fetchall()
    daily = db.execute(
        "SELECT date(ts, 'unixepoch') day, COALESCE(SUM(cost),0) cost FROM usage_log "
        "GROUP BY day ORDER BY day ASC LIMIT 30"
    ).fetchall()
    return jsonify({
        "total_calls": total["n"],
        "total_cost": round(total["cost"], 6),
        "total_tokens": total["tokens"],
        "by_model": [dict(r) for r in by_model],
        "daily": [dict(r) for r in daily],
    })


@app.route("/api/reset", methods=["POST"])
def api_reset():
    db = get_db()
    db.execute("DELETE FROM usage_log")
    db.commit()
    return jsonify({"status": "cleared"})


if __name__ == "__main__":
    init_db()
    app.run(debug=True, port=5000)
