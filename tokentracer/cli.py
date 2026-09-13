"""
CLI report tool.

Usage:
    python -m tokentracer.cli report
    python -m tokentracer.cli report --by tag
    python -m tokentracer.cli report --by day
    python -m tokentracer.cli tail --n 10
    python -m tokentracer.cli clear
"""
from __future__ import annotations

import argparse
from collections import defaultdict

from tokentracer.storage import DEFAULT_LOG_PATH, clear_log, read_records


def _fmt_usd(x: float) -> str:
    return f"${x:,.4f}"


def cmd_report(path: str, group_by: str) -> None:
    records = list(read_records(path))
    if not records:
        print(f"No trace records found at '{path}'. Run some traced calls first.")
        return

    total_calls = len(records)
    total_in = sum(r["input_tokens"] for r in records)
    total_out = sum(r["output_tokens"] for r in records)
    total_cost = sum(r["cost_usd"] for r in records)

    print(f"\n=== TokenTracer Report ({path}) ===")
    print(f"Total calls:  {total_calls}")
    print(f"Input tokens:  {total_in:,}")
    print(f"Output tokens: {total_out:,}")
    print(f"Total tokens:  {total_in + total_out:,}")
    print(f"Total cost:    {_fmt_usd(total_cost)}")

    def key_for(r: dict) -> str:
        if group_by == "model":
            return r.get("model", "unknown")
        if group_by == "tag":
            return r.get("tag") or "(untagged)"
        if group_by == "day":
            return r.get("timestamp", "")[:10]
        return "all"

    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[key_for(r)].append(r)

    print(f"\n--- Breakdown by {group_by} ---")
    header = f"{'':<28} {'calls':>7} {'in_tok':>10} {'out_tok':>10} {'cost':>12}"
    print(header)
    print("-" * len(header))
    for key, rs in sorted(groups.items(), key=lambda kv: -sum(x["cost_usd"] for x in kv[1])):
        c_in = sum(r["input_tokens"] for r in rs)
        c_out = sum(r["output_tokens"] for r in rs)
        c_cost = sum(r["cost_usd"] for r in rs)
        print(f"{key:<28} {len(rs):>7} {c_in:>10,} {c_out:>10,} {_fmt_usd(c_cost):>12}")
    print()


def cmd_tail(path: str, n: int) -> None:
    records = list(read_records(path))
    if not records:
        print(f"No trace records found at '{path}'.")
        return
    for r in records[-n:]:
        tag = r.get("tag") or "-"
        print(
            f"{r['timestamp']}  {r['model']:<28} tag={tag:<15} "
            f"in={r['input_tokens']:<6} out={r['output_tokens']:<6} "
            f"cost={_fmt_usd(r['cost_usd'])}"
        )


def cmd_clear(path: str) -> None:
    clear_log(path)
    print(f"Cleared trace log at '{path}'.")


def main() -> None:
    parser = argparse.ArgumentParser(description="TokenTracer — LLM token usage & cost reporting")
    parser.add_argument("--path", default=DEFAULT_LOG_PATH, help="Path to the JSONL trace log")
    sub = parser.add_subparsers(dest="command", required=True)

    p_report = sub.add_parser("report", help="Summarize usage and cost")
    p_report.add_argument("--by", choices=["model", "tag", "day", "all"], default="model")

    p_tail = sub.add_parser("tail", help="Show the most recent N trace records")
    p_tail.add_argument("--n", type=int, default=10)

    sub.add_parser("clear", help="Delete the trace log")

    args = parser.parse_args()

    if args.command == "report":
        cmd_report(args.path, args.by)
    elif args.command == "tail":
        cmd_tail(args.path, args.n)
    elif args.command == "clear":
        cmd_clear(args.path)


if __name__ == "__main__":
    main()
