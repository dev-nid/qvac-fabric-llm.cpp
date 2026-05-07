#!/usr/bin/env python3
"""
scripts/dflash_bench_diff.py

Compare two bench JSON files emitted by `scripts/dflash_bench_cpu.sh`.
Companion of the optimization plan in
`logs/core_architecture/10_optimization_plan.md`.

Usage:
    scripts/dflash_bench_diff.py BEFORE.json AFTER.json [options]

Options:
    --metric M          Metric to compare. Default: tokens_per_sec.
                        Other useful: t_verify_ms, t_draft_ms, accept_rate_pct.
    --regress-pct N     Threshold (%) for flagging a regression. Default: 5.
    --improve-pct N     Threshold (%) for flagging an improvement. Default: 3.
    --json              Emit JSON instead of a human table.
    --strict            Exit non-zero if any regression >= regress_pct.

Exit codes:
    0  no regressions worse than --regress-pct (or --strict not set)
    1  at least one regression worse than --regress-pct (with --strict)
    2  setup error (file load / schema mismatch)
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any


def load(path: str) -> dict[str, Any]:
    try:
        with open(path) as f:
            doc = json.load(f)
    except Exception as e:
        print(f"[error] cannot load {path}: {e}", file=sys.stderr)
        sys.exit(2)
    if doc.get("schema_version") != 1:
        print(
            f"[warn] {path}: schema_version != 1 (got {doc.get('schema_version')!r}); "
            "trying anyway",
            file=sys.stderr,
        )
    return doc


def index_results(doc: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    """Index results by (config, prompt) tuple."""
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for r in doc.get("results", []):
        key = (r.get("config", ""), r.get("prompt", ""))
        out[key] = r
    return out


# ANSI colors (no-op when not a TTY)
def _color(s: str, code: str) -> str:
    if not sys.stdout.isatty():
        return s
    return f"\033[{code}m{s}\033[0m"


def green(s: str) -> str:
    return _color(s, "32")


def red(s: str) -> str:
    return _color(s, "31")


def yellow(s: str) -> str:
    return _color(s, "33")


def gray(s: str) -> str:
    return _color(s, "90")


def cyan(s: str) -> str:
    return _color(s, "36")


def fmt_delta(before: float | None, after: float | None, higher_is_better: bool, regress_pct: float, improve_pct: float) -> tuple[str, str | None]:
    """Format a delta cell. Returns (text, status) where status is one of
    'win' / 'loss' / 'flat' / 'na'."""
    if before is None or after is None or before == 0:
        return gray("    n/a"), "na"
    delta = after - before
    delta_pct = 100.0 * delta / before
    sign = "+" if delta >= 0 else ""
    txt = f"{after:>8.2f} ({sign}{delta_pct:>6.2f}%)"

    is_improvement = (delta_pct > improve_pct) if higher_is_better else (delta_pct < -improve_pct)
    is_regression = (delta_pct < -regress_pct) if higher_is_better else (delta_pct > regress_pct)

    if is_improvement:
        return green(txt), "win"
    elif is_regression:
        return red(txt), "loss"
    else:
        return txt, "flat"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("before", help="Baseline JSON")
    ap.add_argument("after", help="New JSON")
    ap.add_argument("--metric", default="tokens_per_sec",
                    help="Metric to compare (default: tokens_per_sec)")
    ap.add_argument("--regress-pct", type=float, default=5.0)
    ap.add_argument("--improve-pct", type=float, default=3.0)
    ap.add_argument("--json", action="store_true", help="Emit JSON")
    ap.add_argument("--strict", action="store_true",
                    help="Exit non-zero on any regression >= --regress-pct")
    args = ap.parse_args()

    before_doc = load(args.before)
    after_doc = load(args.after)

    before_idx = index_results(before_doc)
    after_idx = index_results(after_doc)

    higher_is_better = args.metric in {
        "tokens_per_sec",
        "accept_rate_pct",
        "tree_alt_taken_pct",
    }

    # Union of (config, prompt) keys, in insertion order from `after` first.
    seen: dict[tuple[str, str], None] = {}
    for k in after_idx:
        seen.setdefault(k, None)
    for k in before_idx:
        seen.setdefault(k, None)
    keys = list(seen)

    rows = []
    n_win = n_loss = n_flat = n_na = 0
    for key in keys:
        b = before_idx.get(key)
        a = after_idx.get(key)
        bv = b.get(args.metric) if b else None
        av = a.get(args.metric) if a else None
        cell, status = fmt_delta(bv, av, higher_is_better, args.regress_pct, args.improve_pct)
        rows.append((key[0], key[1], bv, av, cell, status))
        if status == "win":
            n_win += 1
        elif status == "loss":
            n_loss += 1
        elif status == "flat":
            n_flat += 1
        else:
            n_na += 1

    if args.json:
        print(json.dumps({
            "before": args.before,
            "after": args.after,
            "metric": args.metric,
            "higher_is_better": higher_is_better,
            "regress_pct": args.regress_pct,
            "improve_pct": args.improve_pct,
            "rows": [
                {"config": c, "prompt": p, "before": b, "after": a, "status": s}
                for c, p, b, a, _, s in rows
            ],
            "summary": {"win": n_win, "loss": n_loss, "flat": n_flat, "na": n_na},
        }, indent=2))
    else:
        print(cyan(f"DFlash bench diff: {args.before} -> {args.after}"))
        print(f"  metric            : {args.metric} ({'higher is better' if higher_is_better else 'lower is better'})")
        print(f"  thresholds        : regression <= -{args.regress_pct}%, improvement >= +{args.improve_pct}%")
        print(f"  before git_sha    : {before_doc.get('git_sha', '?')}")
        print(f"  after  git_sha    : {after_doc.get('git_sha', '?')}")
        print()

        # Compute column widths
        c_w = max(len("config"), max((len(c) for c, _, _, _, _, _ in rows), default=0))
        p_w = max(len("prompt"), max((len(p) for _, p, _, _, _, _ in rows), default=0))

        header = f"  {'config':<{c_w}}  {'prompt':<{p_w}}  {'before':>10}  {'after (delta%)':>20}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for c, p, bv, av, cell, _ in rows:
            bv_s = f"{bv:>10.2f}" if isinstance(bv, (int, float)) else gray(f"{'n/a':>10}")
            print(f"  {c:<{c_w}}  {p:<{p_w}}  {bv_s}  {cell:>20}")

        print()
        print(f"summary: {green(str(n_win) + ' wins')}, "
              f"{red(str(n_loss) + ' regressions')}, "
              f"{n_flat} flat, "
              f"{gray(str(n_na) + ' n/a')}")
        if args.strict and n_loss > 0:
            print(red(f"FAIL: --strict and {n_loss} regression(s) >= {args.regress_pct}% — see above"))
            return 1

    if args.strict and n_loss > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
