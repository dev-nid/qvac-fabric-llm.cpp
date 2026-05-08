#!/usr/bin/env python3
"""
Render bench/cpu/*.json + bench/gpu/*.json into bench/README.md.

Per-commit DFlash speculative-decoding perf timeline. One row per
(commit-tag, hardware, n_predict). Each cell shows tokens_per_sec, with
Δ% vs the prior row in the same (hardware, n_predict) group.

Usage:
    scripts/dflash_bench_render.py            # render bench/README.md in place
    scripts/dflash_bench_render.py --out X.md # render to a custom path
    scripts/dflash_bench_render.py --check    # exit 1 if the rendered output
                                              # differs from the on-disk file
                                              # (useful for CI)
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BENCH_DIRS = [ROOT / "bench" / "cpu", ROOT / "bench" / "gpu"]
DEFAULT_OUT = ROOT / "bench" / "README.md"

# Stable display order. Anything not listed here is appended in
# alphabetical order so a new config / prompt added to the harness later
# doesn't silently drop out of the table.
CONFIG_ORDER = [
    "greedy",
    "chain",
    "chain-topk-2",
    "tree-medusa-k2",
    "tree-budget-18",
    "tree-budget-22",
]
PROMPT_ORDER = [
    "math",
    "nl-low-entropy",
    "nl-high-entropy",
    "code",
    "conv",
]

# Filenames (without .json) that are pre-port qvac snapshots. They're kept
# on disk under bench/{cpu,gpu}/ as reference points for the diff tool, but
# excluded from the per-commit timeline rendered into bench/README.md — the
# README tracks this fork's commits going forward, starting fresh from the
# first bench run on the current HEAD. Any bench JSON whose tag isn't
# listed here is included in the rendered timeline.
HISTORICAL_TAGS = {
    "baseline",
    "baseline_ub1",
    "c33-topk",
    "c34-topk1",
    "c35-qkv",
    "c35-no-redecode",
    "m1a-mem",
    "c32-vulkan-n1024",
    "c35-vulkan-baseline",
    "c35-vulkan-n512",
    "c35-vulkan-n1024",
    "m1a-mem-vulkan",
}


def parse_ts(s):
    """The bench harness emits 2026-05-05T20-19-14Z (dashes inside time)."""
    try:
        return datetime.strptime(s, "%Y-%m-%dT%H-%M-%SZ")
    except Exception:
        return datetime.min


def hardware_tag(path: Path) -> str:
    parent = path.parent.name
    if parent == "cpu":
        return "CPU"
    if parent == "gpu":
        return "GPU (Vulkan)"
    return parent


def model_pair_tag(j: dict) -> str:
    """Short identifier for the (target, draft) GGUF pair, used to keep
    different model pairs in different timeline sections (e.g. 4B vs 8B
    greedy throughput differs ~40% so mixing them in one group reads
    like a regression)."""
    target = Path(j.get("target_model", "") or "").name
    draft  = Path(j.get("draft_model",  "") or "").name
    if not target and not draft:
        return "?"
    # strip extension and quant suffix to keep tag short
    def short(s):
        s = s.replace(".gguf", "")
        return s
    return f"{short(target)} + {short(draft)}"


def ubatch_tag(path: Path) -> str:
    """Heuristic: filenames containing `_ub1` were captured at -ub 1.

    The bench harness doesn't currently emit the ubatch setting into the
    JSON, so we recover it from the filename convention used in the qvac
    snapshot timeline. -ub 1 runs sit on a *different* perf curve from the
    default ubatch (the strong-correctness gate uses -ub 1 for byte-exact
    reduction order; the optimisation timeline uses the default), so they
    must be a separate row group rather than rows in the same timeline.
    """
    return "1" if "_ub1" in path.stem else "auto"


def fmt_timestamp(s: str) -> str:
    """Bench harness emits 2026-05-05T19-47-48Z. Render as 2026-05-05 19:47:48Z."""
    if not s:
        return "?"
    # split on T, replace dashes in the time half with colons
    if "T" in s:
        date, time = s.split("T", 1)
        time = time.replace("-", ":")
        return f"{date} {time}"
    return s


def load_snapshots():
    snaps = []
    for d in BENCH_DIRS:
        if not d.is_dir():
            continue
        for f in sorted(d.glob("*.json")):
            tag = f.stem
            if tag in HISTORICAL_TAGS:
                continue
            try:
                j = json.load(f.open())
            except Exception as e:
                print(f"skip {f}: {e}", file=sys.stderr)
                continue
            j["_path"] = f
            j["_tag"] = tag
            # Display label: prefer the JSON's git_sha (e.g. "5f058cc57"
            # or "5f058cc57-dirty"); fall back to the filename stem for
            # historical entries that don't have a sha-shaped name.
            j["_label"] = j.get("git_sha") or tag
            j["_hw"] = hardware_tag(f)
            j["_ub"] = ubatch_tag(f)
            j["_ts"] = parse_ts(j.get("timestamp_utc", ""))
            j["_pair"] = model_pair_tag(j)
            j["_grid"] = {
                (r["config"], r["prompt"]): r.get("tokens_per_sec")
                for r in j.get("results", [])
            }
            j["_full"] = {
                (r["config"], r["prompt"]): r
                for r in j.get("results", [])
            }
            snaps.append(j)
    return snaps


def ordered(items, preferred):
    """Items in `preferred` order, then leftovers sorted alphabetically."""
    seen = set()
    out = [x for x in preferred if x in items and not (x in seen or seen.add(x))]
    leftover = sorted(x for x in items if x not in seen)
    return out + leftover


def fmt_cell(curr, prev):
    if curr is None:
        return "n/a"
    if prev is None or prev == 0:
        return f"{curr:.2f}"
    delta = (curr - prev) / prev * 100.0
    return f"{curr:.2f} ({delta:+.1f}%)"


def render_savings(snaps, hw, n, ub, pair):
    """For the latest snapshot in this group, show per-prompt time savings:
    greedy wall time -> best DFlash wall time, normalised by tokens_predicted
    so configs that overshoot the n_predict cap don't artificially shrink.
    Speedup = ms_per_token_greedy / ms_per_token_best (>1 = DFlash faster).
    """
    snaps = sorted(
        (s for s in snaps
         if s["_hw"] == hw and s.get("n_predict") == n and s["_ub"] == ub
            and s["_pair"] == pair),
        key=lambda s: (s["_ts"], s["_tag"]),
    )
    if not snaps:
        return ""
    latest = snaps[-1]
    full = latest["_full"]
    prompts = ordered({p for (_, p) in full}, PROMPT_ORDER)
    configs = [c for c in CONFIG_ORDER if c != "greedy" and any((c, p) in full for p in prompts)]
    # any leftover dflash configs not in CONFIG_ORDER:
    extra = sorted({c for (c, _) in full
                    if c != "greedy" and c not in CONFIG_ORDER})
    configs += extra

    def ms_per_tok(rec, key="wall_ms"):
        if not rec or not rec.get("tokens_predicted"):
            return None
        ms = rec.get(key)
        if ms is None:
            return None
        return ms / rec["tokens_predicted"]

    def sec_prompt(rec):
        if not rec:
            return None
        ms = rec.get("prompt_eval_ms")
        if ms is None:
            return None
        return ms / 1000.0

    def sec_norm(rec, key):
        mpt = ms_per_tok(rec, key=key)
        if mpt is None:
            return None
        return mpt * n / 1000.0

    rows = []
    sum_g = 0.0
    sum_best = 0.0
    for p in prompts:
        g = full.get(("greedy", p))
        g_ms = ms_per_tok(g)
        if g_ms is None:
            continue
        # Best DFlash config = lowest ms_per_token across configs that ran ok.
        candidates = []
        for c in configs:
            r = full.get((c, p))
            if not r or not r.get("ok"):
                continue
            mpt = ms_per_tok(r)
            if mpt is None:
                continue
            candidates.append((c, r, mpt))
        if not candidates:
            continue
        best_c, _, best_ms = min(candidates, key=lambda x: x[2])
        # Wall time to generate N=n_predict tokens (normalised; a config
        # that emitted N±1 tokens is rescaled to N for the comparison).
        g_secs = g_ms * n / 1000.0
        best_secs = best_ms * n / 1000.0
        g_prompt = sec_prompt(g)
        g_gen = sec_norm(g, "gen_eval_ms")
        best_rec = full.get((best_c, p))
        best_prompt = sec_prompt(best_rec)
        best_gen = sec_norm(best_rec, "gen_eval_ms")
        saved_secs = g_secs - best_secs
        saved_pct = saved_secs / g_secs * 100.0
        speedup = g_secs / best_secs
        sum_g += g_secs
        sum_best += best_secs
        rows.append({
            "prompt": p,
            "g_secs": g_secs,
            "g_prompt": g_prompt,
            "g_gen": g_gen,
            "best": best_c,
            "best_secs": best_secs,
            "best_prompt": best_prompt,
            "best_gen": best_gen,
            "saved_secs": saved_secs,
            "saved_pct": saved_pct,
            "speedup": speedup,
        })

    if not rows:
        return ""

    suffix = "" if ub == "auto" else f", -ub {ub}"
    out = []
    out.append(f"### {hw}, n_predict={n}{suffix} — {pair} — time savings vs greedy")
    out.append("")
    out.append(
        f"Latest commit `{latest.get('git_sha', '?')}`, "
        f"{fmt_timestamp(latest.get('timestamp_utc',''))}. "
        f"`gen` and `total` are normalised to {n} generated tokens across "
        "the ±1-token tokens_predicted variance; `prompt` is the measured "
        "prompt-eval time from the run. `saved` is negative when DFlash is "
        "slower; `speedup` is `greedy_total / best_dflash_total` (>1× faster)."
    )
    out.append("")
    out.append(
        f"| prompt | greedy<br>prompt | greedy<br>total {n} tok | best DFlash<br>config | best<br>prompt | best<br>total {n} tok | saved | speedup |"
    )
    out.append("|---|---:|---:|---|---:|---:|---:|---:|")
    for r in rows:
        sign = "+" if r["saved_secs"] >= 0 else ""
        g_prompt = f"{r['g_prompt']:.2f} s" if r["g_prompt"] is not None else "n/a"
        b_prompt = f"{r['best_prompt']:.2f} s" if r["best_prompt"] is not None else "n/a"
        out.append(
            f"| {r['prompt']} "
            f"| {g_prompt} "
            f"| {r['g_secs']:.2f} s "
            f"| `{r['best']}` "
            f"| {b_prompt} "
            f"| {r['best_secs']:.2f} s "
            f"| {sign}{r['saved_secs']:.2f} s ({r['saved_pct']:+.1f}%) "
            f"| {r['speedup']:.2f}× |"
        )
    if rows:
        total_saved = sum_g - sum_best
        total_pct = total_saved / sum_g * 100.0 if sum_g else 0.0
        total_spd = sum_g / sum_best if sum_best else 0.0
        sign = "+" if total_saved >= 0 else ""
        out.append(
            f"| **all 5 prompts** | — | **{sum_g:.2f} s** | — | — | **{sum_best:.2f} s** "
            f"| **{sign}{total_saved:.2f} s ({total_pct:+.1f}%)** "
            f"| **{total_spd:.2f}×** |"
        )
    out.append("")
    return "\n".join(out)


def render_group(snaps, hw, n, ub, pair):
    snaps = sorted(
        (s for s in snaps
         if s["_hw"] == hw and s.get("n_predict") == n and s["_ub"] == ub
            and s["_pair"] == pair),
        key=lambda s: (s["_ts"], s["_tag"]),
    )
    if not snaps:
        return ""

    configs = ordered(
        {c for s in snaps for (c, _) in s["_grid"]}, CONFIG_ORDER,
    )
    prompts = ordered(
        {p for s in snaps for (_, p) in s["_grid"]}, PROMPT_ORDER,
    )
    cells = [(c, p) for c in configs for p in prompts]

    suffix = "" if ub == "auto" else f", -ub {ub}"
    lines = []
    lines.append(f"## {hw}, n_predict={n}{suffix} — {pair}")
    lines.append("")
    lines.append(
        f"Host: `{snaps[0].get('host', '?')}`, "
        f"threads={snaps[0].get('threads', '?')}, "
        f"target=`{Path(snaps[0].get('target_model','')).name or '?'}`, "
        f"draft=`{Path(snaps[0].get('draft_model','')).name or '?'}`."
    )
    lines.append("")

    # Two-line header: config above, prompt below (rendered via <br> in
    # markdown — GitHub honours it).
    hdr_cells = [f"{c}<br>{p}" for (c, p) in cells]
    lines.append("| commit | timestamp (UTC) | " + " | ".join(hdr_cells) + " |")
    lines.append("|---|---|" + "---|" * len(cells))

    prev = None
    for s in snaps:
        ts = fmt_timestamp(s.get("timestamp_utc", ""))
        row = [f"`{s['_label']}`", ts]
        for (c, p) in cells:
            curr = s["_grid"].get((c, p))
            pv = prev["_grid"].get((c, p)) if prev else None
            row.append(fmt_cell(curr, pv))
        lines.append("| " + " | ".join(row) + " |")
        prev = s

    lines.append("")
    return "\n".join(lines)


def render(snaps):
    groups = sorted({(s["_hw"], s.get("n_predict"), s["_ub"], s["_pair"]) for s in snaps})

    out = []
    out.append("# DFlash bench timeline")
    out.append("")
    out.append(
        "Per-commit perf snapshots, grouped by (hardware, n_predict, ubatch, "
        "model pair). Auto-generated from `bench/cpu/*.json` and "
        "`bench/gpu/*.json` by `scripts/dflash_bench_render.py`. "
        "Each cell shows `tokens_per_sec`; the parenthesised Δ% is vs the "
        "prior row in the same group."
    )
    out.append("")
    for hw, n, ub, pair in groups:
        timeline = render_group(snaps, hw, n, ub, pair)
        if not timeline:
            continue
        out.append(timeline)
        savings = render_savings(snaps, hw, n, ub, pair)
        if savings:
            out.append(savings)

    return "\n".join(out).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help="output markdown path (default: bench/README.md)")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 if rendered output differs from the on-disk file")
    args = ap.parse_args()

    snaps = load_snapshots()
    if not snaps:
        print("no bench/*/*.json snapshots found; nothing to render",
              file=sys.stderr)
        sys.exit(1)

    rendered = render(snaps)
    out_path = Path(args.out)

    if args.check:
        existing = out_path.read_text() if out_path.exists() else ""
        if existing != rendered:
            print(f"{out_path}: stale (re-run scripts/dflash_bench_render.py)",
                  file=sys.stderr)
            sys.exit(1)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(rendered)
    print(f"wrote {out_path.relative_to(ROOT) if out_path.is_relative_to(ROOT) else out_path}")


if __name__ == "__main__":
    main()
