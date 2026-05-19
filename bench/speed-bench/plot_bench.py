#!/usr/bin/env python3
"""Plot DFlash k-sweep results.

Reads bench.jsonl emitted by run_dflash_k_sweep.py and produces:
  - tps_vs_k.png     (one line per pair × backend, median TPS across prompts)
  - ttft_vs_k.png    (same, but prefill_ms)
  - accept_vs_k.png  (same, but median accept rate)

Each panel includes the greedy AR baseline as a dashed line.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LABELS = {
    "qwen36-27b":    "Qwen 3.6-27B dense",
    "qwen36-35bA3":  "Qwen 3.6-35B-A3B MoE",
    "gemma4-31b":    "Gemma 4-31B dense",
    "gemma4-26bA4":  "Gemma 4-26B-A4B MoE",
}


def load(path: Path) -> pd.DataFrame:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return pd.DataFrame(rows)


def style(pair: str, backend: str):
    """Consistent per-pair color, per-backend linestyle."""
    color_map = {
        "qwen36-27b":   "#1f77b4",
        "qwen36-35bA3": "#ff7f0e",
        "gemma4-31b":   "#2ca02c",
        "gemma4-26bA4": "#d62728",
    }
    return dict(color=color_map.get(pair, "gray"),
                linestyle="-" if backend == "cuda" else "--",
                marker="o" if backend == "cuda" else "s")


def plot_metric(ax, df: pd.DataFrame, value_col: str, ylabel: str, baseline_label: str | None = None):
    spec = df[df["k"] > 0].copy()
    greedy = df[df["k"] == -1].copy()

    for (pair, backend), g in spec.groupby(["pair", "backend"]):
        med = g.groupby("k")[value_col].median().sort_index()
        if med.dropna().empty:
            continue
        label = f"{LABELS.get(pair, pair)} · {backend}"
        ax.plot(med.index, med.values, label=label, **style(pair, backend))

    # Per-(pair, backend) horizontal greedy baseline (single value, faint)
    if baseline_label:
        for (pair, backend), g in greedy.groupby(["pair", "backend"]):
            if g.empty or g["tps"].isna().all():
                continue
            v = float(g["tps"].iloc[0])
            ax.axhline(v, alpha=0.25, lw=1,
                       color=style(pair, backend)["color"],
                       linestyle=":" if backend == "cuda" else (0, (1, 3)))

    ax.set_xlabel("draft tokens per round (k)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="best")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in",  dest="inp", type=Path,
                   default=Path("/tmp/dflash-smoke-full/bench.jsonl"))
    p.add_argument("--out", type=Path,
                   default=Path("/tmp/dflash-smoke-full"))
    args = p.parse_args()

    df = load(args.inp)
    args.out.mkdir(parents=True, exist_ok=True)

    # TPS vs k
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_metric(ax, df, "tps", "median TPS (t/s, across prompts)", baseline_label="greedy")
    ax.set_title("DFlash chain throughput vs draft tokens-per-round\n"
                 "(dotted/dashed thin lines = greedy AR baseline)")
    fig.tight_layout()
    fig.savefig(args.out / "tps_vs_k.png", dpi=120)
    print(f"wrote {args.out / 'tps_vs_k.png'}")

    # TTFT vs k
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_metric(ax, df, "prefill_ms", "median TTFT (prefill_ms, single-stream)")
    ax.set_title("DFlash TTFT (prefill latency) vs draft tokens-per-round")
    fig.tight_layout()
    fig.savefig(args.out / "ttft_vs_k.png", dpi=120)
    print(f"wrote {args.out / 'ttft_vs_k.png'}")

    # Acceptance vs k
    fig, ax = plt.subplots(figsize=(9, 5))
    plot_metric(ax, df, "accept_rate", "median accept rate (%)")
    ax.set_title("DFlash acceptance rate vs draft tokens-per-round")
    fig.tight_layout()
    fig.savefig(args.out / "accept_vs_k.png", dpi=120)
    print(f"wrote {args.out / 'accept_vs_k.png'}")


if __name__ == "__main__":
    main()
