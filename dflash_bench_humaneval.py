#!/usr/bin/env python3
"""DFlash speedup vs greedy on canonical benchmarks (HumanEval / MATH-500 / GSM8K).

Runs `llama-completion` (greedy reference) and `llama-speculative-simple`
(DFlash speculative, multiple configurations) against the same set of prompts.
For each (config, prompt) pair captures throughput (tokens/sec) and acceptance
rate, then aggregates per-config statistics.

The goal of this harness is to answer one question: does our DFlash give a
real speedup on real workloads, or does the speedup pattern depend on prompt
choice? It does NOT measure benchmark correctness (HumanEval pass@1 etc.) --
we only care about latency / throughput here.

Hardware / stack-level comparison with z-lab and lucebox is documented in
`docs/dflash-implementations-comparison.md`; this script produces our own
numbers on the same workloads so the comparison is on the same axis (prompt
distribution) even though hardware / target model / software stack differ.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    name: str
    binary: str   # llama-completion | llama-speculative-simple
    extra_args: list[str] = field(default_factory=list)
    needs_draft: bool = False


def build_configs(use_gdn_history: bool) -> list[Config]:
    # --dflash-gdn-history is required for qwen35-family targets to hit
    # their full acceptance rate but the target context errors out on
    # qwen3-base targets ("model lacks GDN dims"), so it has to be opt-in.
    # Caller passes use_gdn_history based on target path (auto_gdn_history)
    # or the explicit --gdn-history CLI flag.
    DFLASH_COMMON = ["--draft-type", "dflash",
                     "--dflash-no-chat-template"]
    if use_gdn_history:
        DFLASH_COMMON.append("--dflash-gdn-history")
    if os.environ.get("DFLASH_INLINE_ENCODER") == "1":
        DFLASH_COMMON += ["--dflash-inline-encoder", "--dflash-max-ctx", "4096"]
    return [
        Config("greedy", "llama-completion"),
        Config("chain", "llama-speculative-simple",
               list(DFLASH_COMMON), needs_draft=True),
        Config("tree22-uni", "llama-speculative-simple",
               list(DFLASH_COMMON) + ["--dflash-tree", "--dflash-tree-budget", "22"],
               needs_draft=True),
        Config("tree22-bf", "llama-speculative-simple",
               list(DFLASH_COMMON) + ["--dflash-tree", "--dflash-tree-budget", "22",
                                       "--dflash-topk", "4",
                                       "--dflash-tree-best-first"],
               needs_draft=True),
    ]


def auto_gdn_history(target_path: str) -> bool:
    """Heuristic: qwen35/qwen3.6-family targets have GDN layers and benefit
    from --dflash-gdn-history. Qwen3 base / Gemma / LLaMA don't and the
    flag would error out at context init.
    """
    p = target_path.lower()
    return ("qwen3.5" in p) or ("qwen35" in p) or ("qwen3.6" in p) or ("qwen36" in p)


def load_prompts(benchmark: str, limit: int,
                 shuffle_seed: int | None = None) -> list[tuple[str, str]]:
    """Returns [(id, prompt_text), ...] for the requested benchmark.

    If shuffle_seed is set, applies HF `datasets.shuffle(seed=...)` then
    takes the first `limit` rows — this matches lucebox's
    `scripts/bench_llm.py` selection (seed=42, n=10) so cross-fork
    comparisons land on the same prompts.
    """
    from datasets import load_dataset

    if benchmark == "humaneval":
        ds = load_dataset("openai_humaneval", split="test")
        key_id, key_text = "task_id", "prompt"
    elif benchmark == "gsm8k":
        ds = load_dataset("gsm8k", "main", split="test")
        key_id, key_text = None, "question"
    elif benchmark == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
        key_id, key_text = "unique_id", "problem"
    else:
        raise SystemExit(f"unknown benchmark: {benchmark}")

    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
        if limit > 0:
            ds = ds.select(range(limit))

    items = []
    for i, row in enumerate(ds):
        pid = (row.get(key_id) if key_id else None) or f"{benchmark}_{i}"
        items.append((pid, row[key_text]))

    if shuffle_seed is None and limit > 0:
        items = items[:limit]
    return items


_RE_DECODED = re.compile(r"^decoded\s+\d+\s+tokens in\s+([\d.]+)\s+seconds,\s+speed:\s+([\d.]+)\s+t/s",
                        re.MULTILINE)
_RE_N_DRAFTED = re.compile(r"^n_drafted\s*=\s*(\d+)", re.MULTILINE)
_RE_N_ACCEPT = re.compile(r"^n_accept\s*=\s*(\d+)", re.MULTILINE)
_RE_COMMIT_STEP = re.compile(r"^commit/step\s*=\s*([\d.]+)", re.MULTILINE)
# llama-completion uses a different format
_RE_GREEDY_EVAL = re.compile(
    r"common_perf_print:\s+eval time =\s+([\d.]+)\s+ms\s+/\s+(\d+)\s+runs",
)


def _round_up_256(n: int) -> int:
    return ((n + 255) // 256) * 256


def auto_ctx(prompt: str, n_predict: int, pad: int = 64) -> int:
    """Per-prompt context size: tight upper bound on prompt + generation.

    Adapted from lucebox/scripts/bench_llm.py::_auto_max_ctx. Oversizing
    n_ctx makes flash-attn stride over unused KV cells, which on short
    prompts dominates the per-step wall (a 225-char HE prompt at n_ctx=4096
    pays ~16× the stride cost of n_ctx=256). With auto-fit, the same code
    on the same GPU reports stable t/s across prompts of different lengths.

    Heuristic prompt length: 0.5 tok/char + small floor (Qwen / Llama
    BPE-style tokenizers average 0.25-0.35 tok/char on English; 0.5 is
    a conservative upper bound).
    """
    n_prompt_est = max(64, int(len(prompt) * 0.5) + 1)
    return _round_up_256(n_prompt_est + n_predict + pad)


@dataclass
class Sample:
    config: str
    prompt_id: str
    elapsed_s: float
    tokens_per_sec: float
    n_drafted: int = 0
    n_accept: int = 0
    commit_per_step: float = 0.0
    n_ctx: int = 0
    err: str | None = None


def run_one(cfg: Config, prompt: str, target: str, draft: str | None,
            n_predict: int, threads: int, ngl: int, build_bin: Path,
            ctx: int, timeout_s: int) -> Sample:
    bin_path = build_bin / cfg.binary
    cmd: list[str] = [
        str(bin_path),
        "-m", target,
        "-p", prompt,
        "--n-predict", str(n_predict),
        "--temp", "0", "--seed", "0",
        "-c", str(ctx),
    ]
    if ngl > 0:
        cmd += ["-ngl", str(ngl)]
        if cfg.needs_draft:
            cmd += ["-ngld", str(ngl)]
    if threads > 0:
        cmd += ["--threads", str(threads)]
    if cfg.needs_draft:
        assert draft is not None
        cmd += ["-md", draft]
    if cfg.binary == "llama-completion":
        cmd += ["-no-cnv", "--no-display-prompt"]
    cmd += cfg.extra_args

    t0 = time.monotonic()
    try:
        res = subprocess.run(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            timeout=timeout_s, check=False,
        )
    except subprocess.TimeoutExpired:
        return Sample(cfg.name, "", time.monotonic() - t0, 0.0,
                      err="timeout")
    # Decode lossily: model output may produce mid-stream UTF-8 fragments
    # (multi-byte tokens split across chunks) that are not decodable on their
    # own. The metric lines we care about are pure ASCII so this is safe.
    out = res.stdout.decode("utf-8", errors="replace")

    if cfg.binary == "llama-speculative-simple":
        m = _RE_DECODED.search(out)
        if not m:
            return Sample(cfg.name, "", time.monotonic() - t0, 0.0,
                          err="no decoded line")
        elapsed = float(m.group(1))
        tps = float(m.group(2))
        n_drafted = int((_RE_N_DRAFTED.search(out) or [None, "0"])[1])
        n_accept = int((_RE_N_ACCEPT.search(out) or [None, "0"])[1])
        cs_m = _RE_COMMIT_STEP.search(out)
        commit_per_step = float(cs_m.group(1)) if cs_m else 0.0
        return Sample(cfg.name, "", elapsed, tps, n_drafted, n_accept,
                      commit_per_step, ctx)
    else:
        m = _RE_GREEDY_EVAL.search(out)
        if not m:
            return Sample(cfg.name, "", time.monotonic() - t0, 0.0,
                          err="no eval line")
        elapsed_ms = float(m.group(1))
        n_runs = int(m.group(2))
        elapsed = elapsed_ms / 1000.0
        tps = (n_runs / elapsed) if elapsed > 0 else 0.0
        return Sample(cfg.name, "", elapsed, tps, n_ctx=ctx)


def summarise(samples: list[Sample], greedy_by_id: dict[str, float]) -> dict:
    by_cfg: dict[str, list[Sample]] = {}
    for s in samples:
        by_cfg.setdefault(s.config, []).append(s)

    summary = {}
    for cfg_name, group in by_cfg.items():
        good = [s for s in group if s.err is None and s.tokens_per_sec > 0]
        if not good:
            summary[cfg_name] = {"n": 0, "errors": len(group)}
            continue
        tps_vals = [s.tokens_per_sec for s in good]
        speedups = []
        for s in good:
            g = greedy_by_id.get(s.prompt_id)
            if g and g > 0:
                speedups.append(s.tokens_per_sec / g)
        accept_rates = [
            s.n_accept / s.n_drafted
            for s in good
            if s.n_drafted > 0
        ]
        commit_steps = [s.commit_per_step for s in good if s.commit_per_step > 0]
        summary[cfg_name] = {
            "n": len(good),
            "errors": len(group) - len(good),
            "tps_mean":    round(statistics.mean(tps_vals), 2),
            "tps_median":  round(statistics.median(tps_vals), 2),
            "speedup_mean":   round(statistics.mean(speedups), 3) if speedups else None,
            "speedup_median": round(statistics.median(speedups), 3) if speedups else None,
            "speedup_p25": round(statistics.quantiles(speedups, n=4)[0], 3) if len(speedups) >= 4 else None,
            "speedup_p75": round(statistics.quantiles(speedups, n=4)[2], 3) if len(speedups) >= 4 else None,
            "accept_mean": round(statistics.mean(accept_rates), 3) if accept_rates else None,
            "commit_per_step_median": round(statistics.median(commit_steps), 3) if commit_steps else None,
        }
    return summary


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", required=True,
                   choices=["humaneval", "gsm8k", "math500"])
    p.add_argument("--target", required=True, help="path to target gguf")
    p.add_argument("--draft", required=True, help="path to draft gguf")
    p.add_argument("--build-dir", default="build", help="cmake build dir under repo")
    p.add_argument("--n-predict", type=int, default=128)
    p.add_argument("--ctx", type=int, default=4096,
                   help="fixed n_ctx for every run. Ignored when --auto-ctx is set.")
    p.add_argument("--auto-ctx", action="store_true",
                   help="per-prompt n_ctx sizing (lucebox-style): "
                        "round_up_256(prompt_est + n_predict + 64). "
                        "Avoids the flash-attn-strides-over-unused-KV cost on "
                        "short prompts; makes throughput stable across prompt lengths.")
    p.add_argument("--ngl", type=int, default=99)
    p.add_argument("--threads", type=int, default=0)
    p.add_argument("--limit", type=int, default=0,
                   help="cap number of prompts (0 = all)")
    p.add_argument("--shuffle-seed", type=int, default=None,
                   help="if set, HF shuffle(seed) before taking --limit. "
                        "Matches lucebox's scripts/bench_llm.py (seed=42).")
    p.add_argument("--timeout-s", type=int, default=120)
    p.add_argument("--configs", default="all",
                   help="comma-sep config names or 'all'")
    p.add_argument("--out", default=None,
                   help="output JSON path (default: bench/humaneval/<bench>_<tag>.json)")
    p.add_argument("--tag", default="run",
                   help="short tag distinguishing runs (e.g. 'qwen3-4b-vulkan')")
    p.add_argument("--gdn-history", choices=["auto", "on", "off"], default="auto",
                   help="pass --dflash-gdn-history to spec configs. "
                        "'auto' (default): on for qwen35/qwen3.6 targets by path, "
                        "off otherwise (qwen3 base errors with this flag). "
                        "'on' forces, 'off' disables.")
    args = p.parse_args()

    build_bin = REPO_ROOT / args.build_dir / "bin"
    if not (build_bin / "llama-completion").exists():
        print(f"ERROR: {build_bin}/llama-completion not found", file=sys.stderr)
        return 1

    if args.gdn_history == "auto":
        use_gdn_history = auto_gdn_history(args.target)
    else:
        use_gdn_history = (args.gdn_history == "on")
    print(f"gdn-history: {use_gdn_history} ({args.gdn_history})", file=sys.stderr)

    all_cfgs = build_configs(use_gdn_history)
    if args.configs != "all":
        wanted = set(args.configs.split(","))
        cfgs = [c for c in all_cfgs if c.name in wanted]
    else:
        cfgs = all_cfgs

    prompts = load_prompts(args.benchmark, args.limit, args.shuffle_seed)
    print(f"benchmark={args.benchmark} n_prompts={len(prompts)} "
          f"configs=[{','.join(c.name for c in cfgs)}] target={args.target} "
          f"draft={args.draft} n_predict={args.n_predict} ngl={args.ngl}")

    samples: list[Sample] = []
    t_total = time.monotonic()
    for i, (pid, ptext) in enumerate(prompts):
        ctx_for_prompt = auto_ctx(ptext, args.n_predict) if args.auto_ctx else args.ctx
        for cfg in cfgs:
            t0 = time.monotonic()
            s = run_one(
                cfg, ptext, args.target, args.draft if cfg.needs_draft else None,
                args.n_predict, args.threads, args.ngl, build_bin,
                ctx_for_prompt, args.timeout_s,
            )
            s.prompt_id = pid
            samples.append(s)
            dt = time.monotonic() - t0
            extra = f" ctx={s.n_ctx}" if args.auto_ctx else ""
            cs = f" cs={s.commit_per_step:.2f}" if s.commit_per_step > 0 else ""
            tag = f"err={s.err}" if s.err else f"{s.tokens_per_sec:.1f}t/s{cs}"
            print(f"  [{i+1}/{len(prompts)}] {pid:>20s} {cfg.name:<12s} "
                  f"{tag:<22s} ({dt:.1f}s){extra}", flush=True)

    greedy_by_id: dict[str, float] = {
        s.prompt_id: s.tokens_per_sec
        for s in samples
        if s.config == "greedy" and s.tokens_per_sec > 0
    }
    summary = summarise(samples, greedy_by_id)

    out_path = (Path(args.out) if args.out
                else REPO_ROOT / "bench" / "humaneval" /
                     f"{args.benchmark}_{args.tag}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({
            "benchmark": args.benchmark,
            "target": args.target,
            "draft": args.draft,
            "n_predict": args.n_predict,
            "ngl": args.ngl,
            "auto_ctx": args.auto_ctx,
            "ctx": args.ctx,
            "n_prompts": len(prompts),
            "elapsed_total_s": round(time.monotonic() - t_total, 1),
            "summary": summary,
            "samples": [
                {
                    "config": s.config, "prompt_id": s.prompt_id,
                    "elapsed_s": round(s.elapsed_s, 3),
                    "tokens_per_sec": round(s.tokens_per_sec, 2),
                    "n_drafted": s.n_drafted, "n_accept": s.n_accept,
                    "commit_per_step": round(s.commit_per_step, 3),
                    "n_ctx": s.n_ctx,
                    "err": s.err,
                }
                for s in samples
            ],
        }, f, indent=2)
    print(f"\nwrote {out_path}")

    print("\nSummary:")
    print(f"  {'config':<12s} {'n':>4s} {'tps_med':>9s} {'sp_med':>8s} "
          f"{'sp_p25':>8s} {'sp_p75':>8s} {'accept':>8s} {'cmt/step':>9s}")
    for cfg in cfgs:
        s = summary.get(cfg.name, {})
        print(f"  {cfg.name:<12s} {s.get('n', 0):>4d} "
              f"{s.get('tps_median', 0):>9.2f} "
              f"{s.get('speedup_median') or 0:>8.3f} "
              f"{s.get('speedup_p25') or 0:>8.3f} "
              f"{s.get('speedup_p75') or 0:>8.3f} "
              f"{(s.get('accept_mean') or 0):>8.3f} "
              f"{(s.get('commit_per_step_median') or 0):>9.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
