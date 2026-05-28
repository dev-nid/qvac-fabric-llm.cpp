#!/usr/bin/env python3
"""
DFlash k-sweep on SPEED-Bench (qualitative split).

Mirrors the layout of jarvislabs.ai/blog/gemma-4-mtp-vs-dflash-benchmark
adapted for single-stream llama.cpp (concurrency = 1):

  For each (model_pair, backend, k):
    * sample N prompts from SPEED-Bench, default 1 per category (11 prompts).
    * run llama-speculative-simple --draft-type dflash --dflash-block-size=k
    * collect: TPS, prefill_ms (TTFT proxy), n_drafted, n_accept, AL.
  Plus a greedy baseline per (model_pair, backend) via llama-bench.

Reusable: every knob is a CLI flag. Defaults give a fast smoke run
(11 prompts, k in {4,8,16,32}, n_predict=512); pass --full for the
Jarvis-shape 880-prompt sweep with n_predict=4096.

Output:
  --outdir/bench.jsonl         (one line per (pair, backend, k, prompt))
  --outdir/summary.md          (Format-B style per-model tables)
"""
from __future__ import annotations

import argparse
import dataclasses
import datetime as _dt
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


# ---------- defaults --------------------------------------------------------

# All paths are env-var-overridable so the script ports cleanly to a fresh
# box. Defaults derive from the location of this script (REPO root) and a
# BENCH_DATA_DIR sibling that setup.sh populates.
REPO       = Path(os.environ.get("BENCH_REPO",       Path(__file__).resolve().parents[2]))
DATA_DIR   = Path(os.environ.get("BENCH_DATA_DIR",   REPO.parent / "bench-data"))
MODELS_Q4  = Path(os.environ.get("BENCH_MODELS_Q4",  DATA_DIR / "models" / "bench-q4"))
MODELS_Q6  = Path(os.environ.get("BENCH_MODELS_Q6",  DATA_DIR / "models" / "bench-q6"))
MODELS_Q8  = Path(os.environ.get("BENCH_MODELS_Q8",  DATA_DIR / "models" / "bench-q8"))
DRAFTS_DIR = Path(os.environ.get("BENCH_DRAFTS_DIR", DATA_DIR / "models"))
ASSIST_DIR = Path(os.environ.get("BENCH_ASSIST_DIR", DATA_DIR / "models" / "bench-q6-assistant"))
SPEED_BENCH = Path(os.environ.get("BENCH_SPEEDBENCH",
                                  DATA_DIR / "speed-bench" / "qualitative" /
                                  "test-00000-of-00001.parquet"))

# (pair_id, target_gguf_by_quant, draft_gguf, family_label, draft_type)
# target_gguf_by_quant maps "q4_k_m" / "q6_k" / "q8_0" -> (models_dir, filename)
# draft_type: "dflash" (default) or "draft" for vanilla speculative with a small draft model
QWEN3_8B_Q8 = Path(os.environ.get("BENCH_QWEN3_8B_Q8",
                                  DATA_DIR / "models" / "qwen3-8b-q8"))

DEFAULT_PAIRS = [
    ("qwen36-27b", {
        "q4_k_m": (MODELS_Q4, "Qwen3.6-27B-Q4_K_M.gguf"),
        "q6_k":   (MODELS_Q6, "Qwen3.6-27B-Q6_K.gguf"),
        "q8_0":   (MODELS_Q8, "Qwen3.6-27B-Q8_0.gguf"),
     }, "Qwen3.6-27B-DFlash.gguf", "Qwen 3.6-27B dense", "dflash"),
    # Qwen 3-8B Q8_0 sanity-reference pair matching the historical report's
    # "Qwen 3-8B sanity reference" section. Target = Qwen3-8B-Q8_0,
    # draft = z-lab/Qwen3-8B-DFlash-b16 converted to GGUF.
    ("qwen3-8b", {
        "q8_0":   (QWEN3_8B_Q8, "Qwen3-8B-Q8_0.gguf"),
     }, "Qwen3-8B-DFlash-bf16.gguf", "Qwen 3-8B dense", "dflash"),
    ("qwen36-35bA3", {
        "q4_k_m": (MODELS_Q4, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
        "q6_k":   (MODELS_Q6, "Qwen3.6-35B-A3B-UD-Q6_K.gguf"),
        # Q8_0 would be ~37GB > 32GB VRAM, skipped
     }, "Qwen3.6-35B-A3B-DFlash.gguf", "Qwen 3.6-35B-A3B MoE", "dflash"),
    ("gemma4-31b", {
        "q4_k_m": (MODELS_Q4, "gemma-4-31B-it-Q4_K_M.gguf"),
        "q6_k":   (MODELS_Q6, "gemma-4-31B-it-Q6_K.gguf"),
        "q8_0":   (MODELS_Q8, "gemma-4-31B-it-Q8_0.gguf"),
     }, "gemma-4-31B-it-DFlash.gguf", "Gemma 4-31B dense", "dflash"),
    ("gemma4-26bA4", {
        "q4_k_m": (MODELS_Q4, "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
        "q6_k":   (MODELS_Q6, "gemma-4-26B-A4B-it-UD-Q6_K.gguf"),
        "q8_0":   (MODELS_Q8, "gemma-4-26B-A4B-it-Q8_0.gguf"),
     }, "gemma-4-26B-A4B-it-DFlash.gguf", "Gemma 4-26B-A4B MoE", "dflash"),

    # Vanilla speculative decoding with Gemma 4 assistant drafts (what
    # vLLM exposes as "MTP" for Gemma 4 — Gemma 4 has no native MTP head).
    # Currently NOT supported in upstream llama.cpp (community GGUFs use
    # non-standard archs gemma4_assistant / gemma4_mtp). Kept here for
    # when arch support lands.
    ("gemma4-31b-asst", {
        "q4_k_m": (MODELS_Q4, "gemma-4-31B-it-Q4_K_M.gguf"),
        "q6_k":   (MODELS_Q6, "gemma-4-31B-it-Q6_K.gguf"),
        "q8_0":   (MODELS_Q8, "gemma-4-31B-it-Q8_0.gguf"),
     },
     str(ASSIST_DIR / "gemma-4-31B-asst" / "gemma-4-31B-it-assistant.Q8_0.gguf"),
     "Gemma 4-31B dense (assistant draft)", "draft"),
    ("gemma4-26bA4-asst", {
        "q4_k_m": (MODELS_Q4, "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf"),
        "q6_k":   (MODELS_Q6, "gemma-4-26B-A4B-it-UD-Q6_K.gguf"),
        "q8_0":   (MODELS_Q8, "gemma-4-26B-A4B-it-Q8_0.gguf"),
     },
     str(ASSIST_DIR / "gemma-4-26B-A4B-asst" / "gemma-4-26B-A4B-it-assistant.Q8_0.gguf"),
     "Gemma 4-26B-A4B MoE (assistant draft)", "draft"),
]

DEFAULT_K = [2, 4, 8, 16]  # GGUF block_size is 16 for current drafts, k>16 is no-op


# ---------- result row ------------------------------------------------------

@dataclasses.dataclass
class Row:
    pair: str
    quant: str
    backend: str
    k: int
    prompt_idx: int
    category: str
    n_prompt_tokens: int
    tps: float | None
    prefill_ms: float | None
    n_drafted: int | None
    n_accept: int | None
    accept_rate: float | None
    al: float | None  # avg accepted tokens per draft round
    ok: bool
    raw_log: str


# ---------- helpers ---------------------------------------------------------

def parse_spec_log(log: str) -> dict:
    """Pull bench numbers out of llama-speculative-simple stderr."""
    out: dict = {}

    m = re.search(r"decoded\s+(\d+)\s+tokens in\s+([\d.]+)\s+seconds,\s+speed:\s+([\d.]+)\s+t/s", log)
    if m:
        out["tps"] = float(m.group(3))
        out["n_decoded"] = int(m.group(1))

    m = re.search(r"n_drafted\s*=\s*(\d+)", log)
    if m:
        out["n_drafted"] = int(m.group(1))
    m = re.search(r"n_accept\s*=\s*(\d+)", log)
    if m:
        out["n_accept"] = int(m.group(1))

    # prompt-eval time = single-stream TTFT proxy (no queueing component)
    m = re.search(r"prompt eval time =\s+([\d.]+)\s+ms /\s+(\d+)\s+tokens", log)
    if m:
        out["prefill_ms"] = float(m.group(1))
        out["n_prompt_tokens"] = int(m.group(2))

    return out


def parse_greedy_log(log: str) -> float | None:
    """llama-bench prints a table; pull the t/s value from the tg<N> row."""
    for line in log.splitlines():
        if "tg" in line and "|" in line:
            parts = [p.strip() for p in line.split("|")]
            # last non-empty cell is "<float> ± <float>"; first token is the t/s
            for cell in reversed(parts):
                if cell and cell[0].isdigit():
                    try:
                        return float(cell.split()[0])
                    except ValueError:
                        pass
    return None


def run_cmd(cmd: list[str], env: dict | None = None, timeout: int = 900) -> tuple[int, str]:
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=timeout)
    return p.returncode, (p.stdout + "\n" + p.stderr)


def pick_subset(df: pd.DataFrame, per_category: int, seed: int) -> pd.DataFrame:
    """Stratified sample: per_category rows per category, deterministic.

    Explicit loop (avoids the pandas 2.x groupby.apply behaviour that
    drops the grouping column from the result).
    """
    parts = []
    for cat, g in df.groupby("category", sort=True):
        parts.append(g.sample(n=min(per_category, len(g)), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def prompt_text(row: pd.Series) -> str:
    """SPEED-Bench prompts are stored under 'turns' (list of strings)."""
    turns = row["turns"]
    if isinstance(turns, str):
        return turns
    if hasattr(turns, "tolist"):
        turns = turns.tolist()
    return turns[0] if turns else ""


# ---------- bench loop ------------------------------------------------------

def bench_pair(args, pair, quant, backend, prompts_df, results: list[Row]):
    if len(pair) == 5:
        pair_id, targets_by_quant, draft_name, family, draft_type = pair
    else:
        pair_id, targets_by_quant, draft_name, family = pair
        draft_type = "dflash"
    if quant not in targets_by_quant:
        print(f"  [skip] {pair_id} has no {quant} target defined", file=sys.stderr)
        return
    models_dir, target_name = targets_by_quant[quant]
    target = models_dir / target_name
    # draft_name may be a bare filename (relative to DRAFTS_DIR) or an absolute path
    draft = Path(draft_name) if Path(draft_name).is_absolute() else (DRAFTS_DIR / draft_name)

    if not target.exists():
        print(f"  [skip] {target} not present", file=sys.stderr)
        return
    if not draft.exists():
        print(f"  [skip] {draft} not present", file=sys.stderr)
        return

    # Build dirs are overridable via env vars; defaults match our usual layout.
    # Strix Halo / AMD users: build with -DGGML_HIP=ON into build-hip/ and
    # invoke with --backends hip (auto-pin via ROCR_VISIBLE_DEVICES=0).
    if backend == "cuda":
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": os.environ.get("BENCH_GPU_INDEX", "0")}
        build_dir = Path(os.environ.get("BENCH_CUDA_BUILD",   REPO / "build-cuda"))
    elif backend == "vulkan":
        env = {**os.environ, "GGML_VK_VISIBLE_DEVICES": os.environ.get("BENCH_GPU_INDEX", "0")}
        build_dir = Path(os.environ.get("BENCH_VULKAN_BUILD", REPO / "build"))
    elif backend == "hip":
        env = {**os.environ, "ROCR_VISIBLE_DEVICES": os.environ.get("BENCH_GPU_INDEX", "0")}
        build_dir = Path(os.environ.get("BENCH_HIP_BUILD",    REPO / "build-hip"))
    else:
        raise ValueError(f"unknown backend: {backend}")

    bench_bin = str(build_dir / "bin" / "llama-bench")
    spec_bin  = str(build_dir / "bin" / "llama-speculative-simple")
    if not Path(bench_bin).exists():
        print(f"  [skip] {bench_bin} not built; backend={backend} unavailable", file=sys.stderr)
        return

    # --- greedy baseline -----------------------------------------------------
    rc, log = run_cmd(
        [bench_bin, "-m", str(target),
         "-p", "0", "-n", str(args.n_predict), "-ngl", "99", "-t", "8", "-r", "3"],
        env=env, timeout=600,
    )
    greedy_tps = parse_greedy_log(log)
    print(f"  greedy: {greedy_tps} t/s", flush=True)
    results.append(Row(
        pair=pair_id, quant=quant, backend=backend, k=-1, prompt_idx=-1,
        category="(greedy)", n_prompt_tokens=0, tps=greedy_tps,
        prefill_ms=None, n_drafted=None, n_accept=None,
        accept_rate=None, al=None, ok=(greedy_tps is not None), raw_log="",
    ))

    # --- per-k, per-prompt ---------------------------------------------------
    for k in args.k_values:
        for i, row in prompts_df.iterrows():
            ptxt = prompt_text(row)
            cmd = [
                spec_bin,
                "-m", str(target), "-md", str(draft), "--draft-type", draft_type,
                "-p", ptxt,
                "--n-predict", str(args.n_predict),
                "--temp", "0", "--seed", str(args.seed),
                "-c", str(args.ctx), "-cd", str(args.ctx),
                "-ngl", "99", "-ngld", "99", "-t", "8",
            ]
            if args.mode == "tree":
                cmd += ["--dflash-tree", "--dflash-tree-budget", str(k)]
            else:
                cmd += ["--spec-draft-n-max", str(k)]
            # auto-enable --dflash-gdn-history for hybrid-attn targets (Qwen 3.5/3.6).
            wants_gdn = args.gdn_history == "on" or (
                args.gdn_history == "auto" and pair_id.startswith(("qwen35", "qwen36")))
            if wants_gdn:
                cmd += ["--dflash-gdn-history"]
            if os.environ.get("DFLASH_INLINE_ENCODER") == "1":
                cmd += ["--dflash-inline-encoder", "--dflash-max-ctx", "4096"]
            rc, log = run_cmd(cmd, env=env, timeout=args.timeout)
            stats = parse_spec_log(log)
            n_drafted = stats.get("n_drafted")
            n_accept = stats.get("n_accept")
            accept_rate = (100.0 * n_accept / n_drafted) if n_drafted else None
            # AL = avg accepted tokens per draft round (k tokens drafted per round)
            al = (n_accept / (n_drafted / k)) if (n_drafted and k) else None
            tps = stats.get("tps")
            prefill = stats.get("prefill_ms")
            ok = (rc == 0 and tps is not None)

            # Save raw log when something goes wrong so we can diagnose.
            raw_log_path = ""
            if not ok:
                args.outdir.mkdir(parents=True, exist_ok=True)
                fail_log = args.outdir / f"fail_{pair_id}_{backend}_k{k}_p{i}.log"
                fail_log.write_text(log)
                raw_log_path = str(fail_log)

            results.append(Row(
                pair=pair_id, quant=quant, backend=backend, k=k, prompt_idx=int(i),
                category=row.get("category", "?"),
                n_prompt_tokens=stats.get("n_prompt_tokens", 0),
                tps=tps,
                prefill_ms=prefill,
                n_drafted=n_drafted, n_accept=n_accept,
                accept_rate=accept_rate, al=al,
                ok=ok,
                raw_log=raw_log_path,
            ))
            tps_s = f"{tps:7.2f}" if tps is not None else "    FAIL"
            pre_s = f"{prefill:7.1f}" if prefill is not None else "    NA"
            acc_s = f"{n_accept}/{n_drafted}" if n_drafted is not None else "NA"
            print(f"    k={k:>3} cat={row['category']:<14} tps={tps_s} "
                  f"prefill={pre_s}ms accept={acc_s}", flush=True)


# ---------- output formatting ----------------------------------------------

def write_outputs(args, results: list[Row]):
    args.outdir.mkdir(parents=True, exist_ok=True)
    jsonl = args.outdir / "bench.jsonl"
    with jsonl.open("w") as f:
        for r in results:
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")
    print(f"\nwrote {jsonl}")

    df = pd.DataFrame([dataclasses.asdict(r) for r in results])
    md = args.outdir / "summary.md"
    with md.open("w") as f:
        f.write(f"# DFlash k-sweep on SPEED-Bench\n\n")
        f.write(f"_Generated {_dt.datetime.utcnow().isoformat()}Z_\n\n")
        f.write(f"- prompts/category: {args.per_category} (total {args.per_category*11})\n")
        f.write(f"- n_predict: {args.n_predict}\n")
        f.write(f"- k values: {args.k_values}\n")
        f.write(f"- backends: {args.backends}\n\n")

        group_cols = ["pair", "quant", "backend"] if "quant" in df.columns else ["pair", "backend"]
        for keys, g in df.groupby(group_cols):
            if len(group_cols) == 3:
                pair_id, quant, backend = keys
                title = f"{pair_id} · {quant} · {backend}"
            else:
                pair_id, backend = keys
                title = f"{pair_id} · {backend}"
            greedy_row = g[g["k"] == -1]
            greedy = float(greedy_row["tps"].iloc[0]) if not greedy_row.empty else float("nan")
            spec = g[g["k"] >= 0]
            f.write(f"## {title}\n\n")
            f.write(f"greedy: {greedy:.2f} t/s\n\n")

            # k × prompt grid (median across categories per k)
            piv_tps = spec.pivot_table(index="k", columns="category", values="tps", aggfunc="median")
            piv_acc = spec.pivot_table(index="k", columns="category", values="accept_rate", aggfunc="median")
            piv_ttft = spec.pivot_table(index="k", columns="category", values="prefill_ms", aggfunc="median")

            f.write("### TPS (t/s) by k × category\n\n")
            f.write(piv_tps.to_markdown(floatfmt=".1f") + "\n\n")
            f.write("### Accept rate (%) by k × category\n\n")
            f.write(piv_acc.to_markdown(floatfmt=".1f") + "\n\n")
            f.write("### TTFT (prefill_ms, single-stream) by k × category\n\n")
            f.write(piv_ttft.to_markdown(floatfmt=".1f") + "\n\n")

            # summary row: median across all prompts per k
            agg = spec.groupby("k").agg(tps_med=("tps", "median"),
                                        accept_med=("accept_rate", "median"),
                                        prefill_med=("prefill_ms", "median")).reset_index()
            agg["speedup_vs_greedy"] = agg["tps_med"] / greedy
            f.write("### Headline (median across the prompt set)\n\n")
            f.write(agg.to_markdown(index=False, floatfmt=".2f") + "\n\n")

    print(f"wrote {md}")


# ---------- main ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", nargs="+", default=[x[0] for x in DEFAULT_PAIRS],
                   help="pair ids to bench (default: all 4)")
    p.add_argument("--quants", nargs="+", default=["q6_k"],
                   choices=["q4_k_m", "q6_k", "q8_0"],
                   help="target quants to sweep (default: q6_k)")
    p.add_argument("--backends", nargs="+", default=["cuda", "vulkan"],
                   choices=["cuda", "vulkan", "hip"],
                   help="hip = AMD ROCm/HIP build (Strix Halo + dGPU). "
                        "Override build dirs with BENCH_{CUDA,VULKAN,HIP}_BUILD env vars.")
    p.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K)
    p.add_argument("--per-category", type=int, default=1,
                   help="prompts per SPEED-Bench category (1 = 11 prompts total)")
    p.add_argument("--n-predict", type=int, default=512)
    p.add_argument("--mode", choices=["chain", "tree"], default="chain",
                   help="chain: k -> --spec-draft-n-max; tree: k -> --dflash-tree-budget")
    p.add_argument("--gdn-history", choices=["auto", "on", "off"], default="auto",
                   help="add --dflash-gdn-history. 'auto' enables it for qwen35/qwen36 pairs")
    p.add_argument("--ctx", type=int, default=4096)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--outdir", type=Path, default=REPO / "bench/speed-bench/results")
    p.add_argument("--speedbench", type=Path, default=SPEED_BENCH)
    p.add_argument("--full", action="store_true",
                   help="Jarvis-shape: per_category=80, n_predict=4096")
    args = p.parse_args()

    if args.full:
        args.per_category = 80
        args.n_predict = 4096

    df = pd.read_parquet(args.speedbench)
    prompts_df = pick_subset(df, args.per_category, args.seed)
    print(f"selected {len(prompts_df)} prompts ({args.per_category}/cat × 11 categories)")

    pairs_by_id = {p[0]: p for p in DEFAULT_PAIRS}
    selected_pairs = [pairs_by_id[p] for p in args.pairs if p in pairs_by_id]

    results: list[Row] = []
    t0 = time.time()
    for pair in selected_pairs:
        for quant in args.quants:
            for backend in args.backends:
                print(f"\n=== {pair[3]} · {quant} · {backend} ===")
                bench_pair(args, pair, quant, backend, prompts_df, results)
    print(f"\ntotal wall time: {time.time() - t0:.1f}s")

    write_outputs(args, results)


if __name__ == "__main__":
    main()
