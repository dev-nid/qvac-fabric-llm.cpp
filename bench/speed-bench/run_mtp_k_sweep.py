#!/usr/bin/env python3
"""
Upstream llama.cpp MTP k-sweep on SPEED-Bench.

Mirrors run_dflash_k_sweep.py's output schema so the two JSONLs can be
merged for head-to-head DFlash-vs-MTP comparison.

For each (target, backend, k):
  - launch upstream llama-server with --spec-type draft-mtp --spec-draft-n-max k
  - send each SPEED-Bench prompt as a /completion request with stream=true
  - parse SSE stream: TTFT = time to first token chunk,
                      TPS  = generated_tokens / (last_chunk_time - first_chunk_time)
  - acceptance numbers come from /metrics endpoint
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
import urllib.request
import urllib.error
from pathlib import Path
import contextlib
import signal
import socket

import pandas as pd
import requests


# All paths overridable via env vars. Defaults derive from script location +
# a sibling BENCH_DATA_DIR.
REPO        = Path(os.environ.get("BENCH_REPO",      Path(__file__).resolve().parents[2]))
DATA_DIR    = Path(os.environ.get("BENCH_DATA_DIR",  REPO.parent / "bench-data"))
MTP_Q4_DIR  = Path(os.environ.get("BENCH_MTP_Q4_DIR", DATA_DIR / "models" / "bench-q4-mtp"))
MTP_Q6_DIR  = Path(os.environ.get("BENCH_MTP_Q6_DIR", DATA_DIR / "models" / "bench-q6-mtp"))
MTP_Q8_DIR  = Path(os.environ.get("BENCH_MTP_Q8_DIR", DATA_DIR / "models" / "bench-q8-mtp"))
SPEED_BENCH = Path(os.environ.get("BENCH_SPEEDBENCH",
                                  DATA_DIR / "speed-bench" / "qualitative" /
                                  "test-00000-of-00001.parquet"))
UPSTREAM_REPO = Path(os.environ.get("UPSTREAM_LLAMA_CPP", "/tmp/llama-cpp-master"))

# pair_id, target_gguf_by_quant, family_label
DEFAULT_PAIRS = [
    ("qwen36-27b-mtp", {
        "q4_k_m": (MTP_Q4_DIR, "Qwen3.6-27B-Q4_K_M.gguf"),
        "q6_k":   (MTP_Q6_DIR, "Qwen3.6-27B-Q6_K.gguf"),
        "q8_0":   (MTP_Q8_DIR, "Qwen3.6-27B-Q8_0.gguf"),
     }, "Qwen 3.6-27B dense (MTP)"),
    ("qwen36-35bA3-mtp", {
        "q4_k_m": (MTP_Q4_DIR, "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"),
        "q6_k":   (MTP_Q6_DIR, "Qwen3.6-35B-A3B-UD-Q6_K.gguf"),
        # Q8_0 too large for 32 GB GPU with MTP draft state
     }, "Qwen 3.6-35B-A3B MoE (MTP)"),
]

DEFAULT_K = [2, 4, 8, 16]


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
    prefill_ms: float | None  # TTFT
    n_drafted: int | None
    n_accept: int | None
    accept_rate: float | None
    al: float | None
    ok: bool
    raw_log: str


# ---------- server lifecycle ------------------------------------------------

def find_free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@contextlib.contextmanager
def llama_server(server_bin: Path, target: Path, backend: str, k: int,
                 ctx: int, log_path: Path):
    """Start llama-server with MTP enabled and yield (url, server_proc)."""
    port = find_free_port()
    env = os.environ.copy()
    gpu_idx = os.environ.get("BENCH_GPU_INDEX", "0")
    if backend == "cuda":
        env["CUDA_VISIBLE_DEVICES"] = gpu_idx
    elif backend == "vulkan":
        env["GGML_VK_VISIBLE_DEVICES"] = gpu_idx
    elif backend == "hip":
        env["ROCR_VISIBLE_DEVICES"] = gpu_idx

    cmd = [
        str(server_bin),
        "-m", str(target),
        "--spec-type", "draft-mtp",
        "--spec-draft-n-max", str(k),
        "-c", str(ctx),
        "-ngl", "99",
        "-t", "8",
        "--host", "127.0.0.1",
        "--port", str(port),
        # default scheduler; no batching needed at concurrency=1
        "--metrics",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    f = log_path.open("w")
    proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)

    url = f"http://127.0.0.1:{port}"
    # wait for ready (poll /health)
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            f.close()
            raise RuntimeError(
                f"llama-server exited before becoming healthy "
                f"(exit={proc.returncode}); see {log_path}"
            )
        try:
            r = requests.get(f"{url}/health", timeout=2)
            if r.status_code == 200:
                break
        except requests.exceptions.RequestException:
            pass
        time.sleep(0.5)
    else:
        proc.send_signal(signal.SIGTERM)
        proc.wait(timeout=15)
        f.close()
        raise TimeoutError(f"llama-server failed to become healthy in 180s; see {log_path}")

    try:
        yield url, proc
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        f.close()


# ---------- request / parse -------------------------------------------------

def request_completion(url: str, prompt: str, n_predict: int, seed: int, timeout: int = 600):
    """POST /completion (non-streaming); return dict with tps, ttft_ms, n_drafted, n_accepted.

    Uses llama.cpp's native /completion. Server returns a `timings` dict with
    everything we need:
      - prompt_ms       => TTFT (prefill latency)
      - predicted_ms    => generation time
      - predicted_n     => generated tokens
      - draft_n         => total draft tokens proposed (= n_drafted)
      - draft_n_accepted=> total draft tokens accepted (= n_accept)
    """
    body = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": 0.0,
        "top_k": 1,
        "seed": seed,
        "stream": False,
        "cache_prompt": False,
    }
    r = requests.post(f"{url}/completion", json=body, timeout=timeout)
    r.raise_for_status()
    obj = r.json()
    timings = obj.get("timings", {}) or {}

    n_gen = int(timings.get("predicted_n", 0)) or 0
    predicted_ms = float(timings.get("predicted_ms", 0.0))
    tps = (n_gen * 1000.0 / predicted_ms) if predicted_ms > 0 else None
    ttft_ms = float(timings.get("prompt_ms", 0.0)) or None
    n_drafted = int(timings.get("draft_n", 0)) if "draft_n" in timings else None
    n_accept  = int(timings.get("draft_n_accepted", 0)) if "draft_n_accepted" in timings else None
    return {
        "tps": tps,
        "ttft_ms": ttft_ms,
        "n_gen": n_gen,
        "n_drafted": n_drafted,
        "n_accept": n_accept,
    }


def metrics_snapshot(url: str) -> dict:
    """Parse /metrics (Prometheus text format). Return relevant counters."""
    try:
        r = requests.get(f"{url}/metrics", timeout=5)
        if r.status_code != 200:
            return {}
    except requests.exceptions.RequestException:
        return {}
    out: dict[str, float] = {}
    for line in r.text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        m = re.match(r"(\S+)\s+(.+)", line)
        if not m:
            continue
        name, val = m.group(1), m.group(2)
        try:
            out[name] = float(val)
        except ValueError:
            pass
    return out


# ---------- subset selection ------------------------------------------------

def pick_subset(df: pd.DataFrame, per_category: int, seed: int) -> pd.DataFrame:
    parts = []
    for cat, g in df.groupby("category", sort=True):
        parts.append(g.sample(n=min(per_category, len(g)), random_state=seed))
    return pd.concat(parts, ignore_index=True)


def prompt_text(row: pd.Series) -> str:
    turns = row["turns"]
    if isinstance(turns, str):
        return turns
    if hasattr(turns, "tolist"):
        turns = turns.tolist()
    return turns[0] if turns else ""


# ---------- main bench loop -------------------------------------------------

def bench_pair(args, pair, quant, backend, prompts_df, results: list[Row]):
    pair_id, targets_by_quant, family = pair
    if quant not in targets_by_quant:
        print(f"  [skip] {pair_id} has no {quant} target defined", file=sys.stderr)
        return
    models_dir, target_name = targets_by_quant[quant]
    target = models_dir / target_name
    if not target.exists():
        print(f"  [skip] {target} not present", file=sys.stderr)
        return

    # Upstream build dirs overridable via env vars (matches dflash script).
    if backend == "cuda":
        build_dir = Path(os.environ.get("UPSTREAM_CUDA_BUILD",
                                        UPSTREAM_REPO / "build-cuda"))
    elif backend == "vulkan":
        build_dir = Path(os.environ.get("UPSTREAM_VULKAN_BUILD",
                                        UPSTREAM_REPO / "build-vulkan"))
    elif backend == "hip":
        build_dir = Path(os.environ.get("UPSTREAM_HIP_BUILD",
                                        UPSTREAM_REPO / "build-hip"))
    else:
        raise ValueError(f"unknown backend: {backend}")
    server_bin = build_dir / "bin" / "llama-server"

    if not server_bin.exists():
        print(f"  [skip] {server_bin} not built; backend={backend} unavailable", file=sys.stderr)
        return

    for k in args.k_values:
        log_path = args.outdir / f"server_{pair_id}_{backend}_k{k}.log"
        print(f"  starting server: k={k}  log={log_path}", flush=True)
        try:
            with llama_server(server_bin, target, backend, k, args.ctx, log_path) as (url, _proc):
                for i, row in prompts_df.iterrows():
                    ptxt = prompt_text(row)
                    try:
                        s = request_completion(url, ptxt, args.n_predict, args.seed,
                                               timeout=args.timeout)
                        ok = (s["tps"] is not None)
                    except Exception as e:
                        s = {"tps": None, "ttft_ms": None, "n_gen": 0,
                             "n_drafted": None, "n_accept": None}
                        ok = False
                        print(f"    request failed: {e}", file=sys.stderr)

                    n_drafted = s["n_drafted"]
                    n_accept = s["n_accept"]
                    accept_rate = (100.0 * n_accept / n_drafted) if n_drafted else None
                    al = (n_accept / (n_drafted / k)) if (n_drafted and k) else None
                    results.append(Row(
                        pair=pair_id, quant=quant, backend=backend, k=k, prompt_idx=int(i),
                        category=row.get("category", "?"),
                        n_prompt_tokens=0,
                        tps=s["tps"],
                        prefill_ms=s["ttft_ms"],
                        n_drafted=n_drafted, n_accept=n_accept,
                        accept_rate=accept_rate, al=al,
                        ok=ok,
                        raw_log=str(log_path) if not ok else "",
                    ))
                    tps_s = f"{s['tps']:7.2f}" if s["tps"] is not None else "    FAIL"
                    pre_s = f"{s['ttft_ms']:7.1f}" if s["ttft_ms"] is not None else "    NA"
                    acc_s = f"{n_accept}/{n_drafted}" if n_drafted else "NA"
                    print(f"    k={k:>3} cat={row['category']:<14} "
                          f"tps={tps_s} ttft={pre_s}ms accept={acc_s}", flush=True)
        except Exception as e:
            print(f"  server lifecycle failed: {e}", file=sys.stderr)


# ---------- output ----------------------------------------------------------

def write_outputs(args, results: list[Row]):
    args.outdir.mkdir(parents=True, exist_ok=True)
    jsonl = args.outdir / "bench.jsonl"
    with jsonl.open("w") as f:
        for r in results:
            f.write(json.dumps(dataclasses.asdict(r)) + "\n")
    print(f"\nwrote {jsonl}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pairs", nargs="+", default=[x[0] for x in DEFAULT_PAIRS])
    p.add_argument("--quants", nargs="+", default=["q6_k"],
                   choices=["q4_k_m", "q6_k", "q8_0"],
                   help="target quants to sweep (default: q6_k)")
    p.add_argument("--backends", nargs="+", default=["cuda", "vulkan"],
                   choices=["cuda", "vulkan", "hip"],
                   help="hip = AMD ROCm build. Override upstream build dir via "
                        "UPSTREAM_{CUDA,VULKAN,HIP}_BUILD env vars.")
    p.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K)
    p.add_argument("--per-category", type=int, default=1)
    p.add_argument("--n-predict", type=int, default=256)
    p.add_argument("--ctx", type=int, default=8192)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--timeout", type=int, default=900)
    p.add_argument("--outdir", type=Path,
                   default=REPO / "bench/speed-bench/results-mtp")
    p.add_argument("--speedbench", type=Path, default=SPEED_BENCH)
    args = p.parse_args()

    df = pd.read_parquet(args.speedbench)
    prompts_df = pick_subset(df, args.per_category, args.seed)
    print(f"selected {len(prompts_df)} prompts")

    pairs_by_id = {p[0]: p for p in DEFAULT_PAIRS}
    selected_pairs = [pairs_by_id[p] for p in args.pairs if p in pairs_by_id]

    results: list[Row] = []
    t0 = time.time()
    for pair in selected_pairs:
        for quant in args.quants:
            for backend in args.backends:
                print(f"\n=== {pair[2]} · {quant} · {backend} ===")
                bench_pair(args, pair, quant, backend, prompts_df, results)
    print(f"\ntotal wall time: {time.time() - t0:.1f}s")

    write_outputs(args, results)


if __name__ == "__main__":
    main()
