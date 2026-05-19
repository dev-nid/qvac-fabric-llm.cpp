#!/usr/bin/env bash
# Master orchestrator for the SPEED-Bench DFlash vs MTP comparison.
#
# Runs DFlash sweep + MTP sweep, merges JSONLs, regenerates plots.
#
# Paths are all env-overridable (see README.md). Backends auto-detected:
# only backends with a built llama-server / llama-speculative-simple are
# run; others are skipped with a warning.
#
# Usage examples:
#   ./run_all.sh                              # smoke: q6_k only, all available backends
#   ./run_all.sh --quants q4_k_m q6_k q8_0    # full quant sweep
#   ./run_all.sh --backends hip               # AMD ROCm only (e.g. Strix Halo)
#   ./run_all.sh --dflash-only                # skip MTP side
#   ./run_all.sh --mtp-only                   # skip DFlash side
#   ./run_all.sh --per-category 4             # 44 prompts instead of 11
#   ./run_all.sh --n-predict 4096             # Jarvis-shape generation length
#
# Required env (or auto-derived from script location):
#   BENCH_REPO           path to llama.cpp fork (default: derive from script)
#   BENCH_DATA_DIR       where models/, speed-bench/ live (default: $BENCH_REPO/../bench-data)
#   UPSTREAM_LLAMA_CPP   upstream master checkout for MTP server (default: /tmp/llama-cpp-master)
#   BENCH_GPU_INDEX      device index to pin (default: 0)
# Plus per-backend overrides documented in README.md.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DEFAULT=$(cd "$SCRIPT_DIR/../.." && pwd)
BENCH_REPO=${BENCH_REPO:-$REPO_DEFAULT}
BENCH_DATA_DIR=${BENCH_DATA_DIR:-$BENCH_REPO/../bench-data}
UPSTREAM_LLAMA_CPP=${UPSTREAM_LLAMA_CPP:-/tmp/llama-cpp-master}
RESULTS=$SCRIPT_DIR/results

export BENCH_REPO BENCH_DATA_DIR UPSTREAM_LLAMA_CPP

# Defaults
QUANTS=("q6_k")
BACKENDS=()  # auto-detect if empty
K_VALUES=(2 4 8 16)
PER_CATEGORY=1
N_PREDICT=256
RUN_DFLASH=1
RUN_MTP=1
DFLASH_PAIRS=("qwen36-27b" "qwen36-35bA3" "gemma4-31b" "gemma4-26bA4")
MTP_PAIRS=("qwen36-27b-mtp" "qwen36-35bA3-mtp")

usage() { sed -n 's/^# \?//; 2,/^$/p' "$0" | head -25; exit 1; }

while [ $# -gt 0 ]; do
    case "$1" in
        --quants) shift; QUANTS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do QUANTS+=("$1"); shift; done ;;
        --backends) shift; BACKENDS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do BACKENDS+=("$1"); shift; done ;;
        --k-values) shift; K_VALUES=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do K_VALUES+=("$1"); shift; done ;;
        --pairs) shift; DFLASH_PAIRS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do DFLASH_PAIRS+=("$1"); shift; done ;;
        --per-category) shift; PER_CATEGORY="$1"; shift ;;
        --n-predict) shift; N_PREDICT="$1"; shift ;;
        --dflash-only) RUN_MTP=0; shift ;;
        --mtp-only) RUN_DFLASH=0; shift ;;
        -h|--help) usage ;;
        *) echo "unknown arg: $1"; usage ;;
    esac
done

# Backend auto-detection: enable a backend only if the corresponding
# build dir contains the binary we need. User can override with --backends.
if [ ${#BACKENDS[@]} -eq 0 ]; then
    [ -x "$BENCH_REPO/build-cuda/bin/llama-speculative-simple" ] && BACKENDS+=("cuda")
    [ -x "$BENCH_REPO/build/bin/llama-speculative-simple"      ] && BACKENDS+=("vulkan")
    [ -x "$BENCH_REPO/build-hip/bin/llama-speculative-simple"  ] && BACKENDS+=("hip")
    if [ ${#BACKENDS[@]} -eq 0 ]; then
        echo "error: no backends detected; build llama-speculative-simple in build/, build-cuda/, or build-hip/" >&2
        exit 1
    fi
fi

mkdir -p "$RESULTS"
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$RESULTS/run_all.log"; }

log "=== config ==="
log "BENCH_REPO:         $BENCH_REPO"
log "BENCH_DATA_DIR:     $BENCH_DATA_DIR"
log "UPSTREAM_LLAMA_CPP: $UPSTREAM_LLAMA_CPP"
log "quants:             ${QUANTS[*]}"
log "backends:           ${BACKENDS[*]}"
log "k_values:           ${K_VALUES[*]}"
log "per_category:       $PER_CATEGORY  (total prompts: $((PER_CATEGORY * 11)))"
log "n_predict:          $N_PREDICT"
log "dflash:             $RUN_DFLASH  pairs: ${DFLASH_PAIRS[*]}"
log "mtp:                $RUN_MTP  pairs: ${MTP_PAIRS[*]}"

T0=$(date +%s)

# ---- 1. DFlash sweep ------------------------------------------------------
if [ "$RUN_DFLASH" = 1 ]; then
    log ""
    log "=== DFlash sweep ==="
    OUT_DFLASH=$RESULTS/dflash-run-$(date -u +%Y%m%dT%H%M%S)
    python3 "$SCRIPT_DIR/run_dflash_k_sweep.py" \
        --pairs "${DFLASH_PAIRS[@]}" \
        --quants "${QUANTS[@]}" \
        --backends "${BACKENDS[@]}" \
        --k-values "${K_VALUES[@]}" \
        --per-category "$PER_CATEGORY" \
        --n-predict "$N_PREDICT" \
        --outdir "$OUT_DFLASH" 2>&1 | tee -a "$RESULTS/run_all.log"
    [ -f "$OUT_DFLASH/bench.jsonl" ] && cp "$OUT_DFLASH/bench.jsonl" "$RESULTS/dflash-quant-sweep.jsonl"
fi

# ---- 2. MTP sweep (upstream) ----------------------------------------------
if [ "$RUN_MTP" = 1 ]; then
    log ""
    log "=== MTP sweep (upstream llama.cpp at $UPSTREAM_LLAMA_CPP) ==="
    OUT_MTP=$RESULTS/mtp-run-$(date -u +%Y%m%dT%H%M%S)
    python3 "$SCRIPT_DIR/run_mtp_k_sweep.py" \
        --pairs "${MTP_PAIRS[@]}" \
        --quants "${QUANTS[@]}" \
        --backends "${BACKENDS[@]}" \
        --k-values "${K_VALUES[@]}" \
        --per-category "$PER_CATEGORY" \
        --n-predict "$N_PREDICT" \
        --outdir "$OUT_MTP" 2>&1 | tee -a "$RESULTS/run_all.log"
    [ -f "$OUT_MTP/bench.jsonl" ] && cp "$OUT_MTP/bench.jsonl" "$RESULTS/mtp-quant-sweep.jsonl"
fi

# ---- 3. Merge JSONLs -----------------------------------------------------
log ""
log "=== merge to all-bench.jsonl ==="
: > "$RESULTS/all-bench.jsonl.new"
for f in "$RESULTS/dflash-bench.jsonl" "$RESULTS/dflash-quant-sweep.jsonl" \
         "$RESULTS/mtp-bench.jsonl" "$RESULTS/mtp-quant-sweep.jsonl"; do
    [ -f "$f" ] && cat "$f" >> "$RESULTS/all-bench.jsonl.new"
done
mv "$RESULTS/all-bench.jsonl.new" "$RESULTS/all-bench.jsonl"
log "  merged $(wc -l < "$RESULTS/all-bench.jsonl") rows"

# ---- 4. Plots ------------------------------------------------------------
log ""
log "=== plots ==="
python3 "$SCRIPT_DIR/plot_bench.py" \
    --in "$RESULTS/all-bench.jsonl" \
    --out "$RESULTS" 2>&1 | tee -a "$RESULTS/run_all.log"

T1=$(date +%s)
log ""
log "=== done in $((T1 - T0))s ==="
log "results: $RESULTS/{all-bench.jsonl,tps_vs_k.png,accept_vs_k.png,ttft_vs_k.png}"
log "report:  $SCRIPT_DIR/REPORT.md, $SCRIPT_DIR/REPORT-blogstyle.md"
