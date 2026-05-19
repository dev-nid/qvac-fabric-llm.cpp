#!/usr/bin/env bash
# Provision a fresh machine for the SPEED-Bench DFlash vs MTP run:
#   1. Install Python deps (pandas, requests, tabulate, matplotlib, gguf, hf-hub)
#   2. Download SPEED-Bench qualitative split parquet (~3 GB)
#   3. Download Unsloth target GGUFs at the requested quant
#   4. Download Unsloth MTP-GGUFs (Qwen 3.6 only) — for MTP comparison
#   5. Download + convert z-lab DFlash drafts (HF safetensors -> GGUF)
#   6. Clone + build upstream ggml-org/llama.cpp master (for MTP llama-server)
#
# Notes:
#   - HF_TOKEN env var required if pulling z-lab/Qwen3.6-27B-DFlash (gated repo)
#   - Quants: q4_k_m | q6_k | q8_0; default q6_k. Pass --quants to vary.
#   - Backend: cuda | vulkan | hip; default auto. Pass --backend to force.
#   - Skips items already on disk.
#   - Disk needs: ~100 GB per quant for Qwen 3.6 + Gemma 4 dense + MoE
#
# Usage:
#   ./setup.sh                       # q6_k, auto-backend
#   ./setup.sh --quants q4_k_m q6_k q8_0   # full sweep deps
#   ./setup.sh --backend hip         # Strix Halo
#   ./setup.sh --no-mtp              # skip MTP-side prep entirely
#   ./setup.sh --no-upstream         # skip cloning + building upstream
#   ./setup.sh --check               # only verify what's present, don't pull

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_DEFAULT=$(cd "$SCRIPT_DIR/../.." && pwd)
BENCH_REPO=${BENCH_REPO:-$REPO_DEFAULT}
BENCH_DATA_DIR=${BENCH_DATA_DIR:-$BENCH_REPO/../bench-data}
UPSTREAM_LLAMA_CPP=${UPSTREAM_LLAMA_CPP:-/tmp/llama-cpp-master}

QUANTS=("q6_k")
BACKEND=auto
DO_MTP=1
DO_UPSTREAM=1
CHECK_ONLY=0

while [ $# -gt 0 ]; do
    case "$1" in
        --quants) shift; QUANTS=(); while [ $# -gt 0 ] && [[ "$1" != --* ]]; do QUANTS+=("$1"); shift; done ;;
        --backend) shift; BACKEND="$1"; shift ;;
        --no-mtp) DO_MTP=0; shift ;;
        --no-upstream) DO_UPSTREAM=0; shift ;;
        --check) CHECK_ONLY=1; shift ;;
        -h|--help) sed -n 's/^# \?//; 2,/^$/p' "$0" | head -25; exit 0 ;;
        *) echo "unknown arg: $1"; exit 1 ;;
    esac
done

say() { echo "[setup] $*"; }

mkdir -p "$BENCH_DATA_DIR/models" "$BENCH_DATA_DIR/speed-bench"

# ---- backend auto-detect --------------------------------------------------
detect_backend() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q GPU; then
        echo cuda
    elif command -v rocm-smi >/dev/null 2>&1; then
        echo hip
    elif command -v vulkaninfo >/dev/null 2>&1 && vulkaninfo --summary 2>/dev/null | grep -q deviceName; then
        echo vulkan
    else
        echo unknown
    fi
}
if [ "$BACKEND" = auto ]; then BACKEND=$(detect_backend); fi
say "backend: $BACKEND"
say "BENCH_REPO=$BENCH_REPO"
say "BENCH_DATA_DIR=$BENCH_DATA_DIR"
say "UPSTREAM_LLAMA_CPP=$UPSTREAM_LLAMA_CPP"
say "quants: ${QUANTS[*]}"

# ---- 1. Python deps -------------------------------------------------------
if [ "$CHECK_ONLY" = 0 ]; then
    say "=== python deps ==="
    python3 -m pip install --user --break-system-packages --quiet \
        pandas requests tabulate matplotlib "huggingface_hub>=1.14" gguf 2>&1 | tail -3 || true
fi

# ---- 2. SPEED-Bench dataset ----------------------------------------------
PARQUET="$BENCH_DATA_DIR/speed-bench/qualitative/test-00000-of-00001.parquet"
if [ ! -f "$PARQUET" ]; then
    say "=== fetch SPEED-Bench qualitative split ==="
    [ "$CHECK_ONLY" = 1 ] && say "  [check] missing: $PARQUET" || {
        hf download nvidia/SPEED-Bench qualitative/test-00000-of-00001.parquet \
            --repo-type dataset --local-dir "$BENCH_DATA_DIR/speed-bench"
    }
else
    say "  SPEED-Bench parquet present"
fi

# ---- 3. Target GGUFs (Unsloth) -------------------------------------------
pull_target() {
    local quant="$1" repo="$2" filename="$3" subdir="$4"
    local outdir="$BENCH_DATA_DIR/models/$subdir"
    local outfile="$outdir/$filename"
    mkdir -p "$outdir"
    if [ -f "$outfile" ]; then say "  [present] $filename"; return; fi
    [ "$CHECK_ONLY" = 1 ] && { say "  [check] missing: $outfile"; return; }
    say "  download $repo / $filename"
    hf download "$repo" "$filename" --local-dir "$outdir" 2>&1 | tail -2
}

quant_to_unsloth_qtag() {
    case "$1" in
        q4_k_m) echo Q4_K_M ;;
        q6_k)   echo Q6_K ;;
        q8_0)   echo Q8_0 ;;
        *) echo "?"; exit 1 ;;
    esac
}

for q in "${QUANTS[@]}"; do
    QTAG=$(quant_to_unsloth_qtag "$q")
    QDIR="bench-${q//_k_m/4}"; QDIR="${QDIR/_k/6}"; QDIR="${QDIR/_0/8}"
    say "=== target GGUFs · $q ($QTAG) → models/$QDIR/ ==="
    pull_target "$q" unsloth/Qwen3.6-27B-GGUF        "Qwen3.6-27B-$QTAG.gguf"          "$QDIR"
    pull_target "$q" unsloth/Qwen3.6-35B-A3B-GGUF    "Qwen3.6-35B-A3B-UD-$QTAG.gguf"   "$QDIR" || \
    pull_target "$q" unsloth/Qwen3.6-35B-A3B-GGUF    "Qwen3.6-35B-A3B-$QTAG.gguf"      "$QDIR"
    pull_target "$q" unsloth/gemma-4-31B-it-GGUF     "gemma-4-31B-it-$QTAG.gguf"       "$QDIR"
    pull_target "$q" unsloth/gemma-4-26B-A4B-it-GGUF "gemma-4-26B-A4B-it-UD-$QTAG.gguf" "$QDIR" || \
    pull_target "$q" unsloth/gemma-4-26B-A4B-it-GGUF "gemma-4-26B-A4B-it-$QTAG.gguf"    "$QDIR"
done

# ---- 4. MTP-GGUFs (Unsloth, Qwen 3.6 only) -------------------------------
if [ "$DO_MTP" = 1 ]; then
    for q in "${QUANTS[@]}"; do
        QTAG=$(quant_to_unsloth_qtag "$q")
        MTPDIR="bench-${q//_k_m/4}-mtp"; MTPDIR="${MTPDIR/_k/6}"; MTPDIR="${MTPDIR/_0/8}"
        say "=== MTP-GGUFs · $q → models/$MTPDIR/ ==="
        pull_target "$q" unsloth/Qwen3.6-27B-MTP-GGUF      "Qwen3.6-27B-$QTAG.gguf"         "$MTPDIR"
        pull_target "$q" unsloth/Qwen3.6-35B-A3B-MTP-GGUF  "Qwen3.6-35B-A3B-UD-$QTAG.gguf"  "$MTPDIR" || \
        pull_target "$q" unsloth/Qwen3.6-35B-A3B-MTP-GGUF  "Qwen3.6-35B-A3B-$QTAG.gguf"     "$MTPDIR"
    done
fi

# ---- 5. DFlash drafts (z-lab → GGUF) -------------------------------------
say "=== DFlash drafts (z-lab safetensors → GGUF via convert_hf_to_gguf.py) ==="
if [ "$CHECK_ONLY" = 0 ]; then
    DRAFTS_OUT="$BENCH_DATA_DIR/models"
    HF_DRAFTS="$BENCH_DATA_DIR/hf-drafts"
    mkdir -p "$HF_DRAFTS"
    # (repo, target_tok_repo, output_gguf_name)
    for spec in \
        "z-lab/Qwen3.6-27B-DFlash|Qwen/Qwen3.6-27B|Qwen3.6-27B-DFlash.gguf" \
        "z-lab/Qwen3.6-35B-A3B-DFlash|Qwen/Qwen3.6-35B-A3B|Qwen3.6-35B-A3B-DFlash.gguf" \
        "z-lab/gemma-4-31B-it-DFlash|google/gemma-4-31B-it|gemma-4-31B-it-DFlash.gguf" \
        "z-lab/gemma-4-26B-A4B-it-DFlash|google/gemma-4-26B-A4B-it|gemma-4-26B-A4B-it-DFlash.gguf"; do
        IFS='|' read DREPO TREPO OUTNAME <<< "$spec"
        OUT="$DRAFTS_OUT/$OUTNAME"
        if [ -f "$OUT" ]; then say "  [present] $OUTNAME"; continue; fi
        if [ "$CHECK_ONLY" = 1 ]; then say "  [check] missing: $OUT"; continue; fi
        # 1. fetch draft safetensors (gated repos need HF_TOKEN)
        STAGE_SRC="$HF_DRAFTS/$(basename $DREPO)"
        hf download "$DREPO" --local-dir "$STAGE_SRC" 2>&1 | tail -2 || {
            say "  ERROR: failed to fetch $DREPO (gated? need HF_TOKEN?); skipping"
            continue
        }
        # 2. fetch matching tokenizer
        TOKDIR="$BENCH_DATA_DIR/tok/$(basename $TREPO)"
        hf download "$TREPO" --include "tokenizer*" --include "config.json" \
            --local-dir "$TOKDIR" 2>&1 | tail -2
        # 3. stage + convert
        STAGE="$BENCH_DATA_DIR/stage-$(basename $DREPO)"
        rm -rf "$STAGE" && mkdir -p "$STAGE"
        cp -L "$STAGE_SRC/config.json" "$STAGE/"
        cp -L "$STAGE_SRC/model.safetensors" "$STAGE/"
        ln -sf "$TOKDIR/tokenizer.json"        "$STAGE/tokenizer.json"
        ln -sf "$TOKDIR/tokenizer_config.json" "$STAGE/tokenizer_config.json"
        python3 "$BENCH_REPO/convert_hf_to_gguf.py" "$STAGE" --outtype bf16 \
            --outfile "$OUT" 2>&1 | tail -3
    done
fi

# ---- 6. Upstream llama.cpp (for MTP) -------------------------------------
if [ "$DO_UPSTREAM" = 1 ] && [ "$DO_MTP" = 1 ]; then
    say "=== upstream ggml-org/llama.cpp ($UPSTREAM_LLAMA_CPP) ==="
    if [ ! -d "$UPSTREAM_LLAMA_CPP/.git" ]; then
        [ "$CHECK_ONLY" = 1 ] && say "  [check] missing: $UPSTREAM_LLAMA_CPP" || \
            git clone --depth 1 https://github.com/ggml-org/llama.cpp.git "$UPSTREAM_LLAMA_CPP"
    fi
    if [ "$CHECK_ONLY" = 0 ]; then
        case "$BACKEND" in
            cuda)
                say "  build-cuda/"
                cmake -B "$UPSTREAM_LLAMA_CPP/build-cuda" -S "$UPSTREAM_LLAMA_CPP" \
                    -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
                    -DLLAMA_CURL=ON 2>&1 | tail -3
                cmake --build "$UPSTREAM_LLAMA_CPP/build-cuda" --target llama-server -j 2>&1 | tail -3
                ;;
            vulkan)
                say "  build-vulkan/"
                cmake -B "$UPSTREAM_LLAMA_CPP/build-vulkan" -S "$UPSTREAM_LLAMA_CPP" \
                    -DGGML_VULKAN=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
                    -DLLAMA_CURL=ON 2>&1 | tail -3
                cmake --build "$UPSTREAM_LLAMA_CPP/build-vulkan" --target llama-server -j 2>&1 | tail -3
                ;;
            hip)
                say "  build-hip/  (Strix Halo etc.; needs ROCm)"
                cmake -B "$UPSTREAM_LLAMA_CPP/build-hip" -S "$UPSTREAM_LLAMA_CPP" \
                    -DGGML_HIP=ON -DGGML_NATIVE=OFF -DCMAKE_BUILD_TYPE=Release \
                    -DLLAMA_CURL=ON 2>&1 | tail -3
                cmake --build "$UPSTREAM_LLAMA_CPP/build-hip" --target llama-server -j 2>&1 | tail -3
                ;;
            *)
                say "  WARNING: unknown backend, skipping upstream build"
                ;;
        esac
    fi
fi

say "=== setup done ==="
say "next:  ./run_all.sh    (or pass --quants / --backends / etc.)"
