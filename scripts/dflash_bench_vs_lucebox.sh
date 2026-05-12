#!/usr/bin/env bash
# Run our fork's HumanEval / GSM8K / MATH-500 sweep with lucebox-aligned
# config (shuffle seed=42, first 10 prompts per dataset, n_gen=256). Output
# tag controls the result filename.
#
# Usage:
#   scripts/dflash_bench_vs_lucebox.sh <build-dir> <target-gguf> <draft-gguf> <tag> [configs]
#
# Example (apples-to-apples vs lucebox @ Qwen3.5-27B):
#   scripts/dflash_bench_vs_lucebox.sh build \
#     /path/to/Qwen3.5-27B-Q4_K_M.gguf /path/to/Qwen3.5-27B-DFlash.gguf \
#     qwen35-27b-vulkan greedy,chain
set -eu

BUILD_DIR="${1:?build-dir}"
TARGET="${2:?target-gguf}"
DRAFT="${3:?draft-gguf}"
TAG="${4:?tag}"
CONFIGS="${5:-all}"

cd "$(dirname "$0")/.."

for bench in humaneval gsm8k math500; do
    echo "=== $bench ($TAG) ==="
    python3 scripts/dflash_bench_humaneval.py \
        --benchmark "$bench" \
        --target "$TARGET" --draft "$DRAFT" \
        --build-dir "$BUILD_DIR" \
        --n-predict 256 --ctx 4096 --ngl 99 --timeout-s 180 \
        --shuffle-seed 42 --limit 10 \
        --configs "$CONFIGS" \
        --tag "$TAG"
done
