#!/usr/bin/env bash
# Session 25 sweep: Vulkan port of Phase 4 GDN-history. Mirror of
# scripts/dflash_gdn_history_sweep.sh, but pointed at the Vulkan build dir
# (build-vulkan-p4 by default) and without the CUDA_VISIBLE_DEVICES pin.
#
# For each prompt: runs both legacy chain and Phase 4 (--dflash-gdn-history)
# paths on Vulkan, parses n_accept / decoded t/s, prints a row.
# Bottom row reports per-config medians and the per-prompt speedup column.
#
# Usage:
#   scripts/dflash_gdn_history_sweep_vulkan.sh                         # all prompts
#   PROMPTS_DIR=/tmp/other_prompts ./scripts/dflash_gdn_history_sweep_vulkan.sh
#   BUILD_DIR=build-vulkan-other ./scripts/dflash_gdn_history_sweep_vulkan.sh
#
# Env vars:
#   BUILD_DIR              default build-vulkan-p4
#   N_PREDICT              default 64
#   N_RUNS                 default 1 (raise for stable medians; cost
#                                    scales linearly)
#
# Honest framing: we expect Phase 4 chain on Vulkan to land at ~75-90 t/s
# vs ~58-65 t/s legacy, beating Vulkan greedy (~71 t/s) but ~15% below
# lucebox CUDA chain (~107 t/s) — that's a Vulkan dispatch-overhead gap
# that's NOT a Phase-4-shaped problem.
set -eu

prompts_dir="${PROMPTS_DIR:-/tmp/dflash_prompts}"
n_predict="${N_PREDICT:-64}"
n_runs="${N_RUNS:-1}"
build_dir="${BUILD_DIR:-build-vulkan-p4}"

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
bin="$repo_root/$build_dir/bin/llama-speculative-simple"
target="${TARGET_MODEL:-/home/dev/devnid/dflash/models/Qwen3.5-27B-GGUF/Qwen3.5-27B-Q4_K_M.gguf}"
draft="${DRAFT_MODEL:-/home/dev/devnid/dflash/models/Qwen3.5-27B-DFlash.gguf}"

if [[ ! -x "$bin" ]]; then
    echo "ERROR: $bin not found or not executable" >&2
    echo "       (configure / build with: cmake -S . -B $build_dir -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_VULKAN=ON; cmake --build $build_dir -j)" >&2
    exit 1
fi

run_once () {
    local prompt_file="$1"
    local extra_args="$2"
    "$bin" \
        -m "$target" \
        -md "$draft" \
        --draft-type dflash \
        $extra_args \
        -p "$(cat "$prompt_file")" \
        --n-predict "$n_predict" --temp 0 --seed 0 -c 8192 \
        -ngl 99 -ngld 99 --dflash-no-chat-template 2>&1 \
        | awk '
            /^decoded /     { split($0, a, " "); for (i=1; i<=length(a); i++) if (a[i] == "speed:") { speed = a[i+1] } }
            /^n_accept /    { split($0, a, " "); for (i=1; i<=length(a); i++) if (a[i] == "=") { accept = a[i+1] } }
            /^n_drafted /   { split($0, a, " "); for (i=1; i<=length(a); i++) if (a[i] == "=") { drafted = a[i+1] } }
            END { printf "%s %s %s\n", speed, accept, drafted }'
}

median_of_runs () {
    awk '
    {
        speeds[NR] = $1; accept = $2; drafted = $3;
    }
    END {
        n = NR;
        for (i = 1; i <= n; i++) for (j = i+1; j <= n; j++) if (speeds[i] > speeds[j]) { t = speeds[i]; speeds[i] = speeds[j]; speeds[j] = t }
        m = (n % 2 == 1) ? speeds[(n+1)/2] : (speeds[n/2] + speeds[n/2 + 1]) / 2;
        printf "%.2f %s %s\n", m, accept, drafted;
    }'
}

printf "\n=== Vulkan DFlash GDN-history sweep: %s prompts × %s runs each (build: %s) ===\n\n" \
    "$(ls "$prompts_dir"/p*.txt | wc -l)" "$n_runs" "$build_dir"
printf "%-7s | %-22s | %-22s | %-8s\n" "prompt" "legacy (t/s, acc, drafted)" "phase4 (t/s, acc, drafted)" "speedup"
printf "%s\n" "---------|------------------------|------------------------|---------"

legacy_speeds=()
phase4_speeds=()

for prompt_file in "$prompts_dir"/p*.txt; do
    pid="$(basename "$prompt_file" .txt)"

    leg_tmp=""
    for r in $(seq 1 "$n_runs"); do leg_tmp+="$(run_once "$prompt_file" "")"$'\n'; done
    read -r leg_speed leg_acc leg_drafted < <(printf "%s" "$leg_tmp" | median_of_runs)

    p4_tmp=""
    for r in $(seq 1 "$n_runs"); do p4_tmp+="$(run_once "$prompt_file" "--dflash-gdn-history")"$'\n'; done
    read -r p4_speed p4_acc p4_drafted < <(printf "%s" "$p4_tmp" | median_of_runs)

    if [[ -n "$leg_speed" && -n "$p4_speed" && "$leg_speed" != "0" ]]; then
        speedup="$(awk -v a="$p4_speed" -v b="$leg_speed" 'BEGIN { printf "%+.1f%%", (a/b - 1)*100 }')"
    else
        speedup="n/a"
    fi

    printf "%-7s | %7s t/s %3s/%-9s | %7s t/s %3s/%-9s | %s\n" \
        "$pid" "$leg_speed" "$leg_acc" "$leg_drafted" "$p4_speed" "$p4_acc" "$p4_drafted" "$speedup"

    legacy_speeds+=("$leg_speed")
    phase4_speeds+=("$p4_speed")
done

printf "\nSummary (median of per-prompt medians):\n"
printf "  vulkan legacy:  %s t/s\n" "$(printf "%s\n" "${legacy_speeds[@]}" | sort -n | awk 'BEGIN{n=0} {a[++n]=$1} END{print (n%2==1)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2}')"
printf "  vulkan phase4:  %s t/s\n" "$(printf "%s\n" "${phase4_speeds[@]}" | sort -n | awk 'BEGIN{n=0} {a[++n]=$1} END{print (n%2==1)?a[(n+1)/2]:(a[n/2]+a[n/2+1])/2}')"
