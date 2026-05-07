#!/usr/bin/env bash
#
# scripts/dflash_perf_record.sh
#
# Linux `perf` flamegraph wrapper for one DFlash spec-decode invocation.
# Companion of the optimization plan in
# logs/core_architecture/10_optimization_plan.md (Phase 1 — log source #3).
#
# Records `perf` stack samples for one chosen (config, prompt) combo, then
# generates a flamegraph SVG (auto-fetches Brendan Gregg's flamegraph.pl on
# first use, into bench/tools/FlameGraph/). Falls back to `perf report
# --stdio` when flamegraph.pl can't be fetched (e.g. no internet).
#
# Usage:
#     DFLASH_TEST_TARGET=... DFLASH_TEST_DRAFT=... \
#         ./scripts/dflash_perf_record.sh \
#             [--bin-dir DIR]      # default: build-cpu/bin
#             [--config NAME]      # default: tree-budget-18
#             [--prompt NAME]      # default: math (one of math|nl-low-entropy|nl-high-entropy|code|conv)
#             [--n N]              # n_predict, default: 64
#             [--threads T]        # default: 8
#             [--freq HZ]          # perf sample freq, default: 999
#             [--out-dir DIR]      # default: bench/cpu/perf
#             [--no-svg]           # skip flamegraph generation, leave perf.data only
#
# Prerequisites:
#   - linux `perf` (apt: linux-tools-generic) — required.
#   - Internet access to fetch flamegraph.pl on first run (or place it manually
#     at bench/tools/FlameGraph/flamegraph.pl).
#
# Output (in $out-dir):
#   perf-<config>-<prompt>-<git_sha>.data        raw perf data
#   perf-<config>-<prompt>-<git_sha>.folded      collapsed stacks (text)
#   perf-<config>-<prompt>-<git_sha>.svg         flamegraph (interactive in browser)
#   perf-<config>-<prompt>-<git_sha>.report.txt  text top-N (always emitted)
#
# Exit codes:
#   0  perf ran, output written (svg generation may have been skipped if --no-svg)
#   2  setup failure (missing perf, missing models, missing binaries)

set -u
set -o pipefail

# --------------------------------------------------------------------------
# Defaults
# --------------------------------------------------------------------------

bin_dir="build-cpu/bin"
config="tree-budget-18"
prompt_name="math"
n_predict=64
threads=8
freq=999
out_dir="bench/cpu/perf"
make_svg=1

while [ $# -gt 0 ]; do
    case "$1" in
        --bin-dir) bin_dir="$2"; shift 2 ;;
        --config)  config="$2"; shift 2 ;;
        --prompt)  prompt_name="$2"; shift 2 ;;
        --n)       n_predict="$2"; shift 2 ;;
        --threads) threads="$2"; shift 2 ;;
        --freq)    freq="$2"; shift 2 ;;
        --out-dir) out_dir="$2"; shift 2 ;;
        --no-svg)  make_svg=0; shift ;;
        -h|--help) sed -n '3,40p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) printf 'unknown argument: %s\n' "$1" >&2; exit 2 ;;
    esac
done

# --------------------------------------------------------------------------
# Prereqs
# --------------------------------------------------------------------------

if ! command -v perf >/dev/null 2>&1; then
    printf '\033[31m[error]\033[0m `perf` not found (install linux-tools-generic on Ubuntu)\n' >&2
    exit 2
fi

target_model="${DFLASH_TEST_TARGET:-}"
draft_model="${DFLASH_TEST_DRAFT:-}"
if [ -z "$target_model" ] || [ ! -f "$target_model" ] \
   || [ -z "$draft_model" ] || [ ! -f "$draft_model" ]; then
    printf '\033[33m[skip]\033[0m DFLASH_TEST_TARGET/DRAFT unset or missing\n' >&2
    exit 0
fi

llama_spec="$bin_dir/llama-speculative-simple"
llama_cli="$bin_dir/llama-cli"
if [ ! -x "$llama_spec" ]; then
    printf '\033[31m[error]\033[0m %s not found\n' "$llama_spec" >&2
    exit 2
fi

# --------------------------------------------------------------------------
# Per-config args
# --------------------------------------------------------------------------

is_greedy=0
extra_args=""
case "$config" in
    greedy)           is_greedy=1 ;;
    chain)            extra_args="" ;;
    chain-topk-2)     extra_args="--dflash-topk 2" ;;
    tree-medusa-k2)   extra_args="--dflash-tree" ;;
    tree-budget-18)   extra_args="--dflash-tree-budget 18" ;;
    tree-budget-22)   extra_args="--dflash-tree-budget 22" ;;
    *)
        printf '\033[31m[error]\033[0m unknown config: %s\n' "$config" >&2
        exit 2
        ;;
esac

case "$prompt_name" in
    math)            prompt='Solve: 47 * 83 = ' ;;
    nl-low-entropy)  prompt='The capital of France is' ;;
    nl-high-entropy) prompt='Once upon a time, in a land far away,' ;;
    code)            prompt=$'Write a Python function that returns the Fibonacci sequence:\n' ;;
    conv)            prompt=$'User: How do I learn programming?\nAssistant:' ;;
    *)
        printf '\033[31m[error]\033[0m unknown prompt: %s\n' "$prompt_name" >&2
        exit 2
        ;;
esac

# --------------------------------------------------------------------------
# Output paths
# --------------------------------------------------------------------------

mkdir -p "$out_dir"
git_sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
stem="$out_dir/perf-${config}-${prompt_name}-${git_sha}"
data_file="${stem}.data"
folded_file="${stem}.folded"
svg_file="${stem}.svg"
report_file="${stem}.report.txt"

# --------------------------------------------------------------------------
# Try to make flamegraph.pl available (auto-fetch from Brendan Gregg's repo
# into a shared cache dir; idempotent across runs).
# --------------------------------------------------------------------------

flamegraph_dir="bench/tools/FlameGraph"
flamegraph_pl="$flamegraph_dir/flamegraph.pl"
stackcollapse_pl="$flamegraph_dir/stackcollapse-perf.pl"

if [ "$make_svg" = "1" ] && [ ! -x "$flamegraph_pl" ]; then
    if command -v git >/dev/null 2>&1 && [ -z "${DFLASH_NO_NETWORK:-}" ]; then
        printf '[setup] fetching FlameGraph -> %s ...\n' "$flamegraph_dir"
        mkdir -p bench/tools
        git clone --depth=1 https://github.com/brendangregg/FlameGraph.git \
            "$flamegraph_dir" 2>/dev/null \
            && chmod +x "$flamegraph_pl" "$stackcollapse_pl" 2>/dev/null \
            || printf '\033[33m[warn]\033[0m flamegraph.pl fetch failed (will use perf report --stdio)\n' >&2
    fi
fi

# --------------------------------------------------------------------------
# Run perf record
# --------------------------------------------------------------------------

printf '\033[1mperf record\033[0m\n'
printf '  config        : %s\n' "$config"
printf '  prompt        : %s\n' "$prompt_name"
printf '  n_predict     : %d\n' "$n_predict"
printf '  threads       : %d\n' "$threads"
printf '  freq          : %d Hz\n' "$freq"
printf '  out           : %s.{data,folded,svg,report.txt}\n' "$stem"
printf '\n'

# --call-graph=dwarf gives the most accurate stack unwinding for symbolicated C++.
# May be slow for very long runs; stay reasonable on the n_predict.
if [ "$is_greedy" = "1" ]; then
    perf record \
        -F "$freq" \
        -g --call-graph=dwarf \
        -o "$data_file" \
        -- \
        "$llama_cli" \
            -m "$target_model" \
            -p "$prompt" --n-predict "$n_predict" --temp 0 --seed 0 \
            -no-cnv --no-display-prompt -ngl 0 -ub 1 -t "$threads"
else
    # shellcheck disable=SC2086 # intentional word-split of $extra_args
    perf record \
        -F "$freq" \
        -g --call-graph=dwarf \
        -o "$data_file" \
        -- \
        "$llama_spec" \
            -m  "$target_model" \
            -md "$draft_model" \
            --draft-type dflash \
            $extra_args \
            -p "$prompt" --n-predict "$n_predict" --temp 0 --seed 0 \
            -ngl 0 -ngld 0 -ub 1 -t "$threads"
fi

printf '\n'
printf '[done] perf data: %s\n' "$data_file"

# --------------------------------------------------------------------------
# Always: emit a text report (top symbols by self time)
# --------------------------------------------------------------------------

perf report -i "$data_file" --stdio --sort overhead,symbol \
    --no-children -n -g none --percent-limit 0.5 \
    > "$report_file" 2>/dev/null \
    && printf '[done] text report: %s\n' "$report_file"

# --------------------------------------------------------------------------
# Optional: flamegraph SVG
# --------------------------------------------------------------------------

if [ "$make_svg" = "0" ]; then
    printf '[skip] svg generation (--no-svg)\n'
    exit 0
fi

if [ ! -x "$flamegraph_pl" ] || [ ! -x "$stackcollapse_pl" ]; then
    printf '\033[33m[warn]\033[0m flamegraph.pl unavailable; skipping svg.\n' >&2
    printf '       To generate manually later:\n' >&2
    printf '         perf script -i %s | %s | %s > %s\n' \
        "$data_file" "$stackcollapse_pl" "$flamegraph_pl" "$svg_file" >&2
    exit 0
fi

perf script -i "$data_file" 2>/dev/null \
    | "$stackcollapse_pl" \
    > "$folded_file"

"$flamegraph_pl" \
    --title "DFlash $config / $prompt_name (n=$n_predict)" \
    --subtitle "git $git_sha — perf -F $freq" \
    --width 1600 --colors java \
    "$folded_file" > "$svg_file"

printf '[done] flamegraph: %s\n' "$svg_file"
printf '       open in a browser: file://%s/%s\n' "$(pwd)" "$svg_file"
