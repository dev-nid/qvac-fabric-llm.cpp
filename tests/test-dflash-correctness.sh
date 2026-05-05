#!/usr/bin/env bash
#
# tests/test-dflash-correctness.sh
#
# Strong-correctness CTest harness for DFlash speculative decoding.
#
# Verifies the DFlash speculative-decoding contract:
#
#     greedy_decode(target_model, prompt, N) == \
#     dflash_speculative_decode(target_model, dflash_draft, prompt, N, temp=0)
#
# byte-for-byte across a grid of (prompt, n_predict) pairs. Mirrors the
# 18-prompt sweep from logs/core_architecture/05_correctness_test.md and
# logs/core_architecture/08_session_resume_prompt.md "Quick verification
# commands" section. The two binaries under test are `llama-cli` (greedy
# reference) and `llama-speculative-simple --draft-type dflash` (DFlash
# speculative).
#
# This is a SUBPROCESS test, not a libllama-API test, on purpose: the gate
# is "do the two user-facing binaries produce byte-equal output?", not "do
# specific libllama functions return specific values". Mirrors the
# `test-tokenizers-repo.sh` precedent in this directory (a shell script
# wired into CTest via `llama_test_cmd`).
#
# Skip-on-missing-models: exits SUCCESS with a yellow warning if either
# DFLASH_TEST_TARGET or DFLASH_TEST_DRAFT env vars are unset or missing.
# Mirrors tests/get-model.cpp's `get_model_or_exit` convention. Tests that
# require non-standard models stay green in CI as long as the model isn't
# installed; the developer setting the env var opts into running the test.
#
# Usage (CTest):
#
#     DFLASH_TEST_TARGET=/path/to/Qwen3-4B-Q8_0.gguf \
#     DFLASH_TEST_DRAFT=/path/to/Qwen3-4B-DFlash-b16.gguf \
#     ctest -R test-dflash-correctness --output-on-failure
#
# Usage (standalone):
#
#     DFLASH_TEST_TARGET=... DFLASH_TEST_DRAFT=... \
#         ./tests/test-dflash-correctness.sh \
#             [--quick|--full] [--matrix] [--bin-dir DIR] [-v]
#
# Modes:
#   --quick (default)  3 prompts × n=32  → ~10s on Vulkan + 2× RTX 5090
#                      (3 cases). Covers one math + one code + one NL prompt.
#   --full             6 prompts × {32,64,128} → ~80s (canonical 18-case gate).
#                      Set DFLASH_TEST_FULL=1 to default to full mode.
#
# Matrix mode (orthogonal to --quick / --full):
#   --matrix           Run the full cross-product of {default, kv-q4, topk-4,
#                      kv-q4-topk-4} configs × the active prompt grid. Quick
#                      matrix = 3 cases × 4 configs = 12 cases (~30s); full
#                      matrix = 18 × 4 = 72 cases (~5 min). Each prompt's
#                      target reference is computed ONCE and reused across
#                      configs (matters for full matrix). Hardens regression
#                      detection for the surface area added in commits 24
#                      (-ctkd / -ctvd) and 25 (--dflash-topk K). Set
#                      DFLASH_TEST_MATRIX=1 to default to matrix mode.
#
#   Configs exercised in matrix mode:
#     default       no extra spec flags (chain mode, F16 KV)
#     kv-q4         -ctkd q4_0 -ctvd q4_0          (commit 24)
#     topk-4        --dflash-topk 4                 (commit 25; argsort_top_k path)
#     kv-q4-topk-4  -ctkd q4_0 -ctvd q4_0 --dflash-topk 4
#
# Exit codes:
#   0   all tests PASS, OR models missing (skipped per get-model.cpp convention)
#   1   at least one test FAILed (byte-exact match broken — algorithmic regression)
#   2   binary lookup / setup failure (cannot run the test)

set -u
set -o pipefail

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

mode="${DFLASH_TEST_FULL:+full}"
mode="${mode:-quick}"
matrix_mode="${DFLASH_TEST_MATRIX:+1}"
matrix_mode="${matrix_mode:-0}"
bin_dir=""
verbose=0

while [ $# -gt 0 ]; do
    case "$1" in
        --quick)   mode="quick"; shift ;;
        --full)    mode="full";  shift ;;
        --matrix)  matrix_mode=1; shift ;;
        --bin-dir) bin_dir="$2"; shift 2 ;;
        -v|--verbose) verbose=1; shift ;;
        -h|--help)
            sed -n '3,76p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

# ---------------------------------------------------------------------------
# Locate binaries
# ---------------------------------------------------------------------------
#
# Resolution order:
#   1. --bin-dir CLI arg
#   2. ./           (CTest invocation: WORKING_DIRECTORY is build/bin/, so
#                    the binaries are siblings of the script)
#   3. ./build/bin  (standalone invocation from repo root)

if [ -z "$bin_dir" ]; then
    if [ -x "./llama-cli" ] && [ -x "./llama-speculative-simple" ]; then
        bin_dir="."
    elif [ -x "./build/bin/llama-cli" ] && [ -x "./build/bin/llama-speculative-simple" ]; then
        bin_dir="./build/bin"
    else
        printf '\033[31mERROR: cannot locate llama-cli + llama-speculative-simple.\033[0m\n' >&2
        printf 'Tried: ./, ./build/bin/. Pass --bin-dir DIR.\n' >&2
        exit 2
    fi
fi

llama_cli="$bin_dir/llama-cli"
llama_spec="$bin_dir/llama-speculative-simple"

if [ ! -x "$llama_cli" ] || [ ! -x "$llama_spec" ]; then
    printf '\033[31mERROR: %s or %s not executable.\033[0m\n' "$llama_cli" "$llama_spec" >&2
    exit 2
fi

# ---------------------------------------------------------------------------
# Locate models (skip-on-missing per tests/get-model.cpp convention)
# ---------------------------------------------------------------------------

target_model="${DFLASH_TEST_TARGET:-}"
draft_model="${DFLASH_TEST_DRAFT:-}"

if [ -z "$target_model" ] || [ -z "$draft_model" ]; then
    printf '\033[33mWARNING: DFLASH_TEST_TARGET and/or DFLASH_TEST_DRAFT not set. '
    printf 'Skipping DFlash strong-correctness test.\n'
    printf 'Set both env vars to enable:\n'
    printf '  DFLASH_TEST_TARGET=/path/to/target.gguf\n'
    printf '  DFLASH_TEST_DRAFT=/path/to/draft-dflash.gguf\033[0m\n'
    exit 0
fi

if [ ! -r "$target_model" ]; then
    printf '\033[33mWARNING: target model %s not readable. Skipping.\033[0m\n' "$target_model"
    exit 0
fi

if [ ! -r "$draft_model" ]; then
    printf '\033[33mWARNING: draft model %s not readable. Skipping.\033[0m\n' "$draft_model"
    exit 0
fi

# ---------------------------------------------------------------------------
# Test grid + config matrix
# ---------------------------------------------------------------------------
#
# Quick mode: 3 prompts × n=32 (one math, one code, one NL — covers the
# three acceptance regimes documented in 00_HANDOFF.md).
#
# Full mode: the canonical 18-prompt gate from 05_correctness_test.md /
# 08_session_resume_prompt.md.

if [ "$mode" = "full" ]; then
    prompts=(
        "Hello, my name is"
        "The capital of France is"
        "Once upon a time,"
        "function fibonacci(n) {"
        "def add(a, b):"
        "1 + 1 = "
    )
    lengths=(32 64 128)
else
    prompts=(
        "1 + 1 = "
        "function fibonacci(n) {"
        "Hello, my name is"
    )
    lengths=(32)
fi

# Configs to exercise. In default (non-matrix) mode we only run "default" —
# byte-identical to pre-matrix behavior. In matrix mode we run all 4 to
# cover the cross-product of -ctkd/-ctvd (commit 24) × --dflash-topk K
# (commit 25). The order matters: "default" first so a regression in the
# common path surfaces immediately, then the variants.
if [ "$matrix_mode" = "1" ]; then
    active_configs=("default" "kv-q4" "topk-4" "kv-q4-topk-4" "tree-medusa-k2" "tree-budget-22")
else
    active_configs=("default")
fi

# Map a config name to the list of EXTRA spec args it appends to the
# baseline `llama-speculative-simple --draft-type dflash` invocation. Empty
# string for "default" → unquoted expansion in the invocation contributes
# zero argv tokens (correct).
spec_args_for_config() {
    case "$1" in
        default)         echo "" ;;
        kv-q4)           echo "-ctkd q4_0 -ctvd q4_0" ;;
        topk-4)          echo "--dflash-topk 4" ;;
        kv-q4-topk-4)    echo "-ctkd q4_0 -ctvd q4_0 --dflash-topk 4" ;;
        tree-medusa-k2)  echo "--dflash-tree" ;;
        tree-budget-22)  echo "--dflash-tree-budget 22" ;;
        *)
            printf '\033[31mERROR: unknown config name: %s\033[0m\n' "$1" >&2
            exit 2
            ;;
    esac
}

# ---------------------------------------------------------------------------
# Run the sweep
# ---------------------------------------------------------------------------

# Per-test scratch files kept in a private tempdir; cleaned on exit.
tmpdir="$(mktemp -d -t dflash-correctness-XXXXXX)"
trap 'rm -rf "$tmpdir"' EXIT

target_out="$tmpdir/target.txt"
spec_raw="$tmpdir/spec.raw"
spec_out="$tmpdir/spec.txt"
spec_log="$tmpdir/spec.log"
target_log="$tmpdir/target.log"

# Per-config + grand-total counters. Associative arrays require bash 4+;
# our shebang is `#!/usr/bin/env bash` and the script's other constructs
# (e.g. `${arr[@]}` with `set -u`) already require modern bash.
declare -A config_pass
declare -A config_fail
for c in "${active_configs[@]}"; do
    config_pass[$c]=0
    config_fail[$c]=0
done
n_pass=0
n_fail=0
fail_lines=()

# Use bytes (LANG=C) to make text comparisons truly byte-exact.
export LANG=C

# Mark the current case as FAILed: bump the per-config + grand-total
# counters, record the line for the end-of-run replay, print it
# immediately, and (in verbose mode) dump the named log file. $prompt and
# $n are inherited from the surrounding loops; $config is passed as the
# first arg.
record_fail() {
    local config="$1"
    local reason="$2"
    local log_file="${3:-}"
    config_fail[$config]=$(( ${config_fail[$config]} + 1 ))
    n_fail=$(( n_fail + 1 ))
    local line
    if [ "$matrix_mode" = "1" ]; then
        line=$(printf '  FAIL [config=%s] [%s] n=%d (%s)' "$config" "$prompt" "$n" "$reason")
    else
        line=$(printf '  FAIL [%s] n=%d (%s)' "$prompt" "$n" "$reason")
    fi
    fail_lines+=("$line")
    printf '%s\n' "$line"
    if [ "$verbose" -eq 1 ] && [ -n "$log_file" ] && [ -s "$log_file" ]; then
        sed 's/^/    > /' "$log_file" >&2
    fi
}

# Strip the prompt prefix from the speculative-simple raw output (it always
# echoes the prompt; --no-display-prompt is gated to LLAMA_EXAMPLE_MAIN —
# see 00_HANDOFF.md "Two operational notes") into $spec_out, then rstrip
# both files (the two binaries pad with different trailing-newline counts)
# and byte-compare. Single python invocation per case. Side effect: writes
# $spec_out so the verbose-mode dump shows what was actually compared.
# Prints YES on match, NO otherwise.
compare_outputs() {
    local target_file="$1"
    local spec_raw_file="$2"
    local spec_out_file="$3"
    local raw_prompt="$4"
    python3 - "$target_file" "$spec_raw_file" "$spec_out_file" "$raw_prompt" <<'PY'
import sys
t   = open(sys.argv[1], 'rb').read().rstrip()
raw = open(sys.argv[2], 'rb').read().lstrip()
p   = sys.argv[4].encode()
d   = raw[len(p):] if raw.startswith(p) else raw
open(sys.argv[3], 'wb').write(d)
sys.stdout.write('YES' if t == d.rstrip() else 'NO')
PY
}

n_cases=$(( ${#prompts[@]} * ${#lengths[@]} * ${#active_configs[@]} ))
printf 'DFlash strong-correctness sweep (%s mode%s)\n' \
    "$mode" "$([ "$matrix_mode" = "1" ] && echo ', matrix')"
printf '  target binary : %s\n' "$llama_cli"
printf '  spec binary   : %s\n' "$llama_spec"
printf '  target model  : %s\n' "$target_model"
printf '  draft model   : %s\n' "$draft_model"
if [ "$matrix_mode" = "1" ]; then
    printf '  configs       : %d (%s)\n' "${#active_configs[@]}" "${active_configs[*]}"
    printf '  grid          : %d prompts × %d lengths × %d configs = %d cases\n' \
        "${#prompts[@]}" "${#lengths[@]}" "${#active_configs[@]}" "$n_cases"
else
    printf '  grid          : %d prompts × %d lengths = %d cases\n' \
        "${#prompts[@]}" "${#lengths[@]}" "$n_cases"
fi
printf '\n'

for prompt in "${prompts[@]}"; do
    for n in "${lengths[@]}"; do
        # Greedy reference via llama-cli — runs ONCE per (prompt, n) and
        # is reused across all active configs. -ub 1 is required for
        # byte-exact match; documented in 00_HANDOFF.md "Two operational
        # notes". A target-side failure here is fatal for every config in
        # this case, so we record the FAIL once per active config (each
        # config's tally increments by 1) and skip to the next case.
        if ! "$llama_cli" \
                -m "$target_model" \
                -p "$prompt" --n-predict "$n" --temp 0 --seed 0 \
                -no-cnv --no-display-prompt -ngl 99 -ub 1 \
                > "$target_out" 2> "$target_log"; then
            for config in "${active_configs[@]}"; do
                record_fail "$config" "llama-cli exited non-zero" "$target_log"
            done
            continue
        fi

        for config in "${active_configs[@]}"; do
            # `extra_args` contains a space-separated arg list; the
            # unquoted expansion below word-splits it into individual argv
            # tokens. Empty string (the "default" config) contributes
            # zero tokens. None of the values contain whitespace or
            # globs, so word-splitting is safe here.
            extra_args=$(spec_args_for_config "$config")

            # DFlash speculative via llama-speculative-simple --draft-type dflash.
            # shellcheck disable=SC2086 # intentional word-split of $extra_args
            if ! "$llama_spec" \
                    -m  "$target_model" \
                    -md "$draft_model" \
                    --draft-type dflash \
                    $extra_args \
                    -p "$prompt" --n-predict "$n" --temp 0 --seed 0 \
                    -ngl 99 -ngld 99 -ub 1 \
                    > "$spec_raw" 2> "$spec_log"; then
                record_fail "$config" "llama-speculative-simple exited non-zero" "$spec_log"
                continue
            fi

            if [ "$(compare_outputs "$target_out" "$spec_raw" "$spec_out" "$prompt")" = "YES" ]; then
                config_pass[$config]=$(( ${config_pass[$config]} + 1 ))
                n_pass=$(( n_pass + 1 ))
                if [ "$verbose" -eq 1 ]; then
                    if [ "$matrix_mode" = "1" ]; then
                        printf '  PASS [config=%s] [%s] n=%d\n' "$config" "$prompt" "$n"
                    else
                        printf '  PASS [%s] n=%d\n' "$prompt" "$n"
                    fi
                fi
            else
                record_fail "$config" "output mismatch"
                if [ "$verbose" -eq 1 ]; then
                    printf '    target> %s\n' "$(head -c 200 "$target_out" | tr '\n' ' ')" >&2
                    printf '    spec>   %s\n' "$(head -c 200 "$spec_out"   | tr '\n' ' ')" >&2
                fi
            fi
        done
    done
done

# ---------------------------------------------------------------------------
# Tally
# ---------------------------------------------------------------------------

n_total=$(( n_pass + n_fail ))

# Per-config breakdown shown only in matrix mode (otherwise it's just one
# row that adds noise vs. the single-line legacy summary).
if [ "$matrix_mode" = "1" ]; then
    printf '\nPer-config breakdown:\n'
    for c in "${active_configs[@]}"; do
        local_total=$(( ${config_pass[$c]} + ${config_fail[$c]} ))
        printf '  %-16s %d PASS, %d FAIL  (out of %d cases)\n' \
            "$c" "${config_pass[$c]}" "${config_fail[$c]}" "$local_total"
    done
fi

printf '\nTOTAL: %d PASS, %d FAIL  (out of %d cases)\n' "$n_pass" "$n_fail" "$n_total"

if [ "$n_fail" -gt 0 ]; then
    printf '\nFailures:\n'
    for line in "${fail_lines[@]}"; do
        printf '%s\n' "$line"
    done
    exit 1
fi

exit 0
