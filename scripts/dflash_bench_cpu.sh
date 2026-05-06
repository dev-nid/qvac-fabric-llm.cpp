#!/usr/bin/env bash
#
# scripts/dflash_bench_cpu.sh
#
# CPU bench harness for DFlash speculative-decoding optimization work.
# Companion of the optimization plan in
# logs/core_architecture/10_optimization_plan.md (sessions 1..N).
#
# What it does:
#   1. Runs each (config, prompt) tuple in `configs[] x prompts[]` once with
#      the CPU binaries from --bin-dir (default: build-cpu/bin).
#   2. Parses tokens-per-sec, accept-rate, and the per-phase timing block
#      out of stderr.
#   3. Emits one JSON file at $OUT containing all results — designed to be
#      consumed by `scripts/dflash_bench_diff.py` for before/after compare.
#
# Configs covered (matches the matrix CTest set + a CPU baseline):
#   greedy            llama-cli (no spec, no draft)
#   chain             llama-speculative-simple --draft-type dflash
#   chain-topk-2      + --dflash-topk 2
#   tree-medusa-k2    + --dflash-tree                (Stage B)
#   tree-budget-18    + --dflash-tree-budget 18      (Stage C, sweet spot)
#   tree-budget-22    + --dflash-tree-budget 22      (Stage C, Lucebox parity)
#
# Prompts (5 representative classes; lifted from the matrix CTest):
#   math, nl-low-entropy, nl-high-entropy, code, conv
#
# Skip-on-missing-models: exits 0 with a yellow warning if either
# DFLASH_TEST_TARGET or DFLASH_TEST_DRAFT env vars are unset or missing.
# Mirrors test-dflash-correctness.sh's get-model.cpp convention.
#
# Usage:
#   DFLASH_TEST_TARGET=/path/to/Qwen3-4B-Q8_0.gguf \
#   DFLASH_TEST_DRAFT=/path/to/Qwen3-4B-DFlash-b16.gguf \
#       ./scripts/dflash_bench_cpu.sh \
#           [--bin-dir DIR]    # default: build-cpu/bin
#           [--out OUT.json]   # default: bench/cpu/<git-sha>.json
#           [--n N]            # n_predict per run (default: 128)
#           [--threads T]      # OMP threads (default: $(nproc))
#           [--quick]          # 1 prompt, n_predict=32 (~30s smoke run)
#           [--configs LIST]   # comma-sep subset, e.g. "greedy,chain"
#           [--prompts LIST]   # comma-sep subset, e.g. "math,code"
#           [--prof]           # also run with DFLASH_PROF=1 and capture per-op
#           [--seed S]         # default: 0
#           [-v]               # verbose: print stderr from each run
#
# Exit codes:
#   0  all runs completed (regardless of perf), JSON emitted (or skipped)
#   1  at least one run failed (non-zero exit, no tokens-per-sec parsed)
#   2  binary lookup / setup failure

set -u
set -o pipefail

# --------------------------------------------------------------------------
# Defaults & arg parsing
# --------------------------------------------------------------------------

bin_dir="build-cpu/bin"
out_path=""
n_predict=128
threads="$(nproc)"
seed=0
verbose=0
prof_mode=0
configs_filter=""
prompts_filter=""
quick=0
# Physical batch size (-ub). Default empty = use the binary's default
# (typically n_batch / 2048). The correctness gate uses -ub 1 to force
# byte-exact reduction order against llama-cli; for performance bench we
# want the natural batching so multi-token verify ubatches read each weight
# tile ONCE, not 17 times. Set --ub 1 explicitly to reproduce the
# correctness-gate timing if needed.
ubatch=""

while [ $# -gt 0 ]; do
    case "$1" in
        --bin-dir) bin_dir="$2"; shift 2 ;;
        --out)     out_path="$2"; shift 2 ;;
        --n)       n_predict="$2"; shift 2 ;;
        --threads) threads="$2"; shift 2 ;;
        --seed)    seed="$2"; shift 2 ;;
        --quick)   quick=1; shift ;;
        --configs) configs_filter="$2"; shift 2 ;;
        --prompts) prompts_filter="$2"; shift 2 ;;
        --prof)    prof_mode=1; shift ;;
        --ub)      ubatch="$2"; shift 2 ;;
        -v|--verbose) verbose=1; shift ;;
        -h|--help)
            sed -n '3,55p' "$0" | sed 's/^# \{0,1\}//'
            exit 0
            ;;
        *)
            printf 'unknown argument: %s\n' "$1" >&2
            exit 2
            ;;
    esac
done

if [ "$quick" = "1" ]; then
    n_predict=32
    prompts_filter="${prompts_filter:-math}"
fi

# --------------------------------------------------------------------------
# Skip-on-missing-models (test-dflash-correctness.sh convention)
# --------------------------------------------------------------------------

target_model="${DFLASH_TEST_TARGET:-}"
draft_model="${DFLASH_TEST_DRAFT:-}"

if [ -z "$target_model" ] || [ ! -f "$target_model" ]; then
    printf '\033[33m[skip]\033[0m DFLASH_TEST_TARGET unset or missing — nothing to bench\n' >&2
    exit 0
fi
if [ -z "$draft_model" ] || [ ! -f "$draft_model" ]; then
    printf '\033[33m[skip]\033[0m DFLASH_TEST_DRAFT unset or missing — nothing to bench\n' >&2
    exit 0
fi

# --------------------------------------------------------------------------
# Binary lookup
# --------------------------------------------------------------------------

llama_cli="$bin_dir/llama-cli"
llama_spec="$bin_dir/llama-speculative-simple"

if [ ! -x "$llama_cli" ]; then
    printf '\033[31m[error]\033[0m %s not found / not executable\n' "$llama_cli" >&2
    exit 2
fi
if [ ! -x "$llama_spec" ]; then
    printf '\033[31m[error]\033[0m %s not found / not executable\n' "$llama_spec" >&2
    exit 2
fi

# --------------------------------------------------------------------------
# Output path
# --------------------------------------------------------------------------

git_sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
git_dirty=""
if git diff --quiet 2>/dev/null && git diff --cached --quiet 2>/dev/null; then
    git_dirty=""
else
    git_dirty="-dirty"
fi
host="$(hostname -s 2>/dev/null || echo unknown)"
ts="$(date -u +%Y-%m-%dT%H-%M-%SZ)"

if [ -z "$out_path" ]; then
    mkdir -p bench/cpu
    out_path="bench/cpu/${git_sha}${git_dirty}_${ts}.json"
fi

# Make a tempdir to hold per-run logs; cleaned on exit.
tmpdir="$(mktemp -d -t dflash-bench-XXXXXX)"
trap 'rm -rf "$tmpdir"' EXIT

# --------------------------------------------------------------------------
# Configs (name → extra args). "greedy" is special-cased (uses llama-cli).
# --------------------------------------------------------------------------

declare -a all_configs=(
    greedy
    chain
    chain-topk-2
    tree-medusa-k2
    tree-budget-18
    tree-budget-22
)

spec_args_for_config() {
    case "$1" in
        chain)            echo "" ;;
        chain-topk-2)     echo "--dflash-topk 2" ;;
        tree-medusa-k2)   echo "--dflash-tree" ;;
        tree-budget-18)   echo "--dflash-tree-budget 18" ;;
        tree-budget-22)   echo "--dflash-tree-budget 22" ;;
        *)                printf 'unknown config: %s\n' "$1" >&2; exit 2 ;;
    esac
}

# --------------------------------------------------------------------------
# Prompts (name → text). Same set as the matrix CTest plus 5-class coverage.
# --------------------------------------------------------------------------

declare -a all_prompts=(
    math
    nl-low-entropy
    nl-high-entropy
    code
    conv
)

prompt_text_for() {
    case "$1" in
        math)            printf 'Solve: 47 * 83 = ' ;;
        nl-low-entropy)  printf 'The capital of France is' ;;
        nl-high-entropy) printf 'Once upon a time, in a land far away,' ;;
        code)            printf 'Write a Python function that returns the Fibonacci sequence:\n' ;;
        conv)            printf 'User: How do I learn programming?\nAssistant:' ;;
        *)               printf 'unknown prompt: %s\n' "$1" >&2; exit 2 ;;
    esac
}

# --------------------------------------------------------------------------
# Filter helpers
# --------------------------------------------------------------------------

contains_csv() {
    local hay="$1" needle="$2"
    case ",$hay," in *,"$needle",*) return 0 ;; esac
    return 1
}

declare -a active_configs=()
for c in "${all_configs[@]}"; do
    if [ -z "$configs_filter" ] || contains_csv "$configs_filter" "$c"; then
        active_configs+=("$c")
    fi
done

declare -a active_prompts=()
for p in "${all_prompts[@]}"; do
    if [ -z "$prompts_filter" ] || contains_csv "$prompts_filter" "$p"; then
        active_prompts+=("$p")
    fi
done

if [ ${#active_configs[@]} -eq 0 ]; then
    printf '\033[31m[error]\033[0m no configs selected (filter: "%s")\n' "$configs_filter" >&2
    exit 2
fi
if [ ${#active_prompts[@]} -eq 0 ]; then
    printf '\033[31m[error]\033[0m no prompts selected (filter: "%s")\n' "$prompts_filter" >&2
    exit 2
fi

# --------------------------------------------------------------------------
# Parser (Python helper). Ingests a stderr/stdout log + run metadata,
# emits one JSON line per result. Used as a pipe terminator in the loop.
# --------------------------------------------------------------------------

parse_result_py='
import json
import re
import sys

config = sys.argv[1]
prompt = sys.argv[2]
n_predict = int(sys.argv[3])
log_path = sys.argv[4]
prof_path = sys.argv[5] if len(sys.argv) > 5 else ""

with open(log_path, "rb") as f:
    log = f.read().decode("utf-8", errors="replace")

result = {
    "config": config,
    "prompt": prompt,
    "n_predict_requested": n_predict,
    "tokens_per_sec": None,
    "tokens_predicted": None,
    "wall_ms": None,
    "t_draft_ms": None,
    "t_verify_ms": None,
    "t_accept_ms": None,
    "t_redecode_ms": None,
    "t_other_ms": None,
    "t_iter_total_ms": None,
    "n_drafted": None,
    "n_accepted": None,
    "accept_rate_pct": None,
    "tree_iters": None,
    "tree_alt_taken_pct": None,
    "tree_redecoded_avg": None,
    "ok": False,
    "raw_log_path": log_path,
}

# Preferred: speculative-simple summary line (true user-facing throughput).
# llama-cli does not emit this line, so for greedy we fall back to eval time.
m = re.search(
    r"decoded\s+(\d+)\s+tokens in\s+([\d.]+)\s+seconds,\s+speed:\s+([\d.]+)\s+t/s",
    log,
)
if m:
    result["tokens_predicted"] = int(m.group(1))
    result["wall_ms"] = float(m.group(2)) * 1000.0
    result["tokens_per_sec"] = float(m.group(3))

# Greedy (llama-cli) reports eval time directly; spec'\''s "eval time = 0/1"
# is degenerate (all generation goes through prompt_eval batches), so we
# only use it when the spec summary line is absent.
if result["tokens_per_sec"] is None:
    m = re.search(
        r"common_perf_print:\s+eval time =\s+([\d.]+)\s+ms /\s+(\d+)\s+runs.*?(\d+(?:\.\d+)?)\s+tokens per second",
        log,
    )
    if m and float(m.group(1)) > 0 and int(m.group(2)) > 1:
        result["tokens_per_sec"] = float(m.group(3))
        result["tokens_predicted"] = int(m.group(2))

if result["wall_ms"] is None:
    m = re.search(r"common_perf_print:\s+total time =\s+([\d.]+)\s+ms", log)
    if m:
        result["wall_ms"] = float(m.group(1))

# Per-phase block from speculative-simple (lines 712-724)
for label, key in [
    ("draft",     "t_draft_ms"),
    ("verify",    "t_verify_ms"),
    ("accept",    "t_accept_ms"),
    ("re-decode", "t_redecode_ms"),
    ("other",     "t_other_ms"),
    ("TOTAL",     "t_iter_total_ms"),
]:
    pat = r"^\s+%s\s+=\s+([\d.]+)\s+ms" % re.escape(label)
    m = re.search(pat, log, re.MULTILINE)
    if m:
        result[key] = float(m.group(1))

m = re.search(r"n_predict\s+=\s+(\d+)", log)
if m and result["tokens_predicted"] is None:
    result["tokens_predicted"] = int(m.group(1))

m = re.search(r"n_drafted\s+=\s+(\d+)", log)
if m:
    result["n_drafted"] = int(m.group(1))

m = re.search(r"n_accept\s+=\s+(\d+)", log)
if m:
    result["n_accepted"] = int(m.group(1))

m = re.search(r"accept\s+=\s+([\d.]+)%", log)
if m:
    result["accept_rate_pct"] = float(m.group(1))

m = re.search(r"tree iters\s+=\s+(\d+)", log)
if m:
    result["tree_iters"] = int(m.group(1))

m = re.search(r"tree alt taken\s+=\s+\d+\s+\(([\d.]+)%\)", log)
if m:
    result["tree_alt_taken_pct"] = float(m.group(1))

m = re.search(r"tree re-decoded\s+=\s+\d+\s+tokens\s+\(([\d.]+)/iter avg\)", log)
if m:
    result["tree_redecoded_avg"] = float(m.group(1))

result["ok"] = result["tokens_per_sec"] is not None and result["tokens_per_sec"] > 0

if prof_path and result["ok"]:
    try:
        with open(prof_path, "rb") as f:
            prof_text = f.read().decode("utf-8", errors="replace")
        result["prof_dump"] = prof_text
    except Exception as e:
        result["prof_error"] = str(e)

print(json.dumps(result))
'

# --------------------------------------------------------------------------
# Run loop
# --------------------------------------------------------------------------

n_total=$(( ${#active_configs[@]} * ${#active_prompts[@]} ))
i=0
n_failed=0

declare -a result_lines=()

printf '\033[1mDFlash CPU bench\033[0m\n'
printf '  build         : %s\n' "$bin_dir"
printf '  target        : %s\n' "$target_model"
printf '  draft         : %s\n' "$draft_model"
printf '  n_predict     : %d\n' "$n_predict"
printf '  threads       : %s\n' "$threads"
printf '  ubatch        : %s\n' "${ubatch:-default}"
printf '  seed          : %s\n' "$seed"
printf '  configs       : %s\n' "${active_configs[*]}"
printf '  prompts       : %s\n' "${active_prompts[*]}"
printf '  prof          : %s\n' "$([ "$prof_mode" = "1" ] && echo yes || echo no)"
printf '  out           : %s\n' "$out_path"
printf '  git           : %s%s\n' "$git_sha" "$git_dirty"
printf '\n'

start_all=$(date +%s)

for prompt_name in "${active_prompts[@]}"; do
    prompt="$(prompt_text_for "$prompt_name")"
    for config in "${active_configs[@]}"; do
        i=$(( i + 1 ))
        log_path="$tmpdir/log_${i}.txt"
        prof_path=""
        prof_env_args=()
        if [ "$prof_mode" = "1" ]; then
            prof_path="$tmpdir/prof_${i}.txt"
            prof_env_args=(env "DFLASH_PROF=1" "DFLASH_PROF_FILE=$prof_path" "DFLASH_PROF_TOP_N=30")
        fi

        printf '[%2d/%2d] %-18s | %-16s | ' "$i" "$n_total" "$config" "$prompt_name"

        # When --ub is unset, omit the flag entirely so the binary uses its
        # default (typically equal to n_batch). Performance bench expects
        # natural batching; --ub 1 is for the byte-exact correctness gate only.
        ub_arg=()
        if [ -n "$ubatch" ]; then
            ub_arg=(-ub "$ubatch")
        fi

        start_run=$(date +%s)
        if [ "$config" = "greedy" ]; then
            # llama-cli path — no draft, no spec.
            "${prof_env_args[@]}" "$llama_cli" \
                -m "$target_model" \
                -p "$prompt" --n-predict "$n_predict" --temp 0 --seed "$seed" \
                -no-cnv --no-display-prompt -ngl 0 "${ub_arg[@]}" -t "$threads" \
                > /dev/null 2> "$log_path"
            rc=$?
        else
            # shellcheck disable=SC2086 # intentional word-split of $extra_args
            extra_args=$(spec_args_for_config "$config")
            "${prof_env_args[@]}" "$llama_spec" \
                -m  "$target_model" \
                -md "$draft_model" \
                --draft-type dflash \
                $extra_args \
                -p "$prompt" --n-predict "$n_predict" --temp 0 --seed "$seed" \
                -ngl 0 -ngld 0 "${ub_arg[@]}" -t "$threads" \
                > /dev/null 2> "$log_path"
            rc=$?
        fi
        end_run=$(date +%s)
        dt=$(( end_run - start_run ))

        # Parse → one JSON line per result. Append to result_lines.
        result_json="$(python3 -c "$parse_result_py" "$config" "$prompt_name" "$n_predict" "$log_path" "$prof_path")"

        if printf '%s' "$result_json" | python3 -c "import json,sys; sys.exit(0 if json.loads(sys.stdin.read())['ok'] else 1)"; then
            tok=$(printf '%s' "$result_json" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('tokens_per_sec') or 0)")
            printf '\033[32m%6.2f tok/s\033[0m  (%ds wall)\n' "$tok" "$dt"
        else
            printf '\033[31mFAIL (rc=%d)\033[0m  see %s\n' "$rc" "$log_path"
            n_failed=$(( n_failed + 1 ))
            if [ "$verbose" = "1" ]; then
                sed 's/^/    > /' "$log_path" | head -40 >&2
            fi
        fi

        result_lines+=("$result_json")
    done
done

end_all=$(date +%s)
total_dt=$(( end_all - start_all ))

# --------------------------------------------------------------------------
# Emit JSON
# --------------------------------------------------------------------------

mkdir -p "$(dirname "$out_path")"

# Write per-result JSON lines to a tempfile, then assemble the wrapper doc.
results_path="$tmpdir/results.jsonl"
: > "$results_path"
for line in "${result_lines[@]}"; do
    printf '%s\n' "$line" >> "$results_path"
done

python3 - "$out_path" "$results_path" "$git_sha$git_dirty" "$host" "$ts" "$bin_dir" "$target_model" "$draft_model" "$n_predict" "$threads" "$seed" "$total_dt" <<'PY'
import json, sys
out_path, results_path, git_sha, host, ts, bin_dir, target, draft, n_predict, threads, seed, total_dt = sys.argv[1:13]
with open(results_path) as f:
    results = [json.loads(line) for line in f if line.strip()]
doc = {
    "schema_version": 1,
    "git_sha": git_sha,
    "host": host,
    "timestamp_utc": ts,
    "bin_dir": bin_dir,
    "target_model": target,
    "draft_model": draft,
    "n_predict": int(n_predict),
    "threads": int(threads) if threads.isdigit() else threads,
    "seed": int(seed) if seed.isdigit() else seed,
    "total_wall_seconds": int(total_dt),
    "results": results,
}
with open(out_path, "w") as f:
    json.dump(doc, f, indent=2)
PY

printf '\n'
printf 'wrote %s\n' "$out_path"
printf 'total wall: %ds, runs: %d, failed: %d\n' "$total_dt" "$n_total" "$n_failed"

# Pretty summary table from the JSON we just wrote.
python3 - "$out_path" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
res = doc["results"]

# Build a {config -> {prompt -> tok/s}} grid
grid = {}
prompts = []
configs = []
for r in res:
    c, p = r["config"], r["prompt"]
    if c not in configs: configs.append(c)
    if p not in prompts: prompts.append(p)
    grid.setdefault(c, {})[p] = r.get("tokens_per_sec") or 0.0

print()
print("summary (tokens/sec):")
header = "  config              | " + " | ".join(f"{p:>14s}" for p in prompts)
print(header)
print("  " + "-" * (len(header) - 2))
for c in configs:
    row = f"  {c:<18s}  | " + " | ".join(f"{grid[c].get(p, 0.0):>14.2f}" for p in prompts)
    print(row)

# Speedup vs greedy if greedy is present
if "greedy" in configs:
    print()
    print("speedup vs greedy:")
    print(header)
    print("  " + "-" * (len(header) - 2))
    g = grid["greedy"]
    for c in configs:
        if c == "greedy": continue
        cells = []
        for p in prompts:
            base = g.get(p) or 0.0
            cur = grid[c].get(p) or 0.0
            ratio = (cur / base) if base > 0 else 0.0
            cells.append(f"{ratio:>14.2f}x")
        row = f"  {c:<18s}  | " + " | ".join(cells)
        print(row)
PY

if [ "$n_failed" -gt 0 ]; then
    exit 1
fi
exit 0
