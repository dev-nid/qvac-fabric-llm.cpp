#!/usr/bin/env bash
# scripts/dflash_bench_cold.sh
#
# COLD-PROCESS variant of dflash_bench_cpu.sh — spawns a fresh
# llama-speculative-simple per (config, prompt) so the Vulkan pipeline
# cache is cold for the first DFlash invocation in each run. This is
# the realistic single-shot performance, vs the warm-cache numbers
# produced by the standard bench harness (which runs greedy first and
# primes the shader cache).
#
# Outputs JSON in the same schema as dflash_bench_cpu.sh so the same
# render pipeline can pick it up.

set -u
set -o pipefail

bin_dir="${1:-build/bin}"
target="${DFLASH_TEST_TARGET:?need DFLASH_TEST_TARGET}"
draft="${DFLASH_TEST_DRAFT:?need DFLASH_TEST_DRAFT}"
out="${OUT:-bench/gpu/$(git rev-parse --short HEAD)-COLD_$(date -u +%Y-%m-%dT%H-%M-%SZ).json}"
n_predict="${N_PREDICT:-128}"
seed="${SEED:-0}"

declare -a configs=(greedy chain chain-topk-2 tree-medusa-k2 tree-budget-18 tree-budget-22)
declare -a prompts=(math nl-low-entropy nl-high-entropy code conv)

prompt_text() {
    case "$1" in
        math)            printf 'Solve: 47 * 83 = ' ;;
        nl-low-entropy)  printf 'The capital of France is' ;;
        nl-high-entropy) printf 'Once upon a time, in a land far away,' ;;
        code)            printf 'Write a Python function that returns the Fibonacci sequence:\n' ;;
        conv)            printf 'User: How do I learn programming?\nAssistant:' ;;
    esac
}
spec_args() {
    case "$1" in
        chain)            echo "" ;;
        chain-topk-2)     echo "--dflash-topk 2" ;;
        tree-medusa-k2)   echo "--dflash-tree" ;;
        tree-budget-18)   echo "--dflash-tree-budget 18" ;;
        tree-budget-22)   echo "--dflash-tree-budget 22" ;;
    esac
}

mkdir -p "$(dirname "$out")"
git_sha="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
host="$(hostname -s 2>/dev/null || echo unknown)"
ts="$(date -u +%Y-%m-%dT%H-%M-%SZ)"
total_start=$(date +%s)

results_file="$(mktemp)"
echo '[' > "$results_file"
first=1

i=0
total=$(( ${#configs[@]} * ${#prompts[@]} ))

for prompt in "${prompts[@]}"; do
    ptext="$(prompt_text "$prompt")"
    for config in "${configs[@]}"; do
        i=$((i + 1))
        printf '[%2d/%2d] %-15s | %-15s | ' "$i" "$total" "$config" "$prompt" >&2

        log="$(mktemp)"
        if [ "$config" = "greedy" ]; then
            "$bin_dir/llama-completion" -m "$target" \
                -p "$ptext" --n-predict "$n_predict" --temp 0 --seed "$seed" \
                -no-cnv --no-display-prompt -ngl 99 -t 8 \
                </dev/null > /dev/null 2> "$log"
        else
            extra="$(spec_args "$config")"
            # shellcheck disable=SC2086
            "$bin_dir/llama-speculative-simple" \
                -m "$target" -md "$draft" --draft-type dflash $extra \
                -p "$ptext" --n-predict "$n_predict" --temp 0 --seed "$seed" \
                -ngl 99 -ngld 99 -t 8 \
                </dev/null > /dev/null 2> "$log"
        fi
        rc=$?

        # Parse like dflash_bench_cpu.sh
        result_json="$(python3 - <<PY
import json, re, sys
config = "$config"
prompt = "$prompt"
n_predict = $n_predict
log_path = "$log"
with open(log_path, 'rb') as f:
    log = f.read().decode('utf-8', errors='replace')
result = {
    'config': config,
    'prompt': prompt,
    'n_predict_requested': n_predict,
    'tokens_per_sec': None,
    'tokens_predicted': None,
    'wall_ms': None,
    'prompt_eval_ms': None,
    'gen_eval_ms': None,
    'n_drafted': None,
    'n_accepted': None,
    'accept_rate_pct': None,
    'tree_iters': None,
    'ok': False,
}
m = re.search(r'decoded\s+(\d+)\s+tokens in\s+([\d.]+)\s+seconds,\s+speed:\s+([\d.]+)\s+t/s', log)
if m:
    result['tokens_predicted'] = int(m.group(1))
    result['wall_ms'] = float(m.group(2)) * 1000.0
    result['tokens_per_sec'] = float(m.group(3))
if result['tokens_per_sec'] is None:
    m = re.search(r'common_perf_print:\s+eval time =\s+([\d.]+)\s+ms /\s+(\d+)\s+runs.*?(\d+(?:\.\d+)?)\s+tokens per second', log)
    if m and float(m.group(1)) > 0 and int(m.group(2)) > 1:
        result['tokens_per_sec'] = float(m.group(3))
        result['tokens_predicted'] = int(m.group(2))
m = re.search(r'common_perf_print:\s+total time =\s+([\d.]+)\s+ms', log)
if m and result['wall_ms'] is None:
    result['wall_ms'] = float(m.group(1))
m = re.search(r'common_perf_print:\s+prompt eval time =\s+([\d.]+)\s+ms', log)
if m: result['prompt_eval_ms'] = float(m.group(1))
m = re.search(r'common_perf_print:\s+eval time =\s+([\d.]+)\s+ms', log)
if m: result['gen_eval_ms'] = float(m.group(1))
m = re.search(r'n_drafted\s+=\s+(\d+)', log)
if m: result['n_drafted'] = int(m.group(1))
m = re.search(r'n_accept\s+=\s+(\d+)', log)
if m: result['n_accepted'] = int(m.group(1))
m = re.search(r'accept\s+=\s+([\d.]+)%', log)
if m: result['accept_rate_pct'] = float(m.group(1))
m = re.search(r'tree iters\s+=\s+(\d+)', log)
if m: result['tree_iters'] = int(m.group(1))
result['ok'] = result['tokens_per_sec'] is not None and result['tokens_per_sec'] > 0
print(json.dumps(result))
PY
)"
        rm -f "$log"

        if [ "$first" = "1" ]; then
            first=0
        else
            echo ',' >> "$results_file"
        fi
        echo "$result_json" >> "$results_file"

        tps="$(echo "$result_json" | python3 -c "import json,sys; print(json.loads(sys.stdin.read()).get('tokens_per_sec') or 0)")"
        printf '%6.2f tok/s\n' "$tps" >&2
    done
done
echo ']' >> "$results_file"

total_dt=$(( $(date +%s) - total_start ))

# Wrap as a full snapshot
python3 - "$out" "$results_file" "$git_sha" "$host" "$ts" "$bin_dir" "$target" "$draft" "$n_predict" "$total_dt" <<'PY'
import json, sys
out, results_path, git_sha, host, ts, bin_dir, target, draft, n_predict, total_dt = sys.argv[1:11]
with open(results_path) as f:
    results = json.load(f)
doc = {
    'schema_version': 1,
    'git_sha': git_sha,
    'host': host,
    'timestamp_utc': ts,
    'bin_dir': bin_dir,
    'target_model': target,
    'draft_model': draft,
    'n_predict': int(n_predict),
    'threads': 8,
    'seed': 0,
    'total_wall_seconds': int(total_dt),
    'cold_process': True,
    'results': results,
}
with open(out, 'w') as f:
    json.dump(doc, f, indent=2)
print(f'wrote {out}', file=sys.stderr)
PY

rm -f "$results_file"
echo "wrote $out"
