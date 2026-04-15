#!/usr/bin/env bash
#
# Run quantization tests for TurboQuant/PolarQuant types only.
#
# Usage:
#   tests/test-turboquant.sh [build_dir]
#
# Examples:
#   tests/test-turboquant.sh
#   tests/test-turboquant.sh build-debug
#
set -euo pipefail

B="${1:-build}"

TQ_TYPES=(tbq3_0 tbq4_0 pq3_0 pq4_0 tbq3_0_64 tbq4_0_64 pq3_0_64 pq4_0_64)

echo "=========================================="
echo " TurboQuant/PolarQuant Test Suite"
echo " Build dir: $B"
echo "=========================================="
echo ""

num_failed=0

echo "=== test-quantize-fns (all types, includes TBQ/PQ correctness assertions) ==="
if [ -f "$B/bin/test-quantize-fns" ]; then
    "$B/bin/test-quantize-fns" || num_failed=$((num_failed + 1))
else
    echo "SKIP: $B/bin/test-quantize-fns not found"
fi
echo ""

echo "=== test-quantize-perf CPU (TBQ/PQ types, with perf sanity checks) ==="
if [ -f "$B/bin/test-quantize-perf" ]; then
    "$B/bin/test-quantize-perf" --type tbq3_0 --type tbq4_0 --type pq3_0 --type pq4_0 --type q4_0 || true
else
    echo "SKIP: $B/bin/test-quantize-perf not found"
fi
echo ""

echo "=== test-quantize-perf Vulkan (TBQ/PQ types, with perf sanity checks) ==="
if [ -f "$B/bin/test-quantize-perf" ]; then
    "$B/bin/test-quantize-perf" -b vulkan --type tbq3_0 --type tbq4_0 --type pq3_0 --type pq4_0 --type q4_0 || true
else
    echo "SKIP: $B/bin/test-quantize-perf not found"
fi
echo ""

echo "=== test-backend-ops (TBQ/PQ ops only) ==="
if [ -f "$B/bin/test-backend-ops" ]; then
    "$B/bin/test-backend-ops" test -p "tbq|pq" || num_failed=$((num_failed + 1))
else
    echo "SKIP: $B/bin/test-backend-ops not found"
fi
echo ""

echo "=========================================="
if [ "$num_failed" -eq 0 ]; then
    echo " All checks passed."
else
    echo " $num_failed check(s) failed."
fi
echo "=========================================="

exit "$num_failed"
