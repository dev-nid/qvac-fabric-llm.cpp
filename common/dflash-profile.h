// dflash-profile.h
//
// Per-op profiler for the DFlash speculative-decoding driver.
//
// This is an opt-in observability tool. When DFLASH_PROF=1 is set in the
// environment, the driver installs the eval_callback() below as the
// `cb_eval` for both ctx_tgt and ctx_dft. The callback fires once per
// ggml node during graph compute (twice actually: ask=true / ask=false).
// We bracket each node with ggml_time_us() and accumulate per-bucket
// stats keyed by (phase, op_name, shape).
//
// Returning `true` from ask=true forces the scheduler to compute one node
// at a time with explicit ggml_backend_synchronize() between ask=true and
// ask=false (see ggml-backend.cpp:1580-1611). For CPU compute (synchronous
// anyway) the inter-callback delta is the actual op wall time. For GPU
// backends this would (a) defeat batched submission and (b) only measure
// dispatch+sync overhead instead of compute, so this profiler is treated
// as **CPU-only** and the driver only installs it when running on CPU.
//
// Scope guardrail: lives in `common/dflash-*` (per the optimization plan in
// `logs/core_architecture/10_optimization_plan.md`); no changes required
// to ggml or non-DFlash llama.cpp subsystems.

#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

// Forward-declared instead of #include "llama.h" to keep this header light.
struct llama_context;

namespace dflash_prof {

struct op_stats {
    uint64_t count    = 0;
    uint64_t total_us = 0;
    // Captured once per bucket (constant across calls): bytes touched
    // (= sum of input + output tensor sizes) — used as a rough proxy for
    // whether the op is bandwidth- or compute-bound.
    uint64_t bytes_per_call = 0;
};

class profiler {
public:
    static profiler & instance();

    bool enabled() const { return enabled_; }
    void set_enabled(bool e) { enabled_ = e; }

    // Phase tagging — driver calls before each decode to label the bucket.
    // Common phases: "prompt-tgt", "prompt-dft", "draft", "verify",
    // "verify-tree", "rebuild", "warmup".
    void set_phase(const std::string & phase) { current_phase_ = phase; }
    void clear_phase() { current_phase_.clear(); }
    const std::string & phase() const { return current_phase_; }

    // ggml_backend_sched_eval_callback signature.
    static bool eval_callback(struct ggml_tensor * t, bool ask, void * user_data);

    // Dump sorted top-N table to `out` (defaults to stderr).
    // If `top_n <= 0`, dumps all buckets.
    void dump(FILE * out = nullptr, int top_n = 30) const;

    // Also: dump grouped by phase only (one line per phase).
    void dump_phase_summary(FILE * out = nullptr) const;

    // Also: dump grouped by op-type only (one line per op_name).
    void dump_op_summary(FILE * out = nullptr) const;

    void reset();

    size_t n_buckets() const { return stats_.size(); }
    uint64_t total_observed_us() const { return total_observed_us_; }

private:
    profiler() = default;

    bool        enabled_         = false;
    std::string current_phase_;
    int64_t     t_start_us_      = 0;
    std::string current_key_;

    // key: "<phase> | <op_name> | [ne0,ne1,ne2,ne3] | <type>"
    std::unordered_map<std::string, op_stats> stats_;

    uint64_t total_observed_us_ = 0;
};

// Helper for the driver: read DFLASH_PROF / DFLASH_PROF_FILE / DFLASH_PROF_TOP_N
// env vars and apply them to `params`. Returns true if profiling was enabled.
// Caller still needs to call set_phase() at appropriate points and dump() at exit.
//
// Signature is a struct ref (rather than &common_params) so this header has
// no dependency on common.h — keeps DFlash code self-contained.
struct prof_env_config {
    bool        enabled = false;
    std::string out_file;          // empty -> stderr
    int         top_n   = 30;
};

prof_env_config read_env_config();

// ===========================================================================
// Memory profiler — opt-in via DFLASH_MEM=1
// ===========================================================================
//
// Companion to the per-op (timing) profiler. Where the per-op profiler logs
// time per ggml op, this logs BYTES per DFlash-specific allocation bucket
// + process-level RSS / VRAM. Used to find memory-optimization candidates
// (M1-M4 in logs/core_architecture/10_optimization_plan.md).
//
// Coverage:
//   * DFlash buckets via the public C API `llama_dflash_memory_breakdown`:
//     side store K/V (per-layer ggml backend tensors), captured_features
//     and capture_staging (host buffers), draft_topk and draft_topk_argmax.
//   * Process RSS via /proc/self/status (Linux only).
//   * VRAM via ggml_backend_dev_memory() if a device is present.
//
// Coverage gap (relies on existing tools, not this profiler):
//   * Generic llama allocations (model weights, context KV cache, compute
//     graph) — those are covered by `llama_memory_breakdown_print()` which
//     speculative-simple already prints.
//
// Snapshot pattern: take a snapshot at well-defined points in the lifecycle
// (e.g. "after_warmup", "after_5_iters", "exit"). The dump compares them
// side-by-side to surface allocation churn vs. steady-state usage.

struct mem_snapshot {
    std::string label;

    // From llama_dflash_memory_breakdown — sums for the draft context (the
    // primary DFlash consumer; target context's dflash buckets are 0 since
    // it doesn't have a side store).
    size_t side_store_K_bytes        = 0;
    size_t side_store_V_bytes        = 0;
    size_t captured_features_bytes   = 0;
    size_t capture_staging_bytes     = 0;
    size_t draft_topk_bytes          = 0;
    size_t draft_topk_argmax_bytes   = 0;
    int64_t ctx_capacity             = 0;
    int64_t ctx_filled               = 0;
    int     n_layers                 = 0;

    // Process-level. RSS is total resident in physical memory; VRAM_used is
    // sum across present GPU devices (if any).
    size_t rss_bytes      = 0;
    size_t vram_used_bytes = 0;
    size_t vram_free_bytes = 0;
};

class memory_profiler {
public:
    static memory_profiler & instance();

    bool enabled() const { return enabled_; }
    void set_enabled(bool e) { enabled_ = e; }

    // Take a snapshot, optionally walking the DFlash struct on each context.
    // The DFlash buckets are split across contexts:
    //   draft_ctx:  side_store_K/V, draft_topk, draft_topk_argmax
    //   target_ctx: captured_features, capture_staging
    // Pass either as nullptr to skip that side (e.g. nullptr for both on a
    // "before-load" snapshot, or only draft_ctx on a chain-only run).
    void snapshot(const std::string & label,
                  const struct llama_context * draft_ctx,
                  const struct llama_context * target_ctx);

    // Dump all collected snapshots side-by-side.
    void dump(FILE * out = nullptr) const;

    // Reset.
    void reset();

    // Helpers (public for testing / direct use):
    static size_t process_rss_bytes();   // 0 on non-Linux
    static void   query_vram(size_t * used_out, size_t * free_out);

private:
    memory_profiler() = default;

    bool                       enabled_ = false;
    std::vector<mem_snapshot>  snapshots_;
};

struct mem_env_config {
    bool        enabled = false;
    std::string out_file;   // empty -> stderr
};

mem_env_config read_mem_env_config();

}  // namespace dflash_prof
