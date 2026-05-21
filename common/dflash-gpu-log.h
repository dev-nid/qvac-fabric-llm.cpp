// dflash-gpu-log.h
//
// Host-observed wall-time logger for the DFlash speculative-decoding driver,
// designed for GPU backends (Vulkan / Metal / CUDA) where the CPU per-op
// profiler in dflash-profile.h would distort timings.
//
// Methodology: every measured phase calls `llama_synchronize(ctx)` before
// stopping the timer, so the host-observed elapsed time covers the GPU
// command submission AND its execution AND any required transfer back to
// host. Counters track call counts for hot paths so we can attribute
// per-generation overhead.
//
// Opt-in via DFLASH_GPU_LOG=1 (zero overhead when unset). When enabled the
// driver should:
//   1. Wrap each interesting code region in a `dflash_gpu_log::scope_timer`.
//   2. Bump counters via the helpers below at slide-left / extend / etc.
//   3. Call `dflash_gpu_log::dump_summary()` at end of run.
//
// Scope: lives entirely in `common/dflash-*` (per the optimization-plan
// scope guardrail). No ggml or libllama internal changes.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <unordered_map>
#include <vector>

struct llama_context;

namespace dflash_gpu_log {

struct phase_stats {
    uint64_t count    = 0;
    uint64_t total_ns = 0;
    uint64_t min_ns   = UINT64_MAX;
    uint64_t max_ns   = 0;
};

class logger {
public:
    static logger & instance();

    bool enabled() const { return enabled_; }
    void set_enabled(bool e) { enabled_ = e; }

    // Append `ns` to the bucket named `name`.
    void add_sample(const std::string & name, uint64_t ns);

    // Counters (cheap to bump; printed in the summary).
    void incr_extend_calls()             { ++n_extend_calls_; }
    void incr_slide_left_calls()         { ++n_slide_left_calls_; }
    void incr_capture_transpose_calls()  { ++n_capture_transpose_calls_; }
    void incr_alt_remap_calls()          { ++n_alt_remap_calls_; }
    void add_d2h_bytes(uint64_t b)       { d2h_bytes_ += b; }
    void add_h2d_bytes(uint64_t b)       { h2d_bytes_ += b; }

    void dump_summary(FILE * out = nullptr) const;
    void reset();

private:
    logger() = default;

    bool enabled_ = false;

    std::unordered_map<std::string, phase_stats> stats_;

    uint64_t n_extend_calls_             = 0;
    uint64_t n_slide_left_calls_         = 0;
    uint64_t n_capture_transpose_calls_  = 0;
    uint64_t n_alt_remap_calls_          = 0;
    uint64_t d2h_bytes_                  = 0;
    uint64_t h2d_bytes_                  = 0;
};

// RAII timer that records [now, dtor) into `name`. If `ctx` is non-null the
// timer calls `llama_synchronize(ctx)` before reading the stop timestamp,
// so the host-observed elapsed time includes any pending GPU work.
class scope_timer {
public:
    scope_timer(const char * name, llama_context * ctx);
    ~scope_timer();

private:
    const char *    name_;
    llama_context * ctx_;
    uint64_t        t_start_ns_;
    bool            active_;
};

struct env_config {
    bool        enabled  = false;
    std::string out_file;   // empty => stderr
};

env_config read_env_config();

inline void dump_summary(FILE * out = nullptr) { logger::instance().dump_summary(out); }

}  // namespace dflash_gpu_log

// Convenience macro: only constructs the timer when the logger is on, so
// the disabled path is a single load + branch.
#define DFLASH_GPU_TIMER(name, ctx)                                         \
    auto _dflash_gpu_log_var__ = ::dflash_gpu_log::logger::instance().enabled() \
        ? ::dflash_gpu_log::scope_timer((name), (ctx))                      \
        : ::dflash_gpu_log::scope_timer(nullptr, nullptr)
