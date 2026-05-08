// dflash-gpu-log.cpp — see dflash-gpu-log.h for design notes.

#include "dflash-gpu-log.h"

#include "llama.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace dflash_gpu_log {

namespace {

uint64_t now_ns() {
    using clock = std::chrono::steady_clock;
    return (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(
        clock::now().time_since_epoch()).count();
}

double ms(uint64_t ns) { return ns / 1e6; }

}  // namespace

logger & logger::instance() {
    static logger l;
    return l;
}

void logger::add_sample(const std::string & name, uint64_t ns) {
    auto & s = stats_[name];
    ++s.count;
    s.total_ns += ns;
    if (ns < s.min_ns) s.min_ns = ns;
    if (ns > s.max_ns) s.max_ns = ns;
}

void logger::reset() {
    stats_.clear();
    n_extend_calls_            = 0;
    n_slide_left_calls_        = 0;
    n_capture_transpose_calls_ = 0;
    n_alt_remap_calls_         = 0;
    d2h_bytes_                 = 0;
    h2d_bytes_                 = 0;
}

void logger::dump_summary(FILE * out) const {
    if (!enabled_) return;
    if (!out) out = stderr;

    std::vector<std::pair<std::string, phase_stats>> rows(stats_.begin(), stats_.end());
    std::sort(rows.begin(), rows.end(),
              [](const auto & a, const auto & b) { return a.second.total_ns > b.second.total_ns; });

    uint64_t grand_total_ns = 0;
    for (const auto & r : rows) grand_total_ns += r.second.total_ns;

    std::fprintf(out, "\n");
    std::fprintf(out, "DFLASH_GPU_LOG summary (host-observed wall time, sync-anchored):\n");
    std::fprintf(out, "  %-36s %8s %12s %10s %10s %10s %6s\n",
                 "phase", "calls", "total_ms", "avg_ms", "min_ms", "max_ms", "pct");
    std::fprintf(out, "  %s\n", std::string(36 + 8 + 12 + 10 + 10 + 10 + 6 + 6, '-').c_str());
    for (const auto & r : rows) {
        const auto & s = r.second;
        const double pct = grand_total_ns ? 100.0 * (double) s.total_ns / (double) grand_total_ns : 0.0;
        std::fprintf(out, "  %-36s %8llu %12.2f %10.2f %10.2f %10.2f %5.1f%%\n",
                     r.first.c_str(),
                     (unsigned long long) s.count,
                     ms(s.total_ns),
                     ms(s.count ? s.total_ns / s.count : 0),
                     ms(s.min_ns == UINT64_MAX ? 0 : s.min_ns),
                     ms(s.max_ns),
                     pct);
    }
    std::fprintf(out, "  %s\n", std::string(36 + 8 + 12 + 10 + 10 + 10 + 6 + 6, '-').c_str());
    std::fprintf(out, "  %-36s %8s %12.2f\n",
                 "(sum of measured phases)", "", ms(grand_total_ns));

    std::fprintf(out, "\n");
    std::fprintf(out, "DFLASH_GPU_LOG counters:\n");
    std::fprintf(out, "  dflash_extend calls       = %llu\n", (unsigned long long) n_extend_calls_);
    std::fprintf(out, "  slide_left evictions      = %llu\n", (unsigned long long) n_slide_left_calls_);
    std::fprintf(out, "  capture host transposes   = %llu\n", (unsigned long long) n_capture_transpose_calls_);
    std::fprintf(out, "  alt-accept remaps         = %llu\n", (unsigned long long) n_alt_remap_calls_);
    std::fprintf(out, "  device->host transfer     = %.2f MiB\n",
                 (double) d2h_bytes_ / (1024.0 * 1024.0));
    std::fprintf(out, "  host->device transfer     = %.2f MiB\n",
                 (double) h2d_bytes_ / (1024.0 * 1024.0));
}

scope_timer::scope_timer(const char * name, llama_context * ctx)
    : name_(name), ctx_(ctx), t_start_ns_(0), active_(name != nullptr) {
    if (active_) t_start_ns_ = now_ns();
}

scope_timer::~scope_timer() {
    if (!active_) return;
    if (ctx_) llama_synchronize(ctx_);
    const uint64_t t_end_ns = now_ns();
    logger::instance().add_sample(name_, t_end_ns - t_start_ns_);
}

env_config read_env_config() {
    env_config cfg;
    if (const char * e = std::getenv("DFLASH_GPU_LOG")) {
        cfg.enabled = (e[0] != '\0' && std::strcmp(e, "0") != 0);
    }
    if (const char * f = std::getenv("DFLASH_GPU_LOG_FILE")) {
        cfg.out_file = f;
    }
    return cfg;
}

}  // namespace dflash_gpu_log
