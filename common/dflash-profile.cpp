// dflash-profile.cpp — see dflash-profile.h header for the design rationale.

#include "dflash-profile.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "llama.h"   // llama_dflash_memory_buckets + breakdown function

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <sstream>
#include <vector>

namespace dflash_prof {

profiler & profiler::instance() {
    static profiler p;
    return p;
}

static std::string make_key(const std::string & phase, const ggml_tensor * t) {
    char buf[256];
    snprintf(buf, sizeof(buf),
             "%s | %-12s | [%5lld,%5lld,%5lld,%5lld] | %s",
             phase.c_str(),
             ggml_op_name(t->op),
             (long long) t->ne[0], (long long) t->ne[1],
             (long long) t->ne[2], (long long) t->ne[3],
             ggml_type_name(t->type));
    return buf;
}

static uint64_t tensor_bytes(const ggml_tensor * t) {
    if (!t) return 0;
    return (uint64_t) ggml_nbytes(t);
}

bool profiler::eval_callback(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * p = static_cast<profiler *>(user_data);
    if (!p || !p->enabled_) {
        return true;
    }

    if (ask) {
        // Build the key now; record start time. Returning `true` here forces
        // the scheduler to compute exactly this single node and synchronize
        // before firing ask=false (see ggml-backend.cpp:1580).
        p->t_start_us_   = ggml_time_us();
        p->current_key_  = make_key(p->current_phase_, t);
        return true;
    }

    // ask=false: node has been computed and the backend synchronized.
    const uint64_t dt_us = (uint64_t) (ggml_time_us() - p->t_start_us_);

    auto & st = p->stats_[p->current_key_];
    st.count    += 1;
    st.total_us += dt_us;
    if (st.bytes_per_call == 0) {
        // Inputs (src[0..]) + output. Constant per shape, so capture once.
        uint64_t b = tensor_bytes(t);
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            b += tensor_bytes(t->src[i]);
        }
        st.bytes_per_call = b;
    }

    p->total_observed_us_ += dt_us;
    return true;
}

void profiler::reset() {
    stats_.clear();
    total_observed_us_ = 0;
    current_phase_.clear();
    current_key_.clear();
    t_start_us_ = 0;
}

void profiler::dump(FILE * out, int top_n) const {
    if (!out) out = stderr;

    if (stats_.empty()) {
        fprintf(out, "[dflash-prof] no ops observed\n");
        return;
    }

    std::vector<std::pair<std::string, op_stats>> sorted(stats_.begin(), stats_.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto & a, const auto & b) {
                  return a.second.total_us > b.second.total_us;
              });

    const int n_dump = top_n <= 0 ? (int) sorted.size()
                                  : std::min(top_n, (int) sorted.size());

    fprintf(out, "\n");
    fprintf(out, "================================================================================\n");
    fprintf(out, "[dflash-prof] per-op breakdown (top %d of %d unique buckets, sorted by total time)\n",
            n_dump, (int) sorted.size());
    fprintf(out, "[dflash-prof] total observed: %.2f ms across %d nodes\n",
            total_observed_us_ / 1e3, (int) sorted.size());
    fprintf(out, "================================================================================\n");
    fprintf(out, "  rank   calls     tot_ms   avg_us    pct  GB/s   bucket (phase | op | shape | type)\n");
    fprintf(out, "  ----   -----     ------   ------    ---  ----   ------\n");

    for (int i = 0; i < n_dump; i++) {
        const auto & key = sorted[i].first;
        const auto & st  = sorted[i].second;

        const double avg_us  = (double) st.total_us / std::max(st.count, (uint64_t) 1);
        const double pct     = total_observed_us_ > 0
                                   ? 100.0 * (double) st.total_us / (double) total_observed_us_
                                   : 0.0;
        const double bw_gbps = avg_us > 0
                                   ? (double) st.bytes_per_call / (avg_us * 1e3)  // bytes / (us * 1e3) = GB/s
                                   : 0.0;

        fprintf(out, "  %4d   %5llu  %9.2f  %7.1f  %4.1f%%  %5.1f  %s\n",
                i + 1,
                (unsigned long long) st.count,
                st.total_us / 1e3,
                avg_us,
                pct,
                bw_gbps,
                key.c_str());
    }
    fprintf(out, "\n");
}

void profiler::dump_phase_summary(FILE * out) const {
    if (!out) out = stderr;

    std::map<std::string, op_stats> by_phase;
    for (const auto & [key, st] : stats_) {
        const auto pipe = key.find(" | ");
        const std::string phase = (pipe == std::string::npos) ? key : key.substr(0, pipe);
        auto & agg = by_phase[phase];
        agg.count    += st.count;
        agg.total_us += st.total_us;
    }

    fprintf(out, "[dflash-prof] phase summary:\n");
    fprintf(out, "  phase                 calls       tot_ms     pct\n");
    for (const auto & [phase, st] : by_phase) {
        const double pct = total_observed_us_ > 0
                               ? 100.0 * (double) st.total_us / (double) total_observed_us_
                               : 0.0;
        fprintf(out, "  %-20s  %6llu  %10.2f  %5.1f%%\n",
                phase.c_str(),
                (unsigned long long) st.count,
                st.total_us / 1e3,
                pct);
    }
    fprintf(out, "\n");
}

void profiler::dump_op_summary(FILE * out) const {
    if (!out) out = stderr;

    std::map<std::string, op_stats> by_op;
    for (const auto & [key, st] : stats_) {
        const auto p1 = key.find(" | ");
        if (p1 == std::string::npos) continue;
        const auto p2 = key.find(" | ", p1 + 3);
        if (p2 == std::string::npos) continue;
        std::string op = key.substr(p1 + 3, p2 - (p1 + 3));
        // Trim trailing spaces from the formatted op column.
        while (!op.empty() && op.back() == ' ') op.pop_back();
        auto & agg = by_op[op];
        agg.count    += st.count;
        agg.total_us += st.total_us;
    }

    std::vector<std::pair<std::string, op_stats>> sorted(by_op.begin(), by_op.end());
    std::sort(sorted.begin(), sorted.end(),
              [](const auto & a, const auto & b) {
                  return a.second.total_us > b.second.total_us;
              });

    fprintf(out, "[dflash-prof] op summary (sorted by total time):\n");
    fprintf(out, "  op                  calls       tot_ms     pct\n");
    for (const auto & [op, st] : sorted) {
        const double pct = total_observed_us_ > 0
                               ? 100.0 * (double) st.total_us / (double) total_observed_us_
                               : 0.0;
        fprintf(out, "  %-18s  %6llu  %10.2f  %5.1f%%\n",
                op.c_str(),
                (unsigned long long) st.count,
                st.total_us / 1e3,
                pct);
    }
    fprintf(out, "\n");
}

prof_env_config read_env_config() {
    prof_env_config cfg;

    const char * v = getenv("DFLASH_PROF");
    cfg.enabled = (v != nullptr) && (atoi(v) != 0);

    const char * vf = getenv("DFLASH_PROF_FILE");
    cfg.out_file = vf ? vf : "";

    const char * vt = getenv("DFLASH_PROF_TOP_N");
    if (vt) {
        const int n = atoi(vt);
        if (n > 0) cfg.top_n = n;
    }

    return cfg;
}

// ===========================================================================
// memory_profiler implementation
// ===========================================================================

memory_profiler & memory_profiler::instance() {
    static memory_profiler p;
    return p;
}

size_t memory_profiler::process_rss_bytes() {
    // Linux: parse VmRSS from /proc/self/status. Returns bytes (kB-in-file × 1024).
    // Returns 0 on non-Linux or if /proc isn't available.
    std::ifstream f("/proc/self/status");
    if (!f.is_open()) return 0;

    std::string line;
    while (std::getline(f, line)) {
        if (line.compare(0, 6, "VmRSS:") == 0) {
            std::istringstream iss(line.substr(6));
            size_t kb = 0;
            std::string unit;
            iss >> kb >> unit;
            return kb * 1024;
        }
    }
    return 0;
}

void memory_profiler::query_vram(size_t * used_out, size_t * free_out) {
    if (used_out) *used_out = 0;
    if (free_out) *free_out = 0;

    const size_t n_devs = ggml_backend_dev_count();
    size_t total_used = 0;
    size_t total_free = 0;

    for (size_t i = 0; i < n_devs; i++) {
        ggml_backend_dev_t dev = ggml_backend_dev_get(i);
        if (dev == nullptr) continue;
        // Only consider GPU devices for VRAM accounting; CPU's "free/total"
        // is just system RAM and overlaps with RSS.
        if (ggml_backend_dev_type(dev) != GGML_BACKEND_DEVICE_TYPE_GPU) continue;
        size_t free = 0, total = 0;
        ggml_backend_dev_memory(dev, &free, &total);
        if (total > free) total_used += (total - free);
        total_free += free;
    }

    if (used_out) *used_out = total_used;
    if (free_out) *free_out = total_free;
}

void memory_profiler::snapshot(const std::string & label,
                               const struct llama_context * draft_ctx,
                               const struct llama_context * target_ctx) {
    mem_snapshot s;
    s.label = label;

    // `llama_dflash_memory_breakdown` was an API from the old
    // feat/dflash-speculative-decoding branch that didn't survive the port
    // to feat/dflash-spec-decoding. The bucket counters in `mem_snapshot`
    // are left at zero; only the process-level RSS / VRAM fields below
    // remain functional. Restoring the per-bucket breakdown is tracked
    // separately (no consumer currently needs it).
    (void) draft_ctx;
    (void) target_ctx;

    s.rss_bytes = process_rss_bytes();
    query_vram(&s.vram_used_bytes, &s.vram_free_bytes);

    snapshots_.push_back(std::move(s));
}

void memory_profiler::reset() {
    snapshots_.clear();
}

static std::string fmt_mib(size_t bytes) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%9.2f MiB", (double) bytes / (1024.0 * 1024.0));
    return buf;
}

void memory_profiler::dump(FILE * out) const {
    if (out == nullptr) out = stderr;

    if (snapshots_.empty()) {
        fprintf(out, "[dflash-mem] no snapshots taken\n");
        return;
    }

    fprintf(out, "\n");
    fprintf(out, "================================================================================\n");
    fprintf(out, "[dflash-mem] DFlash-specific memory snapshots (DFlash buckets only;\n");
    fprintf(out, "             generic llama allocations covered by llama_memory_breakdown_print)\n");
    fprintf(out, "================================================================================\n");

    // Per-snapshot table: label + each bucket as a row.
    // Header row: bucket name then one column per snapshot.
    std::vector<std::pair<std::string, std::vector<size_t>>> rows = {
        {"side_store_K (sum)",         {}},
        {"side_store_V (sum)",         {}},
        {"captured_features",          {}},
        {"capture_staging",            {}},
        {"draft_topk",                 {}},
        {"draft_topk_argmax",          {}},
        {"---DFlash subtotal---",      {}},
        {"process RSS",                {}},
        {"VRAM used (sum across GPUs)",{}},
        {"VRAM free",                  {}},
    };

    for (const auto & s : snapshots_) {
        const size_t dflash_subtotal =
            s.side_store_K_bytes + s.side_store_V_bytes +
            s.captured_features_bytes + s.capture_staging_bytes +
            s.draft_topk_bytes + s.draft_topk_argmax_bytes;

        rows[0].second.push_back(s.side_store_K_bytes);
        rows[1].second.push_back(s.side_store_V_bytes);
        rows[2].second.push_back(s.captured_features_bytes);
        rows[3].second.push_back(s.capture_staging_bytes);
        rows[4].second.push_back(s.draft_topk_bytes);
        rows[5].second.push_back(s.draft_topk_argmax_bytes);
        rows[6].second.push_back(dflash_subtotal);
        rows[7].second.push_back(s.rss_bytes);
        rows[8].second.push_back(s.vram_used_bytes);
        rows[9].second.push_back(s.vram_free_bytes);
    }

    // Column widths: 30 char bucket, then per-snapshot label column.
    fprintf(out, "  %-30s", "bucket");
    for (const auto & s : snapshots_) {
        fprintf(out, "  %16s", s.label.c_str());
    }
    fprintf(out, "\n");

    for (const auto & row : rows) {
        fprintf(out, "  %-30s", row.first.c_str());
        for (size_t v : row.second) {
            fprintf(out, "  %16s", fmt_mib(v).c_str());
        }
        fprintf(out, "\n");
    }

    // Side-store metadata (capacity / filled / n_layers) for the LAST snapshot
    // — useful to interpret the side_store numbers above.
    if (!snapshots_.empty()) {
        const auto & last = snapshots_.back();
        fprintf(out, "\n");
        fprintf(out, "  side-store metadata (last snapshot '%s'):\n", last.label.c_str());
        fprintf(out, "    n_layers      = %d\n",  last.n_layers);
        fprintf(out, "    ctx_capacity  = %lld\n", (long long) last.ctx_capacity);
        fprintf(out, "    ctx_filled    = %lld\n", (long long) last.ctx_filled);
        if (last.ctx_capacity > 0 && last.ctx_filled <= last.ctx_capacity) {
            fprintf(out, "    fill ratio    = %.1f%%\n",
                    100.0 * (double) last.ctx_filled / (double) last.ctx_capacity);
        }
    }

    // Delta from first to last (if multiple snapshots).
    if (snapshots_.size() >= 2) {
        const auto & a = snapshots_.front();
        const auto & z = snapshots_.back();
        fprintf(out, "\n");
        fprintf(out, "  delta '%s' -> '%s':\n", a.label.c_str(), z.label.c_str());
        auto delta = [](size_t before, size_t after) -> std::string {
            char buf[64];
            const long long d = (long long)(after) - (long long)(before);
            snprintf(buf, sizeof(buf), "%+9.2f MiB", d / (1024.0 * 1024.0));
            return buf;
        };
        fprintf(out, "    DFlash subtotal:  %s\n",
                delta(a.side_store_K_bytes + a.side_store_V_bytes +
                      a.captured_features_bytes + a.capture_staging_bytes +
                      a.draft_topk_bytes + a.draft_topk_argmax_bytes,
                      z.side_store_K_bytes + z.side_store_V_bytes +
                      z.captured_features_bytes + z.capture_staging_bytes +
                      z.draft_topk_bytes + z.draft_topk_argmax_bytes).c_str());
        fprintf(out, "    process RSS:      %s\n", delta(a.rss_bytes, z.rss_bytes).c_str());
        fprintf(out, "    VRAM used:        %s\n", delta(a.vram_used_bytes, z.vram_used_bytes).c_str());
    }

    fprintf(out, "\n");
}

mem_env_config read_mem_env_config() {
    mem_env_config cfg;

    const char * v = getenv("DFLASH_MEM");
    cfg.enabled = (v != nullptr) && (atoi(v) != 0);

    const char * vf = getenv("DFLASH_MEM_FILE");
    cfg.out_file = vf ? vf : "";

    return cfg;
}

}  // namespace dflash_prof
