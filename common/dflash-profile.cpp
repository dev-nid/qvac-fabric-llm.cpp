// dflash-profile.cpp — see dflash-profile.h header for the design rationale.

#include "dflash-profile.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

}  // namespace dflash_prof
