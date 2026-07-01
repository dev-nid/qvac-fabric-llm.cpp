// qvac: standalone Qwen3-VL vision encoder benchmark.
//
// Runs ONLY the vision encoder (ViT + patch-merge projection) of a Qwen3-VL
// mmproj GGUF through ggml, with no dependency on libllama / the SSM-LLM half
// of fabric. This gives a seconds-scale edit -> build -> run loop for
// optimizing the Vulkan / OpenCL kernels shared with production clip.cpp,
// plus a cosine-similarity check so every kernel change stays lossless.
//
// Build target: qwen3vl-encoder  (see tools/mtmd/CMakeLists.txt)
//
// Usage:
//   qwen3vl-encoder --mmproj <mmproj.gguf> --image <img.jpg>
//                   [--backend cpu|vulkan|opencl|gpu|all]
//                   [--device <ggml-dev-name>]
//                   [--threads N] [--iters N] [--list-devices]
//                   [--dump <out.bin>]  [--ref <ref.bin>] [--cos-min 0.9999]
//
//   --backend cpu      : CPU only
//   --backend vulkan   : the GPU device whose name contains "Vulkan"
//   --backend opencl   : the GPU device whose name contains "OpenCL"
//   --backend gpu      : every GPU device found
//   --backend all      : CPU + every GPU device found  (used on-device: the
//                        Pixel emits a Vulkan line, the S25 emits an OpenCL line)
//   --device <name>    : force one specific ggml backend device by exact name
//   --dump  <out.bin>  : (single-backend runs) write the raw float32 embedding
//   --ref   <ref.bin>  : (single-backend runs) cosine-sim vs a reference; exit
//                        non-zero if below --cos-min (default 0.9999)

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "clip.h"
#include "clip-impl.h"   // clip_image_u8/f32 structs, clip_get_projector_type
#include "clip-model.h"  // clip_hparams, clip_get_hparams
#include "mtmd-image.h"  // mtmd_image_preprocessor_dyn_size

#include "ggml.h"
#include "ggml-backend.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <utility>
#include <vector>

static void print_usage(const char * prog) {
    fprintf(stderr,
        "usage: %s --mmproj <mmproj.gguf> --image <img.jpg>\n"
        "          [--backend cpu|vulkan|opencl|gpu|all] [--device <name>]\n"
        "          [--threads N] [--iters N] [--list-devices]\n"
        "          [--dump <out.bin>] [--ref <ref.bin>] [--cos-min <f>]\n",
        prog);
}

struct options {
    std::string mmproj;
    std::string image;
    std::string backend = "cpu";
    std::string device;   // explicit ggml device name (optional)
    std::string dump;
    std::string ref;
    int   threads = 4;
    int   iters   = 10;
    float cos_min = 0.9999f;
    bool  list_devices = false;
};

static std::string to_lower(std::string s) {
    for (auto & c : s) c = (char) std::tolower((unsigned char) c);
    return s;
}

// mobile GPUs (Mali, Adreno) and desktop iGPUs report IGPU (unified memory),
// discrete cards report GPU. Treat both as "a GPU backend to try".
static bool is_gpu_like(ggml_backend_dev_t d) {
    int t = (int) ggml_backend_dev_type(d);
    return t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU;
}

static bool parse_args(int argc, char ** argv, options & o) {
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&](const char * name) -> const char * {
            if (i + 1 >= argc) { fprintf(stderr, "missing value for %s\n", name); return nullptr; }
            return argv[++i];
        };
        if (a == "--mmproj")            { auto v = next("--mmproj");  if (!v) return false; o.mmproj  = v; }
        else if (a == "--image")        { auto v = next("--image");   if (!v) return false; o.image   = v; }
        else if (a == "--backend")      { auto v = next("--backend"); if (!v) return false; o.backend = v; }
        else if (a == "--device")       { auto v = next("--device");  if (!v) return false; o.device  = v; }
        else if (a == "--dump")         { auto v = next("--dump");    if (!v) return false; o.dump    = v; }
        else if (a == "--ref")          { auto v = next("--ref");     if (!v) return false; o.ref     = v; }
        else if (a == "--threads")      { auto v = next("--threads"); if (!v) return false; o.threads = atoi(v); }
        else if (a == "--iters")        { auto v = next("--iters");   if (!v) return false; o.iters   = atoi(v); }
        else if (a == "--cos-min")      { auto v = next("--cos-min"); if (!v) return false; o.cos_min = (float) atof(v); }
        else if (a == "--list-devices") { o.list_devices = true; }
        else if (a == "-h" || a == "--help") { print_usage(argv[0]); exit(0); }
        else { fprintf(stderr, "unknown arg: %s\n", a.c_str()); return false; }
    }
    return true;
}

// load an image file (jpg/png/...) into an RGB clip_image_u8 via stb_image.
static bool load_image_u8(const std::string & path, clip_image_u8 * out) {
    int nx = 0, ny = 0, nc = 0;
    unsigned char * data = stbi_load(path.c_str(), &nx, &ny, &nc, 3);
    if (!data) {
        fprintf(stderr, "error: failed to load image '%s': %s\n", path.c_str(), stbi_failure_reason());
        return false;
    }
    clip_build_img_from_pixels(data, nx, ny, out);
    stbi_image_free(data);
    return true;
}

static double cosine_similarity(const std::vector<float> & a, const std::vector<float> & b) {
    if (a.size() != b.size() || a.empty()) return -2.0;
    double dot = 0.0, na = 0.0, nb = 0.0;
    for (size_t i = 0; i < a.size(); i++) {
        dot += (double) a[i] * (double) b[i];
        na  += (double) a[i] * (double) a[i];
        nb  += (double) b[i] * (double) b[i];
    }
    if (na == 0.0 || nb == 0.0) return -2.0;
    return dot / (std::sqrt(na) * std::sqrt(nb));
}

// one backend configuration to run
struct run_cfg {
    std::string label;    // human label for the [ENC_RESULT] line
    bool        use_gpu;
    std::string device;   // exact ggml device name, or "" for auto
};

static void list_devices() {
    size_t n = ggml_backend_dev_count();
    fprintf(stdout, "[enc] %zu ggml backend device(s):\n", n);
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        const char * name = ggml_backend_dev_name(d);
        const char * desc = ggml_backend_dev_description(d);
        int type = (int) ggml_backend_dev_type(d);
        fprintf(stdout, "  [%zu] name='%s' type=%d desc='%s'\n", i, name ? name : "?", type, desc ? desc : "?");
    }
    fflush(stdout);
}

// find first GPU device whose (lowercased) name contains `key`; "" if none.
static std::string find_gpu_device(const std::string & key) {
    std::string k = to_lower(key);
    size_t n = ggml_backend_dev_count();
    for (size_t i = 0; i < n; i++) {
        ggml_backend_dev_t d = ggml_backend_dev_get(i);
        if (!is_gpu_like(d)) continue;
        std::string name = to_lower(ggml_backend_dev_name(d) ? ggml_backend_dev_name(d) : "");
        if (name.find(k) != std::string::npos) return ggml_backend_dev_name(d);
    }
    return "";
}

// run a single backend config end-to-end; returns process-style rc (0 ok).
// on success the final embedding is written to out_vec (for cross-backend
// accuracy comparison by the caller).
static int run_encode(const options & o, const run_cfg & cfg, const clip_image_u8 & u8,
                      std::vector<float> & out_vec, bool allow_dump_ref) {
    clip_context_params cp{};
    cp.use_gpu           = cfg.use_gpu;
    cp.flash_attn_type   = CLIP_FLASH_ATTN_TYPE_AUTO;
    cp.image_min_tokens  = -1;
    cp.image_max_tokens  = -1;
    cp.warmup            = true;
    cp.has_bf16_weights  = false;
    cp.cb_eval           = nullptr;
    cp.cb_eval_user_data = nullptr;
    cp.backend_device    = cfg.device.empty() ? nullptr : cfg.device.c_str();

    fprintf(stderr, "[enc] === config '%s' (use_gpu=%d device=%s) ===\n",
            cfg.label.c_str(), (int) cfg.use_gpu, cfg.device.empty() ? "(auto)" : cfg.device.c_str());

    clip_init_result init = clip_init(o.mmproj.c_str(), cp);
    clip_ctx * ctx = init.ctx_v;
    if (!ctx) {
        fprintf(stderr, "[enc] SKIP '%s': clip_init failed / backend unavailable\n", cfg.label.c_str());
        return 1;
    }

    clip_image_f32_batch batch;
    {
        mtmd_image_preprocessor_dyn_size pp(ctx);
        if (!pp.preprocess(u8, batch)) {
            fprintf(stderr, "[enc] SKIP '%s': preprocess failed\n", cfg.label.c_str());
            clip_free(ctx);
            return 1;
        }
    }

    const int n_embd = clip_n_mmproj_embd(ctx);
    size_t total_tokens = 0;
    for (auto & e : batch.entries) total_tokens += (size_t) clip_n_output_tokens(ctx, e.get());
    out_vec.assign(total_tokens * (size_t) n_embd, 0.0f);
    std::vector<float> & vec = out_vec;

    for (int i = 0; i < 3; i++) { // warmup
        if (!clip_image_batch_encode(ctx, o.threads, &batch, vec.data())) {
            fprintf(stderr, "[enc] FAIL '%s': encode failed (warmup)\n", cfg.label.c_str());
            clip_free(ctx);
            return 1;
        }
    }

    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < o.iters; i++) {
        if (!clip_image_batch_encode(ctx, o.threads, &batch, vec.data())) {
            fprintf(stderr, "[enc] FAIL '%s': encode failed (iter %d)\n", cfg.label.c_str(), i);
            clip_free(ctx);
            return 1;
        }
    }
    const int64_t t1 = ggml_time_us();
    const double avg_ms = (t1 - t0) / 1000.0 / o.iters;

    fprintf(stdout,
        "[ENC_RESULT] label=%s device=%s tokens=%zu embd=%d iters=%d avg_ms=%.3f\n",
        cfg.label.c_str(), cfg.device.empty() ? "(auto)" : cfg.device.c_str(),
        total_tokens, n_embd, o.iters, avg_ms);
    fflush(stdout);

    int rc = 0;
    if (allow_dump_ref && !o.dump.empty()) {
        FILE * f = fopen(o.dump.c_str(), "wb");
        if (f) { fwrite(vec.data(), sizeof(float), vec.size(), f); fclose(f);
            fprintf(stderr, "[enc] dumped %zu floats to %s\n", vec.size(), o.dump.c_str()); }
        else { fprintf(stderr, "error: cannot open dump file %s\n", o.dump.c_str()); rc = 1; }
    }
    if (allow_dump_ref && !o.ref.empty()) {
        FILE * f = fopen(o.ref.c_str(), "rb");
        if (!f) { fprintf(stderr, "error: cannot open ref file %s\n", o.ref.c_str()); rc = 1; }
        else {
            std::vector<float> ref(vec.size());
            size_t got = fread(ref.data(), sizeof(float), ref.size(), f);
            fclose(f);
            if (got != ref.size()) {
                fprintf(stderr, "error: ref size mismatch (got %zu, expected %zu)\n", got, ref.size());
                rc = 1;
            } else {
                double cos = cosine_similarity(vec, ref);
                fprintf(stdout, "[ENC_COSSIM] label=%s cos=%.8f min=%.8f %s\n",
                        cfg.label.c_str(), cos, (double) o.cos_min, cos >= o.cos_min ? "PASS" : "FAIL");
                fflush(stdout);
                if (cos < o.cos_min) rc = 1;
            }
        }
    }

    clip_free(ctx);
    return rc;
}

// CPU tuning sweep: for each flash-attn setting (needs re-init) loop a set of
// thread counts (per-encode, no re-init) so one on-device run yields the whole
// matrix. Also reports a compute-only steady-state floor per flash setting via
// the in-graph re-run (isolates graph-build/alloc overhead from raw compute).
static int run_cpu_sweep(const options & o, const clip_image_u8 & u8) {
    const int thread_list[] = { 2, 3, 4, 6 };
    struct flash_opt { const char * name; clip_flash_attn_type t; };
    const flash_opt flash_opts[] = {
        { "faAuto", CLIP_FLASH_ATTN_TYPE_AUTO },
        { "faOff",  CLIP_FLASH_ATTN_TYPE_DISABLED },
    };

    int ran = 0;
    for (const auto & fo : flash_opts) {
        clip_context_params cp{};
        cp.use_gpu = false;
        cp.flash_attn_type = fo.t;
        cp.image_min_tokens = -1;
        cp.image_max_tokens = -1;
        cp.warmup = true;
        cp.has_bf16_weights = false;
        cp.cb_eval = nullptr;
        cp.cb_eval_user_data = nullptr;
        cp.backend_device = nullptr;

        fprintf(stderr, "[enc] === cpu-sweep flash=%s ===\n", fo.name);
        clip_init_result init = clip_init(o.mmproj.c_str(), cp);
        clip_ctx * ctx = init.ctx_v;
        if (!ctx) { fprintf(stderr, "[enc] sweep: clip_init failed (flash=%s)\n", fo.name); continue; }

        clip_image_f32_batch batch;
        {
            mtmd_image_preprocessor_dyn_size pp(ctx);
            if (!pp.preprocess(u8, batch)) { fprintf(stderr, "[enc] sweep: preprocess failed\n"); clip_free(ctx); continue; }
        }
        const int n_embd = clip_n_mmproj_embd(ctx);
        size_t total_tokens = 0;
        for (auto & e : batch.entries) total_tokens += (size_t) clip_n_output_tokens(ctx, e.get());
        std::vector<float> vec(total_tokens * (size_t) n_embd);

        for (int th : thread_list) {
            for (int i = 0; i < 2; i++) clip_image_batch_encode(ctx, th, &batch, vec.data()); // warmup
            const int64_t t0 = ggml_time_us();
            for (int i = 0; i < o.iters; i++) clip_image_batch_encode(ctx, th, &batch, vec.data());
            const int64_t t1 = ggml_time_us();
            const double avg = (t1 - t0) / 1000.0 / o.iters;
            fprintf(stdout, "[ENC_RESULT] label=cpu-%s-t%d tokens=%zu embd=%d iters=%d avg_ms=%.3f\n",
                    fo.name, th, total_tokens, n_embd, o.iters, avg);
            fflush(stdout);
            ran++;
        }
        clip_free(ctx);
    }
    fprintf(stdout, "[ENC_DONE] cpu-sweep ran=%d rc=%d\n", ran, ran > 0 ? 0 : 1);
    fflush(stdout);
    return ran > 0 ? 0 : 1;
}

// ---- per-op CPU profiler via the scheduler eval callback ----
// The callback brackets each graph node (ask=true just before, ask=false just
// after it computes on CPU), so elapsed time attributes to that node's op.
struct prof_state {
    std::map<std::string, double> op_ms;   // total ms per op type
    std::map<std::string, long>   op_cnt;  // node count per op type
    int64_t t0 = 0;
    bool    counting = false;              // ignore warmup
};

static bool prof_eval_cb(struct ggml_tensor * t, bool ask, void * ud) {
    prof_state * p = static_cast<prof_state *>(ud);
    if (ask) {
        p->t0 = ggml_time_us();
    } else if (p->counting) {
        const double dt_ms = (ggml_time_us() - p->t0) / 1000.0;
        const char * name = ggml_op_name(t->op);
        p->op_ms[name]  += dt_ms;
        p->op_cnt[name] += 1;
    }
    return true; // observe every node
}

static int run_cpu_prof(const options & o, const clip_image_u8 & u8) {
    prof_state prof;

    clip_context_params cp{};
    cp.use_gpu = false;
    cp.flash_attn_type = CLIP_FLASH_ATTN_TYPE_AUTO;
    cp.image_min_tokens = -1;
    cp.image_max_tokens = -1;
    cp.warmup = true;
    cp.has_bf16_weights = false;
    cp.cb_eval = prof_eval_cb;
    cp.cb_eval_user_data = &prof;
    cp.backend_device = nullptr;

    clip_init_result init = clip_init(o.mmproj.c_str(), cp);
    clip_ctx * ctx = init.ctx_v;
    if (!ctx) { fprintf(stderr, "[enc] prof: clip_init failed\n"); return 1; }

    clip_image_f32_batch batch;
    {
        mtmd_image_preprocessor_dyn_size pp(ctx);
        if (!pp.preprocess(u8, batch)) { fprintf(stderr, "[enc] prof: preprocess failed\n"); clip_free(ctx); return 1; }
    }
    const int n_embd = clip_n_mmproj_embd(ctx);
    size_t total_tokens = 0;
    for (auto & e : batch.entries) total_tokens += (size_t) clip_n_output_tokens(ctx, e.get());
    std::vector<float> vec(total_tokens * (size_t) n_embd);

    const int threads = o.threads > 0 ? o.threads : 4;
    for (int i = 0; i < 2; i++) clip_image_batch_encode(ctx, threads, &batch, vec.data()); // warmup (not counted)

    prof.counting = true;
    const int iters = o.iters > 0 ? o.iters : 6;
    const int64_t t0 = ggml_time_us();
    for (int i = 0; i < iters; i++) clip_image_batch_encode(ctx, threads, &batch, vec.data());
    const int64_t t1 = ggml_time_us();
    prof.counting = false;
    const double total_ms = (t1 - t0) / 1000.0 / iters;

    fprintf(stdout, "[ENC_RESULT] label=cpu-prof-t%d tokens=%zu embd=%d iters=%d avg_ms=%.3f\n",
            threads, total_tokens, n_embd, iters, total_ms);

    // sort ops by total time desc and print the breakdown (per-encode ms)
    std::vector<std::pair<std::string, double>> items(prof.op_ms.begin(), prof.op_ms.end());
    std::sort(items.begin(), items.end(),
              [](const std::pair<std::string,double>&a, const std::pair<std::string,double>&b){ return a.second > b.second; });
    for (const auto & it : items) {
        const double per_encode = it.second / iters;
        const double pct = 100.0 * it.second / ((t1 - t0) / 1000.0);
        fprintf(stdout, "[ENC_PROF] op=%-12s ms=%8.2f pct=%5.1f nodes/enc=%ld\n",
                it.first.c_str(), per_encode, pct, prof.op_cnt[it.first] / iters);
    }
    fprintf(stdout, "[ENC_DONE] cpu-prof rc=0\n");
    fflush(stdout);
    clip_free(ctx);
    return 0;
}

int main(int argc, char ** argv) {
    options o;
    if (!parse_args(argc, argv, o)) { print_usage(argv[0]); return 1; }

    if (o.list_devices) { list_devices(); return 0; }

    if (o.mmproj.empty() || o.image.empty()) {
        fprintf(stderr, "error: --mmproj and --image are required\n");
        print_usage(argv[0]);
        return 1;
    }

    fprintf(stderr, "[enc] mmproj=%s backend=%s threads=%d iters=%d\n",
            o.mmproj.c_str(), o.backend.c_str(), o.threads, o.iters);
    list_devices();

    // load the image once (backend-independent u8 RGB).
    clip_image_u8 * u8 = clip_image_u8_init();
    if (!load_image_u8(o.image, u8)) { clip_image_u8_free(u8); return 1; }

    // CPU tuning sweep (thread x flash matrix) short-circuits the normal path.
    if (o.backend == "cpu-sweep") {
        int rc = run_cpu_sweep(o, *u8);
        clip_image_u8_free(u8);
        return rc;
    }
    // Per-op CPU profiler (attributes encode time by ggml op).
    if (o.backend == "cpu-prof") {
        int rc = run_cpu_prof(o, *u8);
        clip_image_u8_free(u8);
        return rc;
    }

    // build the list of backend configs to run.
    std::vector<run_cfg> cfgs;
    if (!o.device.empty()) {
        cfgs.push_back({ "device:" + o.device, true, o.device });
    } else if (o.backend == "cpu") {
        cfgs.push_back({ "cpu", false, "" });
    } else if (o.backend == "vulkan") {
        cfgs.push_back({ "vulkan", true, find_gpu_device("vulkan") });
    } else if (o.backend == "opencl") {
        cfgs.push_back({ "opencl", true, find_gpu_device("opencl") });
    } else if (o.backend == "gpu" || o.backend == "all") {
        if (o.backend == "all") cfgs.push_back({ "cpu", false, "" });
        size_t n = ggml_backend_dev_count();
        for (size_t i = 0; i < n; i++) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            if (!is_gpu_like(d)) continue;
            const char * name = ggml_backend_dev_name(d);
            cfgs.push_back({ std::string("gpu:") + (name ? name : "?"), true, name ? name : "" });
        }
    } else {
        fprintf(stderr, "error: unknown --backend '%s'\n", o.backend.c_str());
        clip_image_u8_free(u8);
        return 1;
    }

    // dump/ref only make sense for a single-config run.
    const bool allow_dump_ref = (cfgs.size() == 1);

    int rc = 0;
    int ran = 0;
    // In multi-backend mode we use the CPU run as the accuracy reference and
    // report cosine similarity for every GPU backend against it — this is the
    // on-device losslessness check (token count alone does NOT prove the GPU
    // embedding is correct).
    std::vector<float> cpu_ref;
    bool have_ref = false;
    for (const auto & cfg : cfgs) {
        std::vector<float> vec;
        int r = run_encode(o, cfg, *u8, vec, allow_dump_ref);
        if (r == 0) ran++;

        if (r == 0 && cfgs.size() > 1) {
            if (!cfg.use_gpu && !have_ref) {
                cpu_ref = vec;          // CPU is the trusted reference
                have_ref = true;
            } else if (cfg.use_gpu && have_ref) {
                // Cross-backend (GPU vs CPU) tolerates normal fp16-accumulate
                // divergence (~0.995); a garbage/broken backend scores ~0. Use a
                // looser bar than the same-backend regression threshold, but
                // always print the exact cos so the real accuracy is visible.
                const double cross_min = 0.99;
                double cos = cosine_similarity(vec, cpu_ref);
                bool pass = cos >= cross_min;
                fprintf(stdout, "[ENC_COSSIM] label=%s cos=%.8f min=%.8f %s (vs cpu)\n",
                        cfg.label.c_str(), cos, cross_min, pass ? "PASS" : "FAIL");
                fflush(stdout);
                if (!pass) rc = 1;      // a GPU backend that truly diverges is a failure
            }
        }

        // in multi-backend mode, a skipped/failed backend is non-fatal.
        if (cfgs.size() == 1) rc = r;
    }
    if (cfgs.size() > 1 && ran == 0) rc = 1; // nothing ran at all
    if (cfgs.size() > 1 && !have_ref) {
        fprintf(stderr, "[enc] WARN: no CPU reference ran; GPU accuracy not checked\n");
    }

    clip_image_u8_free(u8);
    fprintf(stdout, "[ENC_DONE] ran=%d/%zu rc=%d\n", ran, cfgs.size(), rc);
    fflush(stdout);
    return rc;
}
