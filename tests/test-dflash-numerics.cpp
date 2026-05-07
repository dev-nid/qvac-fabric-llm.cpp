// tests/test-dflash-numerics.cpp
//
// DFlash drafter graph numerics oracle test. Loads a deterministic input
// (`noise_embedding` + `target_hidden`) from a binary artifact produced by
// `scripts/gen_dflash_oracle.py`, runs ONE forward pass through our C++
// DFlash decoder graph (via libllama), and compares the resulting post-norm
// hidden state against the PyTorch reference's output stored in the same
// artifact. PASSES when the per-position cosine similarity is at least
// `kCosThreshold` AND the max relative L2 deviation is at most `kRelL2Max`.
//
// What this catches that the text-level CTest can't:
// * Per-layer projection / norm / RoPE bugs that don't flip the argmax
//   (the strong-correctness gate only sees text-level differences).
// * Subtle plumbing regressions in the encoder graph (wrong fc_in dim,
//   wrong target_hidden offset into the side store, etc.).
// * Drift introduced by future refactors of the dflash decoder graph.
//
// Two-tier tolerance (chain-mode driver consumes positions [1..bs-1] only;
// position 0 is the "anchor" = the id_last input token whose DFlash
// bidirectional draft prediction equals itself by paper §3.1):
//
//   * Predictor positions [1..bs-1]: cos-sim >= 0.999, rel-L2 <= 0.05.
//     Empirical baseline on Qwen3-4B-DFlash-b16 is cos-sim > 0.9996 and
//     rel-L2 < 0.03 uniformly. The gate has clear signal: a planted
//     regression that scales the final norm output by 1.005 drops cos-sim
//     to ~0.985 at every predictor position (well below threshold), while
//     the natural f16-vs-bf16 baseline drift stays well above.
//
//   * Anchor position 0: cos-sim >= 0.95, rel-L2 <= 0.30 (lenient).
//     Position 0 has a systematic ~0.97-0.99 cos-sim across seeds — the
//     first query position has the largest f16-vs-bf16 drift in our
//     setup (further-from-the-mean intermediate activations after the
//     first attention pass). The chain-mode speculative driver doesn't
//     consume pos 0's logit, so the pos-0 baseline is uninteresting for
//     correctness; the lenient threshold here just catches gross
//     regressions (e.g., a global scale bug would push pos 0 below 0.9
//     too).
//
// We run the PyTorch reference in bf16 (the published checkpoint's
// training dtype) and our C++ in f16 (default GGUF dtype). 5 transformer
// layers + RoPE + attention softmax + per-layer norms accumulate the
// 10-bit vs 7-bit mantissa precision difference predictably.
//
// Skip-on-missing-models convention (mirrors test-dflash-correctness.sh):
// the test EXITS 0 with a yellow warning if any of the env vars
//     DFLASH_TEST_TARGET, DFLASH_TEST_DRAFT, DFLASH_TEST_ORACLE
// are unset or unreadable. CI without the models gets a green skip.
//
// Build: registered via tests/CMakeLists.txt as `test-dflash-numerics`.
// Run (one shell line per env var):
//     DFLASH_TEST_TARGET=/path/to/Qwen3-4B-Q8_0.gguf
//     DFLASH_TEST_DRAFT=/path/to/Qwen3-4B-DFlash-b16.gguf
//     DFLASH_TEST_ORACLE=/path/to/oracle.bin
//     ctest --test-dir build -R test-dflash-numerics --output-on-failure
//
// Generate a fresh oracle artifact:
//     .venv/bin/python3 scripts/gen_dflash_oracle.py
//         --model /home/dev/devnid/dflash/models/Qwen3-4B-DFlash-b16
//         --out   /tmp/dflash_oracle/oracle.bin

#include "llama.h"

#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

// Match scripts/gen_dflash_oracle.py's HEADER_FMT exactly.
struct OracleHeader {
    uint32_t magic;            // 'DORC' = 0x43524F44
    uint32_t version;          // 1
    uint32_t ctx_len;          // cross-attn target context length
    uint32_t block_size;       // drafter intra-block length (16 for b16 ckpt)
    uint32_t hidden;           // per-token hidden dim (2560 for Qwen3-4B-DFlash-b16)
    uint32_t n_target_layers;  // captured target layers (5)
    uint32_t fc_in;            // n_target_layers * hidden (12800)
    uint32_t seed;             // RNG seed
    uint32_t dtype_ref;        // 1 = bf16, 2 = fp32
    uint32_t reserved[7];      // future use
};
static_assert(sizeof(OracleHeader) == 64, "OracleHeader must be 64 bytes");

constexpr uint32_t kOracleMagic   = 0x43524F44;  // 'DORC' little-endian
constexpr uint32_t kOracleVersion = 1;

// PASS thresholds — two-tier; see file header for rationale.
constexpr float kCosThreshold        = 0.999f;  // predictor pos [1..bs-1]: min cos-sim
constexpr float kRelL2Max            = 0.05f;   // predictor pos [1..bs-1]: max rel-L2
constexpr float kCosThresholdAnchor  = 0.95f;   // anchor pos 0: min cos-sim
constexpr float kRelL2MaxAnchor      = 0.30f;   // anchor pos 0: max rel-L2

const char * kClrYellow = "\033[33m";
const char * kClrRed    = "\033[31m";
const char * kClrGreen  = "\033[32m";
const char * kClrReset  = "\033[0m";

void log_warn(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "%sWARN%s ", kClrYellow, kClrReset);
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
}

void log_err(const char * fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    std::fprintf(stderr, "%sERROR%s ", kClrRed, kClrReset);
    std::vfprintf(stderr, fmt, ap);
    std::fprintf(stderr, "\n");
    va_end(ap);
}

const char * env_or_null(const char * name) {
    const char * v = std::getenv(name);
    if (!v || !*v) return nullptr;
    return v;
}

bool file_readable(const char * path) {
    if (!path) return false;
    std::ifstream f(path, std::ios::binary);
    return f.good();
}

struct Oracle {
    OracleHeader       header{};
    std::vector<float> noise;          // [block_size * hidden]
    std::vector<float> target_hidden;  // [ctx_len * fc_in]
    std::vector<float> expected;       // [block_size * hidden]
};

bool load_oracle(const char * path, Oracle & out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) {
        log_err("oracle: cannot open '%s'", path);
        return false;
    }
    f.read(reinterpret_cast<char *>(&out.header), sizeof(out.header));
    if (!f) {
        log_err("oracle: short read on header");
        return false;
    }
    if (out.header.magic != kOracleMagic) {
        log_err("oracle: bad magic 0x%08x (expected 0x%08x = 'DORC')",
                out.header.magic, kOracleMagic);
        return false;
    }
    if (out.header.version != kOracleVersion) {
        log_err("oracle: unsupported version %u (expected %u)",
                out.header.version, kOracleVersion);
        return false;
    }
    if (out.header.block_size == 0 || out.header.hidden == 0 ||
        out.header.ctx_len == 0 || out.header.fc_in == 0) {
        log_err("oracle: zero-sized dims (ctx=%u block=%u hidden=%u fc_in=%u)",
                out.header.ctx_len, out.header.block_size,
                out.header.hidden, out.header.fc_in);
        return false;
    }
    if (out.header.fc_in != out.header.n_target_layers * out.header.hidden) {
        log_err("oracle: fc_in=%u != n_target_layers=%u * hidden=%u",
                out.header.fc_in, out.header.n_target_layers, out.header.hidden);
        return false;
    }

    const size_t n_noise  = (size_t) out.header.block_size * out.header.hidden;
    const size_t n_target = (size_t) out.header.ctx_len    * out.header.fc_in;
    const size_t n_expect = (size_t) out.header.block_size * out.header.hidden;

    out.noise.resize(n_noise);
    out.target_hidden.resize(n_target);
    out.expected.resize(n_expect);

    f.read(reinterpret_cast<char *>(out.noise.data()),         n_noise * sizeof(float));
    f.read(reinterpret_cast<char *>(out.target_hidden.data()), n_target * sizeof(float));
    f.read(reinterpret_cast<char *>(out.expected.data()),      n_expect * sizeof(float));
    if (!f) {
        log_err("oracle: short read on payload");
        return false;
    }

    std::printf("oracle: loaded %s\n", path);
    std::printf("  ctx_len=%u block_size=%u hidden=%u n_target_layers=%u fc_in=%u\n",
                out.header.ctx_len, out.header.block_size, out.header.hidden,
                out.header.n_target_layers, out.header.fc_in);
    std::printf("  ref dtype=%s seed=%u  noise[%zu] target[%zu] expected[%zu]\n",
                out.header.dtype_ref == 1 ? "bf16" :
                out.header.dtype_ref == 2 ? "fp32" : "?",
                out.header.seed, n_noise, n_target, n_expect);
    return true;
}

// Per-position cosine similarity between a row of the actual output and the
// matching row of the reference. Returns 1.0 for identical rows, 0.0 for
// orthogonal, -1.0 for anti-parallel. A small denominator (||ref|| < eps)
// is treated as "no signal" and reported as 1.0 (passes by default).
float row_cosine(const float * actual, const float * ref, int hidden) {
    double dot = 0.0;
    double na2 = 0.0;
    double nr2 = 0.0;
    for (int i = 0; i < hidden; ++i) {
        const double a = actual[i];
        const double r = ref[i];
        dot += a * r;
        na2 += a * a;
        nr2 += r * r;
    }
    if (nr2 < 1e-30 || na2 < 1e-30) {
        return 1.0f;
    }
    return (float) (dot / (std::sqrt(na2) * std::sqrt(nr2)));
}

float row_relative_l2(const float * actual, const float * ref, int hidden) {
    double diff2 = 0.0;
    double ref2  = 0.0;
    for (int i = 0; i < hidden; ++i) {
        const double d = (double) actual[i] - (double) ref[i];
        diff2 += d * d;
        ref2  += (double) ref[i] * (double) ref[i];
    }
    if (ref2 < 1e-30) {
        return 0.0f;
    }
    return (float) std::sqrt(diff2 / ref2);
}

}  // namespace

int main(int argc, char ** argv) {
    (void) argc;
    (void) argv;

    // ---------------- skip-on-missing-config ----------------
    const char * tgt_path    = env_or_null("DFLASH_TEST_TARGET");
    const char * dft_path    = env_or_null("DFLASH_TEST_DRAFT");
    const char * oracle_path = env_or_null("DFLASH_TEST_ORACLE");

    auto skip_with_warn = [](const char * what) {
        log_warn("test-dflash-numerics: %s", what);
        log_warn("test-dflash-numerics: skipping (set DFLASH_TEST_TARGET, "
                 "DFLASH_TEST_DRAFT, DFLASH_TEST_ORACLE to enable). "
                 "Generate the oracle with scripts/gen_dflash_oracle.py.");
        return 0;  // GREEN skip — mirrors test-dflash-correctness convention
    };

    if (!tgt_path)    return skip_with_warn("DFLASH_TEST_TARGET unset");
    if (!dft_path)    return skip_with_warn("DFLASH_TEST_DRAFT unset");
    if (!oracle_path) return skip_with_warn("DFLASH_TEST_ORACLE unset");
    if (!file_readable(tgt_path))    return skip_with_warn("DFLASH_TEST_TARGET unreadable");
    if (!file_readable(dft_path))    return skip_with_warn("DFLASH_TEST_DRAFT unreadable");
    if (!file_readable(oracle_path)) return skip_with_warn("DFLASH_TEST_ORACLE unreadable");

    // ---------------- load oracle ----------------
    Oracle ora;
    if (!load_oracle(oracle_path, ora)) {
        return 1;
    }

    // ---------------- libllama setup ----------------
    llama_backend_init();

    auto cleanup = [&]() { llama_backend_free(); };

    // Load target (only used for binding tok_embd / lm_head into the draft,
    // per paper §4.2). Quick-load: no offload, no GPU layers needed for the
    // numerics test (CPU is fine; the oracle is small).
    llama_model_params mparams_tgt = llama_model_default_params();
    mparams_tgt.n_gpu_layers = 99;  // use whatever the runtime supports; the
                                    // weights themselves don't enter the
                                    // compare, only the bound tensor pointers
                                    // for the draft graph build.
    llama_model * model_tgt = llama_model_load_from_file(tgt_path, mparams_tgt);
    if (!model_tgt) {
        log_err("failed to load target model from '%s'", tgt_path);
        cleanup();
        return 1;
    }

    // Load draft model (DFlash arch).
    llama_model_params mparams_dft = llama_model_default_params();
    mparams_dft.n_gpu_layers = 99;
    llama_model * model_dft = llama_model_load_from_file(dft_path, mparams_dft);
    if (!model_dft) {
        log_err("failed to load draft model from '%s'", dft_path);
        llama_model_free(model_tgt);
        cleanup();
        return 1;
    }

    // Paper §4.2: bind target's tok_embd / lm_head into the draft. Required
    // before draft context creation (graph_reserve runs at construction).
    if (!llama_dflash_bind_target(model_dft, model_tgt)) {
        log_err("llama_dflash_bind_target failed (draft is not LLM_ARCH_DFLASH "
                "or one of the models is null)");
        llama_model_free(model_dft);
        llama_model_free(model_tgt);
        cleanup();
        return 1;
    }

    // Sanity-check the dflash hparams against the oracle.
    {
        const uint32_t bs    = llama_model_dflash_block_size(model_dft);
        const int      n_tlid = llama_model_dflash_n_target_layer_ids(model_dft);
        if ((uint32_t) bs != ora.header.block_size) {
            log_err("draft block_size=%u != oracle block_size=%u", bs, ora.header.block_size);
            llama_model_free(model_dft);
            llama_model_free(model_tgt);
            cleanup();
            return 1;
        }
        if ((uint32_t) n_tlid != ora.header.n_target_layers) {
            log_err("draft n_target_layer_ids=%d != oracle n_target_layers=%u",
                    n_tlid, ora.header.n_target_layers);
            llama_model_free(model_dft);
            llama_model_free(model_tgt);
            cleanup();
            return 1;
        }
        const int n_embd_dft = llama_model_n_embd(model_dft);
        if ((uint32_t) n_embd_dft != ora.header.hidden) {
            log_err("draft n_embd=%d != oracle hidden=%u", n_embd_dft, ora.header.hidden);
            llama_model_free(model_dft);
            llama_model_free(model_tgt);
            cleanup();
            return 1;
        }
    }

    // Create the draft context. embeddings=true so llama_get_embeddings()
    // returns the post-norm hidden state (= res->t_embd) after decode. We
    // also need the side store sized big enough for ctx_len target features.
    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx          = std::max<uint32_t>(1024, ora.header.ctx_len + ora.header.block_size + 32);
    cparams.n_batch        = ora.header.block_size;
    cparams.n_ubatch       = ora.header.block_size;  // single-ubatch for the verify-shaped decode
    cparams.embeddings     = true;
    cparams.dflash_max_ctx = std::max<int32_t>((int32_t) ora.header.ctx_len + 64, 256);
    cparams.dflash_topk    = 1;  // standard chain-mode kernel; we don't read topk here

    llama_context * ctx_dft = llama_init_from_model(model_dft, cparams);
    if (!ctx_dft) {
        log_err("failed to create draft context");
        llama_model_free(model_dft);
        llama_model_free(model_tgt);
        cleanup();
        return 1;
    }

    // ---------------- populate the side store ----------------
    // The PyTorch reference passes target_hidden directly into the model's
    // forward (post-fc input). Our C++ encoder graph
    // (`llm_build_dflash_encode`) takes target_hidden and writes
    // pre-projected K/V into the side store. The two paths are
    // mathematically equivalent under the same target_hidden input, modulo
    // f16 vs bf16 precision through the per-layer wk/wv + k_norm + RoPE on
    // the C++ side (vs the equivalent ops inside the layer on the PyTorch
    // side). The cumulative drift is captured by kCosThreshold.
    {
        const int rc = llama_dflash_extend(
            ctx_dft, ora.target_hidden.data(),
            (int64_t) ora.header.ctx_len, /*pos_start=*/0);
        if (rc != 0) {
            log_err("llama_dflash_extend failed (rc=%d)", rc);
            llama_free(ctx_dft);
            llama_model_free(model_dft);
            llama_model_free(model_tgt);
            cleanup();
            return 1;
        }
    }

    // ---------------- decode the noise embedding ----------------
    // We use the embd field (skip tok_embed lookup) so the C++ input matches
    // the PyTorch reference's `noise_embedding` argument exactly. Positions
    // are [ctx_len .. ctx_len + block_size - 1] to mirror the q-positions in
    // the reference's RoPE (which receives the FULL [0..ctx_len+block_size-1]
    // range and slices the last block_size for q).
    const int     bs         = (int) ora.header.block_size;
    const int     n_embd     = (int) ora.header.hidden;
    const int64_t pos_offset = (int64_t) ora.header.ctx_len;

    // llama_batch_init pre-allocates: token (or embd), pos, n_seq_id, and
    // for each token slot a seq_id array of length n_seq_max. We populate
    // the existing storage rather than overwriting pointers (overwriting
    // would leak the allocated arrays AND double-free at llama_batch_free).
    llama_batch batch = llama_batch_init(/*n_tokens=*/bs, /*embd=*/n_embd, /*n_seq_max=*/1);
    batch.n_tokens = bs;
    std::memcpy(batch.embd, ora.noise.data(), (size_t) bs * n_embd * sizeof(float));

    for (int i = 0; i < bs; ++i) {
        batch.pos[i]        = (llama_pos) (pos_offset + i);
        batch.n_seq_id[i]   = 1;
        batch.seq_id[i][0]  = 0;
        batch.logits[i]     = 1;  // request output (embeddings) at every position
    }

    if (llama_decode(ctx_dft, batch) != 0) {
        log_err("llama_decode failed on the draft");
        llama_batch_free(batch);
        llama_free(ctx_dft);
        llama_model_free(model_dft);
        llama_model_free(model_tgt);
        cleanup();
        return 1;
    }

    // ---------------- read back + compare ----------------
    const float * embd = llama_get_embeddings(ctx_dft);
    if (!embd) {
        log_err("llama_get_embeddings returned nullptr (cparams.embeddings was true)");
        llama_batch_free(batch);
        llama_free(ctx_dft);
        llama_model_free(model_dft);
        llama_model_free(model_tgt);
        cleanup();
        return 1;
    }

    int n_fail              = 0;
    float min_cos_predictor = 1.0f;
    float max_rel_l2_predictor = 0.0f;
    int   worst_pos_predictor  = -1;
    std::printf("\nper-position numerics:\n");
    std::printf("  pos   role        cos-sim    rel-l2    threshold              verdict\n");
    for (int p = 0; p < bs; ++p) {
        const float * actual_row = embd + (size_t) p * n_embd;
        const float * ref_row    = ora.expected.data() + (size_t) p * n_embd;
        const float cos = row_cosine(actual_row, ref_row, n_embd);
        const float rl2 = row_relative_l2(actual_row, ref_row, n_embd);

        const bool  is_anchor = (p == 0);
        const float cos_thr   = is_anchor ? kCosThresholdAnchor : kCosThreshold;
        const float rl2_thr   = is_anchor ? kRelL2MaxAnchor     : kRelL2Max;
        const bool  ok        = (cos >= cos_thr) && (rl2 <= rl2_thr);

        if (!is_anchor) {
            if (cos < min_cos_predictor)    { min_cos_predictor = cos; worst_pos_predictor = p; }
            if (rl2 > max_rel_l2_predictor) { max_rel_l2_predictor = rl2; }
        }
        if (!ok) ++n_fail;
        std::printf("  %3d   %-9s   %8.6f   %7.5f   cos>=%.3f rl2<=%.2f   %s\n",
                    p, is_anchor ? "anchor" : "predictor",
                    cos, rl2, cos_thr, rl2_thr,
                    ok ? "PASS" : (std::string(kClrRed) + "FAIL" + kClrReset).c_str());
    }
    std::printf("\nsummary:\n");
    std::printf("  positions             : %d\n", bs);
    std::printf("  failed positions      : %d\n", n_fail);
    std::printf("  predictor min cos-sim : %.6f  (at pos %d; threshold %.6f)\n",
                min_cos_predictor, worst_pos_predictor, kCosThreshold);
    std::printf("  predictor max rel-l2  : %.6f  (threshold %.6f)\n",
                max_rel_l2_predictor, kRelL2Max);

    llama_batch_free(batch);
    llama_free(ctx_dft);
    llama_model_free(model_dft);
    llama_model_free(model_tgt);
    cleanup();

    if (n_fail > 0) {
        std::printf("\n%sFAIL%s test-dflash-numerics: %d/%d positions below threshold\n",
                    kClrRed, kClrReset, n_fail, bs);
        return 1;
    }
    std::printf("\n%sPASS%s test-dflash-numerics: %d/%d positions within thresholds\n",
                kClrGreen, kClrReset, bs, bs);
    return 0;
}
