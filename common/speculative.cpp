#include "speculative.h"

#include "common.h"
#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "ngram-cache.h"
#include "ngram-map.h"
#include "ngram-mod.h"
#include "nvtx_helper.h"
#include "sampling.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <map>
#include <queue>
#include <cinttypes>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// DFlash spec-round profiling. Opt-in via DFLASH_PROFILE=1 env var.
// Accumulates per-round timings across one process lifetime; printed by
// common_dflash_prof_print() at end of main(). Wall-time only — no GPU
// sync, so the numbers reflect host-side observed durations (which is
// what matters for driver-overhead analysis on a workload that's GPU-bound
// anyway: the host blocks on llama_decode until the kernels return).
struct common_dflash_prof {
    int64_t encoder_ns       = 0; // extend_side_store() — encoder graph exec (or inline-fast-path advance)
    int64_t draft_decode_ns  = 0; // llama_decode(ctx_dft, ...) — draft block forward
    int64_t draft_other_ns   = 0; // draft()-internal bookkeeping (batch fill, top-k read, KV reset)
    int     n_rounds         = 0;
};
static common_dflash_prof g_dflash_prof;

static inline bool common_dflash_prof_enabled() {
    static const bool enabled = []() {
        const char * s = std::getenv("DFLASH_PROFILE");
        return s != nullptr && std::atoi(s) != 0;
    }();
    return enabled;
}

void common_dflash_prof_print() {
    if (!common_dflash_prof_enabled() || g_dflash_prof.n_rounds == 0) {
        return;
    }
    const double n = (double) g_dflash_prof.n_rounds;
    fprintf(stderr,
            "\n=== DFlash profile (n_rounds=%d) ===\n"
            "  encoder       (avg/round) : %8.3f ms  (total %.1f ms)\n"
            "  draft_decode  (avg/round) : %8.3f ms  (total %.1f ms)\n"
            "  draft_other   (avg/round) : %8.3f ms  (total %.1f ms)\n"
            "  spec_loop_in_speculative.cpp / round = encoder + draft_decode + draft_other\n",
            g_dflash_prof.n_rounds,
            g_dflash_prof.encoder_ns      / 1e6 / n, g_dflash_prof.encoder_ns      / 1e6,
            g_dflash_prof.draft_decode_ns / 1e6 / n, g_dflash_prof.draft_decode_ns / 1e6,
            g_dflash_prof.draft_other_ns  / 1e6 / n, g_dflash_prof.draft_other_ns  / 1e6);
}

const std::vector<enum common_speculative_type> common_speculative_types = {
    COMMON_SPECULATIVE_TYPE_NONE,
    COMMON_SPECULATIVE_TYPE_DRAFT,
    COMMON_SPECULATIVE_TYPE_EAGLE3,
    COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K,
    COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V,
    COMMON_SPECULATIVE_TYPE_NGRAM_MOD,
    COMMON_SPECULATIVE_TYPE_NGRAM_CACHE,
    COMMON_SPECULATIVE_TYPE_DFLASH
};

const std::map<std::string, enum common_speculative_type> common_speculative_type_from_name_map = {
    {"none",          COMMON_SPECULATIVE_TYPE_NONE},
    {"draft",         COMMON_SPECULATIVE_TYPE_DRAFT},
    {"eagle3",        COMMON_SPECULATIVE_TYPE_EAGLE3},
    {"ngram_simple",  COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE},
    {"ngram_map_k",   COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K},
    {"ngram_map_k4v", COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V},
    {"ngram_mod",     COMMON_SPECULATIVE_TYPE_NGRAM_MOD},
    {"ngram_cache",   COMMON_SPECULATIVE_TYPE_NGRAM_CACHE},
    {"dflash",        COMMON_SPECULATIVE_TYPE_DFLASH}
};

struct common_speculative_config {
    common_speculative_type type;
    common_params_speculative params;

    common_speculative_config(common_speculative_type t,
            const common_params_speculative & p = common_params_speculative{}) : type(t), params(p) {}
};

static bool common_speculative_are_compatible(
    const llama_model * model_tgt,
    const llama_model * model_dft) {
    const llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_WRN("%s: draft model vocab type must match target model to use speculation but "
                "vocab_type_dft = %d while vocab_type_tgt = %d\n", __func__, vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        (llama_vocab_get_add_bos(vocab_tgt) && llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft))) {
        LOG_WRN("%s: draft model bos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_bos(vocab_tgt), llama_vocab_get_add_bos(vocab_dft),
                llama_vocab_bos(vocab_tgt), llama_vocab_bos(vocab_dft));
        return false;
    }

    if (llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        (llama_vocab_get_add_eos(vocab_tgt) && llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft))) {
        LOG_WRN("%s: draft model eos tokens must match target model to use speculation. add: %d - %d, id: %d - %d)\n",
                __func__,
                llama_vocab_get_add_eos(vocab_tgt), llama_vocab_get_add_eos(vocab_dft),
                llama_vocab_eos(vocab_tgt), llama_vocab_eos(vocab_dft));
        return false;
    }

    {
        const int n_vocab_tgt = llama_vocab_n_tokens(vocab_tgt);
        const int n_vocab_dft = llama_vocab_n_tokens(vocab_dft);
        const int vocab_diff  = n_vocab_tgt > n_vocab_dft
            ? n_vocab_tgt - n_vocab_dft
            : n_vocab_dft - n_vocab_tgt;

        if (vocab_diff > SPEC_VOCAB_MAX_SIZE_DIFFERENCE) {
            LOG_DBG("%s: draft model vocab must closely match target model to use speculation but ", __func__);
            LOG_DBG("target vocab size %d does not match draft vocab size %d - difference %d, max allowed %d\n",
                    n_vocab_tgt, llama_vocab_n_tokens(vocab_dft), vocab_diff, SPEC_VOCAB_MAX_SIZE_DIFFERENCE);
            return false;
        }

        for (int i = SPEC_VOCAB_CHECK_START_TOKEN_ID; i < std::min(n_vocab_tgt, n_vocab_dft); ++i) {
            const char * token_text_tgt = llama_vocab_get_text(vocab_tgt, i);
            const char * token_text_dft = llama_vocab_get_text(vocab_dft, i);

            if (std::strcmp(token_text_tgt, token_text_dft) != 0) {
                LOG_DBG("%s: draft model vocab must match target model to use speculation but ", __func__);
                LOG_DBG("token %d content differs - target '%s', draft '%s'\n", i,
                        common_token_to_piece(vocab_tgt, i).c_str(),
                        common_token_to_piece(vocab_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

// state of an implementation of speculative decoding
//
// each implementation has a unique type and a state that is implementation-specific
// in a subclass of common_speculative_state
struct common_speculative_state {
    const enum common_speculative_type type;

    size_t n_call_begin  = 0; // number of times this implementation was called for refresh.
    size_t n_call_draft  = 0; // number of times this implementation was called for generation.
    size_t n_call_accept = 0; // number of times this implementation was called for accumulation.

    size_t n_gen_drafts = 0; // number of times a draft or part was generated by this implementation.
    size_t n_acc_drafts = 0; // number of times a draft or part was accepted by the target model.
    size_t n_gen_tokens = 0; // number of tokens generated by this implementation.
    size_t n_acc_tokens = 0; // number of tokens accepted by the target model.

    // TODO: track performance of most recent calls
    const bool gen_perf = true; // whether to generate performance stats.

    int64_t t_begin_us  = 0; // total time spent in refresh of this implementation in microseconds.
    int64_t t_draft_us  = 0; // total time spent in generating drafts in this implementation in microseconds.
    int64_t t_accept_us = 0; // total time spent in accumulation of this implementation in microseconds.

    common_speculative_state(enum common_speculative_type type) : type(type) {}

    virtual ~common_speculative_state() = default;

    virtual void begin(const llama_tokens & prompt) = 0;

    virtual void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) = 0;

    // tree-shaped draft; default impl returns an empty tree (chain-only states)
    virtual void draft_tree(
            const common_params_speculative & /*params*/,
            const llama_tokens & /*prompt_tgt*/,
            llama_token /*id_last*/,
            common_speculative_tree & result) {
        result = {};
    }

    // alt-accept fast-path hint; default impl is a no-op (DRAFT speculative
    // state has no captures buffer to remap)
    virtual void record_alt_accept(int /*alt_capture_idx*/, int /*alt_depth*/) {}

    virtual void accept(uint16_t n_accepted) = 0;

    virtual int32_t n_max(const common_params_speculative & params) const = 0;
    virtual int32_t n_min(const common_params_speculative & params) const = 0;
};

struct common_speculative_checkpoint {
    llama_pos pos_min  = 0;
    llama_pos pos_max  = 0;

    int64_t   n_tokens = 0;

    std::vector<uint8_t> data;

    size_t size() const {
        return data.size();
    }
};

struct common_speculative_state_draft : public common_speculative_state {
    llama_context * ctx_tgt; // only used for retokenizing from ctx_dft
    llama_context * ctx_dft;

    bool use_ckpt = false;
    common_speculative_checkpoint ckpt;

    common_sampler * smpl;

    llama_batch  batch;
    llama_tokens prompt_dft;

    bool vocab_cmpt = true; // whether retokenization is needed
    std::unordered_map<std::string, std::string> vocab_map;

    common_speculative_state_draft(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            const std::vector<std::pair<std::string, std::string>> & replacements,
            bool use_ckpt)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
        , use_ckpt(use_ckpt)
    {
        batch = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);
        smpl = nullptr;

        // TODO: optimize or pass from outside?
        // {
        //     common_params_sampling params;
        //     params.no_perf = false;
        //
        //     params.top_k = 40;
        //     params.top_p = 0.9;
        //
        //     params.samplers = {
        //         COMMON_SAMPLER_TYPE_TOP_K,
        //         COMMON_SAMPLER_TYPE_TOP_P,
        //         COMMON_SAMPLER_TYPE_INFILL,
        //     };
        //
        //     result->smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        // }
        {
            common_params_sampling params;
            params.no_perf = false;
            params.top_k = 10;
            params.samplers = {
                COMMON_SAMPLER_TYPE_TOP_K,
            };

            smpl = common_sampler_init(llama_get_model(ctx_dft), params);
        }

        vocab_cmpt = common_speculative_are_compatible(llama_get_model(ctx_tgt), llama_get_model(ctx_dft));
        LOG_DBG("vocab_cmpt = %d\n", vocab_cmpt);

        if (!vocab_cmpt) {
            LOG_WRN("the target and draft vocabs are not compatible - tokens will be translated between the two\n");

            for (const auto & pair : replacements) {
                vocab_map[pair.first] = pair.second;
            }
        }
    }

    ~common_speculative_state_draft() override {
        llama_perf_context_print(ctx_dft);

        llama_free(ctx_dft);

        common_sampler_free(smpl);

        llama_batch_free(batch);
    }

    void begin(const llama_tokens & /*prompt*/) override {
    }

    size_t create_checkpoint(int n_tokens_prompt) {
        int slot_id = 0;
        const size_t checkpoint_size = llama_state_seq_get_size_ext(ctx_dft, slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);

        ckpt.pos_min  = llama_memory_seq_pos_min(llama_get_memory(ctx_dft), slot_id);
        ckpt.pos_max  = llama_memory_seq_pos_max(llama_get_memory(ctx_dft), slot_id);
        ckpt.n_tokens = n_tokens_prompt;
        ckpt.data.resize(checkpoint_size);

        const size_t n = llama_state_seq_get_data_ext(ctx_dft, ckpt.data.data(), checkpoint_size, slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (n != checkpoint_size) {
            GGML_ABORT("checkpoint size mismatch: expected %zu, got %zu\n", checkpoint_size, n);
        }

        LOG_DBG("%s: pos_min = %d, pos_max = %d, size = %.3f MiB\n", __func__,
                ckpt.pos_min, ckpt.pos_max, (float) ckpt.data.size() / 1024 / 1024);
        return n;
    }

    size_t restore_checkpoint() {
        int slot_id = 0;
        LOG_DBG("%s: pos_min = %d, pos_max = %d\n", __func__, ckpt.pos_min, ckpt.pos_max);
        const size_t n = llama_state_seq_set_data_ext(ctx_dft, ckpt.data.data(), ckpt.size(), slot_id, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY | LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
        if (n != ckpt.size()) {
            GGML_ABORT("%s: failed to restore context checkpoint (pos_min=%d, pos_max=%d, size=%zu",
                        __func__, ckpt.pos_min, ckpt.pos_max, ckpt.size());
        }
        llama_memory_seq_rm(llama_get_memory(ctx_dft), slot_id, ckpt.pos_max + 1, -1);

        return n;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        const auto & sparams = params.draft;

        auto * spec = this;

        auto & batch      = spec->batch;
        auto & ctx_tgt    = spec->ctx_tgt;
        auto & ctx_dft    = spec->ctx_dft;
        auto & smpl       = spec->smpl;
        auto & prompt_dft = spec->prompt_dft;

        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0; // index of part to be reused in prompt_dft
        int reuse_n = 0; // length of part to be reused in prompt_dft

        const int n_ctx = llama_n_ctx(ctx_dft) - sparams.n_max;

        llama_tokens prompt_cnv;
        if (!spec->vocab_cmpt) {
            std::string text;

            text = common_detokenize(ctx_tgt, prompt_tgt, true);
            text = replace_to_dft(text);

            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());

            prompt_cnv = common_tokenize(ctx_dft, text, false, true);

            // convert id_last to draft vocab. llama_detokenize is called directly to avoid an allocation
            const auto * model_tgt = llama_get_model(ctx_tgt);
            const auto * vocab_tgt = llama_model_get_vocab(model_tgt);

            int32_t n_chars = llama_detokenize(vocab_tgt, &id_last, 1, nullptr, 0, false, false);
            GGML_ASSERT(n_chars < 0 && "failed to detokenize id_last");

            text.resize(-n_chars);
            llama_detokenize(vocab_tgt, &id_last, 1, text.data(), text.size(), false, false);
            text = replace_to_dft(text);

            LOG_DBG("main->draft detokenized id_last(%d): '%s'\n", id_last, text.c_str());
            id_last = common_tokenize(ctx_dft, text, false, true)[0];
        }

        const llama_tokens & prompt_cur = spec->vocab_cmpt ? prompt_tgt : prompt_cnv;

        const int i_start = std::max<int>(0, (int) prompt_cur.size() - n_ctx);

        if (use_ckpt && i_start > 0) {
            LOG_WRN("%s: context shift is not supported with checkpoint-based contexts - skipping\n", __func__);
            return;
        }

        // reuse as much as possible from the old draft context
        // ideally, the draft context should be as big as the target context and we will always reuse the entire prompt
        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_cur.size() &&
                   i       + cur < (int) prompt_dft.size() &&
                   prompt_cur[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= 256 || n_ctx >= (int) prompt_cur.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }

            if (use_ckpt) {
                break;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, #prompt_dft = %zu, #prompt_cur = %zu\n",
                __func__, reuse_i, reuse_n, prompt_dft.size(), prompt_cur.size());
        if (use_ckpt && ckpt.n_tokens > reuse_n) {
            LOG_DBG("%s: checkpoint (n_tokens = %d) is outdated -> delete it\n", __func__, (int) ckpt.n_tokens);

            reuse_i = 0;
            reuse_n = 0;

            ckpt = {};
        }

        result.clear();
        result.reserve(sparams.n_max);

        if (reuse_n == 0 || (use_ckpt && reuse_i > 0)) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // this happens when a previous draft has been discarded (for example, due to being too small), but the
            // target model agreed with it. in this case, we simply pass back the previous results to save compute
            if (reuse_i + reuse_n < (int64_t) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);

                    if (sparams.n_max <= (int) result.size()) {
                        break;
                    }
                }

                return;
            }

            if (reuse_i > 0) {
                GGML_ASSERT(!use_ckpt);

                bool is_removed = llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                if (!is_removed) {
                    LOG_ERR("%s: llama_memory_seq_rm failed, reuse_i=%d\n", __func__, reuse_i);
                    return;
                }
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);

                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                if (use_ckpt) {
                    if (ckpt.n_tokens > 0) {
                        LOG_DBG("%s: restoring checkpoint, reuse_n=%d, prompt_dft.size=%zu\n", __func__, reuse_n, prompt_dft.size());
                        restore_checkpoint();
                        reuse_n = ckpt.n_tokens;
                        prompt_dft.resize(reuse_n);
                    }
                } else {
                    const bool is_removed = llama_memory_seq_rm(mem_dft, 0, reuse_n, -1);
                    if (!is_removed) {
                        LOG_ERR("%s: llama_memory_seq_rm failed, reuse_n=%d, prompt_dft.size=%zu\n", __func__, reuse_n, prompt_dft.size());
                        return;
                    }
                    prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
                }
            }
        }

        // prepare a batch to evaluate any new tokens in the prompt
        common_batch_clear(batch);

        for (size_t i = i_start + reuse_n; i < prompt_cur.size(); ++i) {
            //LOG_DBG("i = %d, i_start = %d, reuse_n = %d, i - i_start = %d, id = %6d\n", i, i_start, reuse_n, i - i_start, prompt_cur[i]);
            common_batch_add(batch, prompt_cur[i], i - i_start, { 0 }, false);

            prompt_dft.push_back(prompt_cur[i]);
        }

        // we should rarely end-up here during normal decoding
        if (batch.n_tokens > 0) {
            //LOG_DBG("%s: draft prompt batch: %s\n", __func__, string_from(ctx, batch).c_str());
            LOG_DBG("%s: draft prompt batch: %d tokens\n", __func__, batch.n_tokens);

            int ret = llama_decode(ctx_dft, batch);
            if (ret != 0 && ret != 1) {
                LOG_WRN("%s: llama_decode returned %d, prompt_cur.size=%zu\n",
                        __func__, ret, prompt_cur.size());
            }

            if (use_ckpt) {
                create_checkpoint(prompt_dft.size());
            }
        }

        const llama_pos n_past = prompt_dft.size();

        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);

        prompt_dft.push_back(id_last);

        //LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        int ret = llama_decode(ctx_dft, batch);
        if (ret != 0 && ret != 1) {
            LOG_WRN("%s: llama_decode returned %d, prompt_cur.size=%zu, prompt_dft.size=%zu\n",
                    __func__, ret, prompt_cur.size(), prompt_dft.size());
        }

        common_sampler_reset(smpl);

        // sample n_draft tokens from the draft model
        for (int i = 0; i < sparams.n_max; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);

            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            // add drafted token for each sequence
            const llama_token id = cur_p->data[0].id;

            common_sampler_accept(smpl, id, true);

            // only collect very high-confidence draft tokens
            if (cur_p->data[0].p < sparams.p_min) {
                break;
            }

            result.push_back(id);

            if (sparams.n_max <= (int) result.size()) {
                break;
            }

            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);

            // evaluate the drafted tokens on the draft model
            ret = llama_decode(ctx_dft, batch);
            if (ret != 0) {
                LOG_WRN("%s: llama_decode[%d] returned %d, prompt_cur.size=%zu, prompt_dft.size=%zu\n",
                        __func__, i, ret, prompt_cur.size(), prompt_dft.size());
            }

            prompt_dft.push_back(id);
        }

        if (!spec->vocab_cmpt) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t) sparams.n_max) {
                result.resize(sparams.n_max);
            }
        }

        if (result.size() < (size_t) sparams.n_min) {
            result.clear();
        }
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.draft.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.draft.n_min;
    }

    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.first);
            while (pos != std::string::npos) {
                result.replace(pos, pair.first.length(), pair.second);
                pos = result.find(pair.first, pos + pair.second.length());
            }
        }

        return result;
    }

    std::string replace_to_tgt(const std::string & input) const {
        std::string result = input;

        for (const auto & pair : this->vocab_map) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }

        return result;
    }
};

struct common_speculative_state_eagle3 : public common_speculative_state {
    common_speculative_state_eagle3(enum common_speculative_type type) : common_speculative_state(type) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & draft_tokens) override {
        // TODO: implement
        GGML_UNUSED(params);
        GGML_UNUSED(prompt_tgt);
        GGML_UNUSED(id_last);
        GGML_UNUSED(draft_tokens);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.draft.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.draft.n_min;
    }
};

// state of self-speculation (simple implementation, not ngram-map)
struct common_speculative_state_ngram_simple : public common_speculative_state {
    common_ngram_simple_config config;

    common_speculative_state_ngram_simple(
            enum common_speculative_type type,
            common_ngram_simple_config config)
        : common_speculative_state(type), config(config) {}

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {

        result = common_ngram_simple_draft(config, prompt_tgt, id_last);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        // noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return config.size_mgram;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return config.size_mgram;
    }
};

struct common_speculative_state_ngram_map_k : public common_speculative_state {
    // draft ngram map for speculative decoding without draft model
    common_ngram_map config;

    common_speculative_state_ngram_map_k(
            enum common_speculative_type type,
            common_ngram_map config)
        : common_speculative_state(type), config(std::move(config)) {}

    void begin(const llama_tokens & prompt) override {
        common_ngram_map_begin(config, prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        common_ngram_map_draft(config, prompt_tgt, id_last, result);
        GGML_UNUSED(params);
    }

    void accept(uint16_t n_accepted) override {
        common_ngram_map_accept(config, n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return config.size_value;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return config.size_value;
    }
};

struct common_speculative_state_ngram_mod : public common_speculative_state {
    common_ngram_mod & mod;

    // the last position in the prompt that was added to the ngram container
    size_t i_last = 0;

    // length of the last drafted n‑gram (number of tokens returned by draft)
    size_t n_draft_last = 0;

    // consecutive accept rounds with low acceptance fraction (< 0.5)
    int n_low = 0;

    // enable trace logging if LLAMA_TRACE is set
    const bool verbose;

    common_speculative_state_ngram_mod(enum common_speculative_type type, common_ngram_mod & mod)
        : common_speculative_state(type), mod(mod), verbose(std::getenv("LLAMA_TRACE") != nullptr) {
        static_assert(sizeof(llama_token) == sizeof(common_ngram_mod::entry_t));
    }

    void begin(const llama_tokens & prompt) override {
        i_last = 0;

        n_draft_last = 0;

        const size_t n = mod.get_n();

        if (prompt.size() < n) {
            return;
        }

        for (size_t i = 0; i < prompt.size() - n; ++i) {
            mod.add(prompt.data() + i);
        }

        i_last = prompt.size() - n;

        const double f = (double)mod.get_used() / (double)mod.size();
        LOG_INF("%s: ngram_mod occupancy = %zu/%zu (%.2f)\n", __func__, mod.get_used(), mod.size(), f);

        constexpr double f_thold = 0.25;
        if (f > f_thold) {
            LOG_WRN("%s: ngram_mod occupancy %.2f exceeds threshold (%.2f) - resetting\n", __func__, f, f_thold);

            mod.reset();
        }
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        const auto & sparams = params.ngram_mod;

        n_draft_last = 0;

        const size_t cur_len = prompt_tgt.size();
        if (cur_len < mod.get_n()) {
            return;
        }

        const size_t n = mod.get_n();

        // add new ngrams in chunks
        if (i_last + 32 < cur_len) {
            for (size_t i = i_last; i < cur_len - n; ++i) {
                mod.add(prompt_tgt.data() + i);
            }

            i_last = cur_len - n;
        }

        result.resize(n + sparams.n_max);
        for (size_t i = 0; i < n - 1; ++i) {
            result[i] = prompt_tgt[cur_len - n + 1 + i];
        }
        result[n - 1] = id_last;

        for (int i = 0; i < sparams.n_max; ++i) {
            const llama_token token = mod.get(result.data() + i);
            if (token == common_ngram_mod::EMPTY) {
                if (i < sparams.n_min) {
                    result.clear();
                    return;
                }

                result.resize(n + i);
                break;
            }
            result[n + i] = token;
        }

        // only return the m tokens that were drafted
        for (size_t i = 0; n + i < result.size(); ++i) {
            result[i] = result[n + i];
        }
        result.resize(result.size() - n);

        // store length of drafted n‑gram for later acceptance analysis
        n_draft_last = result.size();
    }

    void accept(uint16_t n_accepted) override {
        // compute acceptance fraction if we have a recorded draft length
        if (n_draft_last > 0) {
            const double f_acc = (double)n_accepted / (double)n_draft_last;
            if (f_acc < 0.5) {
                n_low++;
                if (n_low >= 3) {
                    if (verbose) {
                        LOG_WRN("%s: low acceptance streak (%d) – resetting ngram_mod\n", __func__, n_low);
                    }

                    mod.reset();
                    n_low = 0;
                    i_last = 0;
                }
            } else {
                n_low = 0;
            }
        }
    }

    int32_t n_max(const common_params_speculative & params) const override {
        return params.ngram_mod.n_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.ngram_mod.n_min;
    }
};

struct common_speculative_state_ngram_cache : public common_speculative_state {
    uint16_t n_draft;
    bool save_dynamic;
    bool save_static;

    common_ngram_cache ngram_cache_context;
    common_ngram_cache ngram_cache_dynamic;
    common_ngram_cache ngram_cache_static;

    size_t cache_size = 0; // number of tokens in n-gram cache

    common_speculative_state_ngram_cache(
            const enum common_speculative_type type,
            const std::string & path_static,
            const std::string & path_dynamic,
            uint16_t            n_draft,
            bool                save_dynamic,
            bool                save_static)
        : common_speculative_state(type)
        , n_draft(n_draft)
        , save_dynamic(save_dynamic)
        , save_static(save_static)
    {
        if (!path_static.empty()) {
            try {
                ngram_cache_static = common_ngram_cache_load(path_static);
            } catch (...) {
                LOG_ERR("failed to open static lookup cache: %s", path_static.c_str());
                GGML_ABORT("Couldn't read static lookup cache");
            }
        }

        if (!path_dynamic.empty()) {
            try {
                ngram_cache_dynamic = common_ngram_cache_load(path_dynamic);
            } catch (...) {
                LOG_ERR("failed to open dynamic lookup cache: %s", path_dynamic.c_str());
                GGML_ABORT("Couldn't read dynamic lookup cache");
            }
        }
    }

    void begin(const llama_tokens & prompt) override {
        GGML_UNUSED(prompt);
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        GGML_UNUSED(params);

        if (cache_size < prompt_tgt.size() + 1) {
            llama_tokens tokens_new;
            tokens_new.reserve(prompt_tgt.size() + 1 - cache_size);
            for (size_t j = cache_size; j < prompt_tgt.size(); ++j) {
                tokens_new.push_back(prompt_tgt[j]);
            }
            tokens_new.push_back(id_last); // add the last token

            // Update context ngram cache with new prompt_tgt:
            common_ngram_cache_update(ngram_cache_context, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                    tokens_new, tokens_new.size(), false);
            cache_size = prompt_tgt.size() + 1;
        }

        llama_tokens inp;
        inp.reserve(prompt_tgt.size() + 1);
        for (size_t j = 0; j < prompt_tgt.size(); ++j) {
            inp.push_back(prompt_tgt[j]);
        }
        inp.push_back(id_last);

        result.push_back(id_last);

        common_ngram_cache_draft(inp, result, n_draft, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX,
                ngram_cache_context,
                ngram_cache_dynamic,
                ngram_cache_static);

        if (result.size() > 0) {
            // delete first token in result (which is the id_last token)
            result.erase(result.begin());
        }
    }

    void accept(uint16_t n_accepted) override {
        // TODO: noop
        GGML_UNUSED(n_accepted);
    }

    int32_t n_max(const common_params_speculative & /*params*/) const override {
        return n_draft;
    }

    int32_t n_min(const common_params_speculative & /*params*/) const override {
        return 0;
    }
};

// =============================================================================
// DFlash speculative-decoding state
// =============================================================================
//
// Reads block_size, mask_token_id, target_layer_ids from the draft GGUF;
// installs hidden-state capture on the target context; runs a single
// DFlash decoder forward per draft step and reads back the per-position
// top-K argmax tokens via llama_dflash_get_draft_topk().
//
// Invariants:
//   * The draft context must be created with cparams.dflash_topk set
//     (the in-graph top-K layout depends on it).
//   * The target's tok_embd / lm_head are bound into the draft model
//     in `common_speculative_init` (before the draft context is created),
//     so the DFlash decoder graph can read them via model.target_*.
//   * Capture is installed in this state's ctor and detached in its dtor;
//     the target context's capture state is owned by this state for its
//     lifetime.
struct common_speculative_state_dflash : public common_speculative_state {
    llama_context * ctx_tgt;
    llama_context * ctx_dft;

    llama_batch batch_tgt {};
    llama_batch batch_dft {};

    uint32_t              block_size_default = 0;
    llama_token           mask_token_id      = LLAMA_TOKEN_NULL;
    int64_t               n_features         = 0;
    int64_t               n_embd_target      = 0;
    std::vector<int32_t>  target_layer_ids;

    // Number of tokens whose target hidden states have been pushed through
    // the DFlash encoder into the draft's per-layer K/V side store.
    int64_t               n_committed_total  = 0;

    // alt-accept fast path: when the driver's tree-verify accept walk descends
    // into an alt
    // branch, it calls record_alt_accept(alt_capture_idx, alt_depth)
    // INSTEAD OF redecoding `[id_last, m_1, ..., m_{d-1}, alt_token]` in
    // seq 0. The fields below stash the hint; the next draft_tree →
    // extend_side_store consumes them by remapping the captures buffer
    // offsets. -1 = no hint pending (linear chain-prefix path applies).
    int                   pending_alt_capture_idx = -1;
    int                   pending_alt_depth       = 0;

    // Scratch buffer used by extend_side_store to materialise the
    // remapped captures rows for an alt-accept extend.
    std::vector<float>    alt_remap_buf;

    // scratch buffers for best-first tree expansion: per-position top-K
    // log-probs and token IDs computed host-side from the draft's full-vocab
    // logits, sized to `block_size * topk_K`
    // and reused across draft_tree() calls.
    std::vector<float>    best_first_logprobs;
    std::vector<int32_t>  best_first_tokens;

    // when true, the inline encoder runs on ctx_tgt's scheduler via
    // llama_dflash_inline_encode_from_ctx() instead of on ctx_dft's via
    // llama_dflash_extend_from_ctx(). Set from params.dflash_inline_encoder
    // at construction time; immutable for the life of this state.
    bool                  dflash_inline_encoder = false;

    common_speculative_state_dflash(
            enum common_speculative_type type,
            llama_context * ctx_tgt,
            llama_context * ctx_dft,
            bool            dflash_inline_encoder_in)
        : common_speculative_state(type)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
        , dflash_inline_encoder(dflash_inline_encoder_in)
    {
        batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
        batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

        const llama_model * model_dft = llama_get_model(ctx_dft);
        const llama_model * model_tgt = llama_get_model(ctx_tgt);

        block_size_default = llama_model_dflash_block_size(model_dft);
        mask_token_id      = llama_model_dflash_mask_token_id(model_dft);
        n_embd_target      = llama_model_n_embd(model_tgt);

        const int n_tlid = llama_model_dflash_n_target_layer_ids(model_dft);
        target_layer_ids.reserve(n_tlid > 0 ? n_tlid : 0);
        for (int i = 0; i < n_tlid; ++i) {
            target_layer_ids.push_back(llama_model_dflash_target_layer_id(model_dft, i));
        }
        n_features = (int64_t) target_layer_ids.size() * n_embd_target;

        // Defensive: bind target tensors into draft (idempotent — also done
        // in common_speculative_init before draft context creation).
        llama_dflash_bind_target(const_cast<llama_model *>(model_dft), model_tgt);

        // Install per-layer hidden-state capture on the target context so
        // every decode tees out the layers the draft was trained against.
        if (!target_layer_ids.empty()) {
            llama_dflash_set_capture(ctx_tgt,
                                     target_layer_ids.data(),
                                     target_layer_ids.size(),
                                     n_embd_target);
        }

        // enable the device-to-device capture path. The target skips the
        // per-decode D2H of captured_features; this consumer pulls captures
        // straight from the target's device-resident packed tensor via
        // llama_dflash_extend_from_ctx(). The rare alt-remap path
        // (record_alt_accept) forces a one-shot D2H readback inline.
        if (!target_layer_ids.empty()) {
            llama_dflash_set_skip_host_readback(ctx_tgt, true);
        }

        LOG_INF("%s: dflash spec initialised: block_size=%u, mask_token=%d, "
                "n_features=%lld, target_layer_ids=[",
                __func__, block_size_default, (int) mask_token_id,
                (long long) n_features);
        for (size_t i = 0; i < target_layer_ids.size(); ++i) {
            LOG_INF("%s%d", i == 0 ? "" : ",", (int) target_layer_ids[i]);
        }
        LOG_INF("]\n");
    }

    ~common_speculative_state_dflash() override {
        if (ctx_tgt != nullptr) {
            llama_dflash_set_capture(ctx_tgt, nullptr, 0, 0);
        }
        llama_perf_context_print(ctx_dft);
        llama_batch_free(batch_tgt);
        llama_batch_free(batch_dft);
        llama_free(ctx_dft);
    }

    // begin() runs at start of each new generation.
    //
    // Fast path: when `common_speculative_init` ran before the caller's
    // prefill, DFlash capture is already installed on ctx_tgt. Provided
    // the prefill requested logits at every prompt position, the
    // captured features for the entire prompt are already in
    // `dflash.captured_features`; push them straight into the side store
    // without re-decoding.
    //
    // Slow path: callers that did not install capture before prefill
    // still get the re-decode path as a fallback.
    void begin(const llama_tokens & prompt) override {
        llama_dflash_reset_ctx_kv(ctx_dft);
        n_committed_total       = 0;
        pending_alt_capture_idx = -1;
        pending_alt_depth       = 0;

        if (prompt.empty() || target_layer_ids.empty()) {
            return;
        }

        const int n_to_decode = (int) prompt.size();
        if (n_to_decode <= 0) {
            return;
        }

        // Fast path: caller's prefill already produced captures covering
        // the prompt. Two signals:
        //   (1) inline-encoder path: ask ctx_tgt for the cumulative
        //       committed-rows high-water mark. Survives chunked prompt-
        //       eval, where captured_n_outputs would only reflect the
        //       LAST chunk's count.
        //   (2) legacy (non-inline) path: captured-feature count, set
        //       per-llama_decode-call. Sufficient when the caller's
        //       prefill is a single llama_decode (CLI prefill, or server
        //       prefill that fits in one ubatch with logits=true on
        //       every position).
        // The device path in extend_side_store handles both cases
        // uniformly when n_committed >= n_to_decode.
        {
            int64_t n_captures   = 0;
            (void) llama_dflash_get_captured_features(ctx_tgt, &n_captures);
            const int64_t n_committed_inline =
                dflash_inline_encoder
                    ? llama_dflash_inline_get_n_committed(ctx_tgt)
                    : 0;
            const int64_t n_committed = std::max(n_captures, n_committed_inline);

            if (n_committed >= (int64_t) n_to_decode) {
                if (extend_side_store(n_to_decode, /*pos_start=*/0)) {
                    n_committed_total = n_to_decode;
                    return;
                }
                // extend_side_store already logged the error; fall through.
            }
        }

        // Fallback: caller didn't install capture before its prefill; do
        // the documented full re-decode of the prompt with logits=true at
        // every position so capture fires for each.
        llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, 0, -1);

        common_batch_clear(batch_tgt);
        for (int i = 0; i < n_to_decode; ++i) {
            common_batch_add(batch_tgt, prompt[i], (llama_pos) i, { 0 }, /*logits=*/true);
        }
        if (llama_decode(ctx_tgt, batch_tgt) != 0) {
            LOG_ERR("%s: target prompt prefill (with capture) failed\n", __func__);
            return;
        }

        if (!extend_side_store(n_to_decode, /*pos_start=*/0)) {
            return;
        }
        n_committed_total = n_to_decode;
    }

    void draft(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            llama_tokens & result) override {
        if (target_layer_ids.empty() || mask_token_id == LLAMA_TOKEN_NULL) {
            return;
        }

        // Resolve block size from draft's trained block_size.
        const int block_size = (int) block_size_default;
        if (block_size <= 1) {
            return;
        }
        // Note: params.draft.n_max may be smaller than block_size - 1 (the
        // user asked for a shorter draft). DFlash always runs a full
        // block decode; we emit min(block_size - 1, params.draft.n_max)
        // tokens to the verifier. Per paper §5.4.4 only the larger-than-
        // trained direction is un-generalisable; emitting fewer is fine.
        const int n_emit = (params.draft.n_max > 0)
            ? std::min<int>(block_size - 1, params.draft.n_max)
            : block_size - 1;

        using clk = std::chrono::steady_clock;
        const bool prof = common_dflash_prof_enabled();
        const auto t0_round = prof ? clk::now() : clk::time_point{};

        // Step 1: bring the K/V side store up to date with whatever the
        // verify loop accepted since the previous call.
        const int64_t n_committed_after = (int64_t) prompt_tgt.size();
        const int64_t n_to_extend       = n_committed_after - n_committed_total;
        if (n_to_extend < 0) {
            LOG_ERR("%s: prompt_tgt shrank below n_committed_total (%lld < %lld)\n",
                    __func__, (long long) n_committed_after, (long long) n_committed_total);
            return;
        }
        const auto t1_enc_start = prof ? clk::now() : clk::time_point{};
        if (n_to_extend > 0) {
            NVTX_PUSH("dflash_encoder");
            if (!extend_side_store(n_to_extend, /*pos_start=*/n_committed_total)) {
                NVTX_POP();
                return;
            }
            NVTX_POP();
            n_committed_total = n_committed_after;
        }
        const auto t2_enc_end = prof ? clk::now() : clk::time_point{};

        // Step 2: run the draft on a block [id_last, MASK, ..., MASK].
        // Reset draft KV cache (side store carries the persistent state).
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, 0, -1);

        const llama_pos pos_block_start = (llama_pos) n_committed_total;
        common_batch_clear(batch_dft);
        for (int i = 0; i < block_size; ++i) {
            const llama_token tok = (i == 0) ? id_last : mask_token_id;
            common_batch_add(batch_dft, tok, pos_block_start + i, { 0 }, /*logits=*/true);
        }
        const auto t3_dec_start = prof ? clk::now() : clk::time_point{};
        NVTX_PUSH("draft_decode");
        if (llama_decode(ctx_dft, batch_dft) != 0) {
            NVTX_POP();
            LOG_ERR("%s: draft llama_decode failed\n", __func__);
            return;
        }
        NVTX_POP();
        const auto t4_dec_end = prof ? clk::now() : clk::time_point{};

        // Step 3: read top-K candidates per draft position (bidirectional
        // intra-block attention: position i predicts the token AT position
        // i, position 0 is the anchor = id_last).
        int64_t  n_topk_outputs = 0;
        uint32_t topk_K         = 0;
        const int32_t * draft_topk = llama_dflash_get_draft_topk(ctx_dft, &n_topk_outputs, &topk_K);
        if (draft_topk == nullptr || n_topk_outputs < block_size || topk_K < 1) {
            LOG_ERR("%s: draft top-K buffer missing or too small (got n=%lld K=%u, need n>=%d K>=1)\n",
                    __func__, (long long) n_topk_outputs, topk_K, block_size);
            return;
        }

        // Emit positions [1..n_emit] as draft tokens (position 0 is the anchor).
        result.reserve(n_emit);
        for (int i = 1; i <= n_emit; ++i) {
            result.push_back((llama_token) draft_topk[(size_t) i * topk_K + 0]);
        }

        // Reset draft cache so the next iteration starts from a clean slate.
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, 0, -1);

        if (prof) {
            const auto t5_round_end = clk::now();
            const int64_t enc_ns   = std::chrono::duration_cast<std::chrono::nanoseconds>(t2_enc_end - t1_enc_start).count();
            const int64_t dec_ns   = std::chrono::duration_cast<std::chrono::nanoseconds>(t4_dec_end - t3_dec_start).count();
            const int64_t total_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t5_round_end - t0_round).count();
            g_dflash_prof.encoder_ns      += enc_ns;
            g_dflash_prof.draft_decode_ns += dec_ns;
            g_dflash_prof.draft_other_ns  += (total_ns - enc_ns - dec_ns);
            g_dflash_prof.n_rounds        += 1;
        }
    }

    void accept(uint16_t /*n_accepted*/) override {
        // n_committed_total is updated lazily in draft() based on the size
        // of the incoming prompt_tgt; nothing to do here.
    }

    // tree-shaped variant of draft(): same draft compute (one DFlash forward
    // pass), different post-processing - the tree builder consumes draft_topk
    // directly instead
    // of flattening to a chain.
    void draft_tree(
            const common_params_speculative & params,
            const llama_tokens & prompt_tgt,
            llama_token id_last,
            common_speculative_tree & result) override {
        result = {};
        if (target_layer_ids.empty() || mask_token_id == LLAMA_TOKEN_NULL) {
            return;
        }

        const int block_size = (int) block_size_default;
        if (block_size <= 1) {
            return;
        }

        // Bring the K/V side store up to date with whatever was committed
        // since the previous draft / draft_tree call.
        const int64_t n_committed_after = (int64_t) prompt_tgt.size();
        const int64_t n_to_extend       = n_committed_after - n_committed_total;
        if (n_to_extend < 0) {
            LOG_ERR("%s: prompt_tgt shrank below n_committed_total (%lld < %lld)\n",
                    __func__, (long long) n_committed_after, (long long) n_committed_total);
            return;
        }
        if (n_to_extend > 0) {
            if (!extend_side_store(n_to_extend, /*pos_start=*/n_committed_total)) {
                return;
            }
            n_committed_total = n_committed_after;
        }

        // Run the draft once on [id_last, MASK, ..., MASK].
        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, 0, -1);

        const llama_pos pos_block_start = (llama_pos) n_committed_total;
        common_batch_clear(batch_dft);
        for (int i = 0; i < block_size; ++i) {
            const llama_token tok = (i == 0) ? id_last : mask_token_id;
            common_batch_add(batch_dft, tok, pos_block_start + i, { 0 }, /*logits=*/true);
        }
        if (llama_decode(ctx_dft, batch_dft) != 0) {
            LOG_ERR("%s: draft llama_decode failed\n", __func__);
            return;
        }

        // Read top-K candidates per draft position.
        int64_t  n_topk_outputs = 0;
        uint32_t topk_K         = 0;
        const int32_t * draft_topk = llama_dflash_get_draft_topk(ctx_dft, &n_topk_outputs, &topk_K);
        if (draft_topk == nullptr || n_topk_outputs < block_size || topk_K < 1) {
            LOG_ERR("%s: draft top-K buffer missing or too small (got n=%lld K=%u, need n>=%d K>=1)\n",
                    __func__, (long long) n_topk_outputs, topk_K, block_size);
            return;
        }

        // resolve tree budget; 0 = default shape (= block_size)
        const int budget = (params.dflash_tree_budget > 0)
                         ? params.dflash_tree_budget
                         : block_size;

        if (params.dflash_tree_best_first && topk_K >= 3) {
            // pull full-vocab logits, compute per-position log-probs (online
            // logsumexp), and select leaf-alts ordered by log-prob instead of
            // round-robin rank. dflash_emit_logits must be set on ctx_dft
            // work; if absent we fall through to uniform.
            //
            // Skipped for K<3: with K=2 there is exactly one alt per depth,
            // so best-first picks the same set as uniform — no point paying
            // the logit-readback cost.
            const float * logits = llama_get_logits(ctx_dft);
            const int     vocab  = llama_vocab_n_tokens(
                llama_model_get_vocab(llama_get_model(ctx_dft)));
            if (logits != nullptr && vocab > 0) {
                const size_t buf_floats = (size_t) block_size * (size_t) topk_K;
                if (best_first_logprobs.size() < buf_floats) {
                    best_first_logprobs.resize(buf_floats);
                    best_first_tokens.resize(buf_floats);
                }
                extract_draft_topk_logprobs(
                    logits, block_size, vocab, (int) topk_K,
                    best_first_logprobs.data(), best_first_tokens.data());
                build_ddtree_tree_bestfirst_leaf(
                    best_first_logprobs.data(), best_first_tokens.data(),
                    block_size, topk_K, budget, result);
            } else {
                LOG_WRN("%s: dflash_tree_best_first set but ctx_dft did not emit logits; "
                        "falling back to uniform expansion. Did you forget "
                        "dflash_emit_logits=true on the draft cparams?\n", __func__);
                build_ddtree_tree(draft_topk, block_size, topk_K, budget, result);
            }
        } else {
            build_ddtree_tree(draft_topk, block_size, topk_K, budget, result);
        }

        llama_memory_seq_rm(llama_get_memory(ctx_dft), 0, 0, -1);
    }

    void record_alt_accept(int alt_capture_idx, int alt_depth) override {
        pending_alt_capture_idx = alt_capture_idx;
        pending_alt_depth       = alt_depth;
    }

    int32_t n_max(const common_params_speculative & params) const override {
        const int32_t bs_max = (int32_t) block_size_default - 1;
        return params.draft.n_max > 0 ? std::min(params.draft.n_max, bs_max) : bs_max;
    }

    int32_t n_min(const common_params_speculative & params) const override {
        return params.draft.n_min;
    }

private:
    // Read freshly-captured target features from ctx_tgt and push the first
    // `n_keep` of them into the draft's per-layer K/V side store at offset
    // `pos_start`. Wraps llama_dflash_extend with a friendlier error path.
    //
    // alt-accept fast path: when a record_alt_accept hint is pending, the
    // captures buffer holds the previous tree-decode outputs at indices
    // [0..main_path_len, alt_1, ..., alt_n]; push
    // [0, 1, ..., d-1, alt_capture_idx] into the side store, replacing the
    // rejected m_d capture with the accepted alt's at depth d. The remap is
    // done host-side into alt_remap_buf.
    bool extend_side_store(int64_t n_keep, int64_t pos_start) {
        // inline-encoder fast path: the encoder ops already ran inside the
        // target's main decode graph and wrote n_outputs captured rows to
        // side-store slots [write_offset..write_offset
        // + n_outputs - 1]. All we need to do is advance ctx_filled by
        // n_keep (the accepted-token count) so the draft attention sees
        // exactly those rows. Unaccepted rows live above the boundary
        // and are overwritten by the next iter's extend before any read.
        //
        // Chain mode only — tree mode's alt-remap still needs the host
        // path. pending_alt_capture_idx >= 0 signals alt-accept.
        if (dflash_inline_encoder && pending_alt_capture_idx < 0) {
            // The inline encoder already wrote captured rows into the draft
            // side store during target's decode. Advance ctx_filled so the
            // draft attention sees them. slide_left in inline mode isn't
            // wired yet — fail explicitly if we'd need it (sensible chain-
            // mode limits don't hit this in practice).
            (void) pos_start;
            llama_dflash_inline_advance_ctx_filled(ctx_dft, n_keep);
            return true;
        }

        int64_t n_outputs = 0;
        const float * captures = llama_dflash_get_captured_features(ctx_tgt, &n_outputs);

        // alt-remap needs host bytes for the row reshuffle. In
        // skip_host_readback mode the target didn't D2H this iteration, so
        // issue a one-shot readback. The blocking sync inside is the
        // (rare) cost for tree-mode alt-accept.
        if (pending_alt_capture_idx >= 0 && captures == nullptr && n_outputs > 0) {
            const int32_t rc = llama_dflash_force_host_readback(ctx_tgt);
            if (rc != 0) {
                LOG_ERR("%s: alt-remap force readback failed (rc=%d)\n", __func__, rc);
                return false;
            }
            captures = llama_dflash_get_captured_features(ctx_tgt, &n_outputs);
        }

        // device fast path: when no alt-remap is pending, read straight from
        // the target's device-resident packed captures and skip the host
        // bounce entirely.
        const bool can_use_device_path = (pending_alt_capture_idx < 0);
        if (can_use_device_path) {
            // n_outputs may be 0 here if skip_host_readback dropped the
            // host buffer; ask the target context how many captures it
            // produced via the embedded host-side counter (still kept up
            // to date even when bytes are skipped).
            int64_t n_outputs_dev = 0;
            (void) llama_dflash_get_captured_features(ctx_tgt, &n_outputs_dev);
            if (n_outputs_dev == 0) {
                // No captures produced this step — fall through to error
                // path below if we have nothing usable.
            } else if (n_keep > n_outputs_dev) {
                LOG_ERR("%s: asked to keep %lld captures but target produced only %lld\n",
                        __func__, (long long) n_keep, (long long) n_outputs_dev);
                return false;
            } else {
                // when inline-encoder is requested, run the encoder on the
                // target's scheduler. Falls back to the draft-side path on
                // failure so a stricter regression is still possible.
                if (dflash_inline_encoder) {
                    const int32_t rc_inline = llama_dflash_inline_encode_from_ctx(
                        ctx_tgt, ctx_dft, /*src_row_offset=*/0, n_keep, pos_start);
                    if (rc_inline == 0) {
                        return true;
                    }
                    LOG_WRN("%s: llama_dflash_inline_encode_from_ctx failed "
                            "(rc=%d); falling back to draft-side encoder\n",
                            __func__, rc_inline);
                }
                const int32_t rc = llama_dflash_extend_from_ctx(
                    ctx_dft, ctx_tgt, /*src_row_offset=*/0, n_keep, pos_start);
                if (rc == 0) {
                    return true;
                }
                // device path unavailable (e.g. last_packed_captures was
                // cleared because the prefill spanned multiple ubatches and
                // the per-ubatch tensor only holds a subset of the rows);
                // fall through to the host path below, which is accumulated
                // correctly across ubatches.
                if (captures == nullptr || n_outputs < n_keep) {
                    LOG_ERR("%s: llama_dflash_extend_from_ctx failed (rc=%d) "
                            "and no host fallback available (captures=%p, "
                            "n_outputs=%lld, n_keep=%lld)\n",
                            __func__, rc, (const void *) captures,
                            (long long) n_outputs, (long long) n_keep);
                    return false;
                }
            }
        }

        if (captures == nullptr || n_outputs == 0) {
            LOG_ERR("%s: target captured no features (capture not installed or no outputs requested)\n", __func__);
            return false;
        }
        if (n_keep > n_outputs) {
            LOG_ERR("%s: asked to keep %lld captures but target produced only %lld\n",
                    __func__, (long long) n_keep, (long long) n_outputs);
            return false;
        }

        const float * extend_buf = captures;

        if (pending_alt_capture_idx >= 0) {
            const int     d                = pending_alt_depth;
            const int     alt_idx          = pending_alt_capture_idx;
            const int64_t expected_n_keep  = (int64_t) d + 1;
            const size_t  row_floats       = (size_t) n_features;

            if (n_keep != expected_n_keep) {
                LOG_ERR("%s: alt-accept remap: n_keep=%lld but alt_depth+1=%lld (out of sync)\n",
                        __func__, (long long) n_keep, (long long) expected_n_keep);
                return false;
            }
            if (alt_idx >= n_outputs) {
                LOG_ERR("%s: alt-accept remap: alt_capture_idx=%d but only %lld outputs\n",
                        __func__, alt_idx, (long long) n_outputs);
                return false;
            }
            if (row_floats == 0) {
                LOG_ERR("%s: alt-accept remap: n_features=0\n", __func__);
                return false;
            }

            const size_t buf_size = (size_t) (d + 1) * row_floats;
            if (alt_remap_buf.size() < buf_size) {
                alt_remap_buf.resize(buf_size);
            }
            if (d > 0) {
                std::memcpy(
                    alt_remap_buf.data(),
                    captures,
                    (size_t) d * row_floats * sizeof(float));
            }
            std::memcpy(
                alt_remap_buf.data() + (size_t) d * row_floats,
                captures + (size_t) alt_idx * row_floats,
                row_floats * sizeof(float));
            extend_buf = alt_remap_buf.data();

            // Consume the hint: subsequent extends use the linear path.
            pending_alt_capture_idx = -1;
            pending_alt_depth       = 0;
        }

        const int32_t rc = llama_dflash_extend(ctx_dft, extend_buf, n_keep, pos_start);
        if (rc != 0) {
            LOG_ERR("%s: llama_dflash_extend failed (rc=%d)\n", __func__, rc);
            return false;
        }
        return true;
    }

    // host-side top-K + log-prob extraction from raw draft logits, used by
    // best-first tree expansion: single pass over the vocab per position,
    // online logsumexp + bounded-K heap. Output is descending by log-prob.
    //
    // Inputs:  logits [n_pos * vocab]
    // Outputs: out_log_probs [n_pos * K], out_token_ids [n_pos * K]
    //          both sorted by log-prob DESCENDING (rank 0 = argmax).
    static void extract_draft_topk_logprobs(const float * logits,
                                            int           n_pos,
                                            int           vocab,
                                            int           K,
                                            float       * out_log_probs,
                                            int32_t     * out_token_ids) {
        struct Entry { float logit; int32_t id; };
        auto cmp_greater = [](const Entry & a, const Entry & b) {
            return a.logit > b.logit;
        };

        for (int i = 0; i < n_pos; ++i) {
            const float * li = logits + (size_t) i * (size_t) vocab;
            std::vector<Entry> heap;
            heap.reserve(K);

            float running_max     = -INFINITY;
            float running_sum_exp = 0.0f;
            for (int j = 0; j < vocab; ++j) {
                const float l = li[j];

                if (l > running_max) {
                    if (running_max > -INFINITY) {
                        running_sum_exp = running_sum_exp * std::exp(running_max - l);
                    }
                    running_sum_exp += 1.0f;
                    running_max = l;
                } else {
                    running_sum_exp += std::exp(l - running_max);
                }

                if ((int) heap.size() < K) {
                    heap.push_back({ l, (int32_t) j });
                    std::push_heap(heap.begin(), heap.end(), cmp_greater);
                } else if (l > heap.front().logit) {
                    std::pop_heap(heap.begin(), heap.end(), cmp_greater);
                    heap.back() = { l, (int32_t) j };
                    std::push_heap(heap.begin(), heap.end(), cmp_greater);
                }
            }

            const float log_z = running_max + std::log(running_sum_exp);
            std::sort_heap(heap.begin(), heap.end(), cmp_greater);
            for (int k = 0; k < K; ++k) {
                out_log_probs[(size_t) i * (size_t) K + k] = heap[k].logit - log_z;
                out_token_ids[(size_t) i * (size_t) K + k] = heap[k].id;
            }
        }
    }

    // best-first tree shape: chain seed (top-1 at depths 1..bs-1, capped at
    // budget) plus best-first leaf-alt selection from (depth, rank) tuples
    // ordered by per-position log-prob.
    //
    // Same topology family as the uniform-expansion shape (leaf-alts only,
    // compatible with the verify-side seq_id tagging in speculative-simple.cpp);
    // the difference is *which* alts get included when budget < (K-1)*chain_len.
    // Uniform expansion fills rank 1 across all depths first, then rank 2,
    // etc. — depth-blind. Best-first picks the (depth, rank) tuples with the
    // highest log-prob regardless of rank, which biases the budget toward
    // alts the draft is most uncertain about.
    //
    // Deeper alt subtrees would require ancestry-aware seq_id tagging on the
    // verify batch and are deferred.
    static void build_ddtree_tree_bestfirst_leaf(
            const float   * topk_logprobs,
            const int32_t * topk_tokens,
            int             block_size,
            uint32_t        topk_K,
            int             budget,
            common_speculative_tree & tree) {
        tree.tokens.clear();
        tree.parents.clear();
        tree.depths.clear();
        tree.branch_ids.clear();
        tree.child_maps.clear();
        tree.visibility.clear();
        tree.n_nodes       = 0;
        tree.main_path_len = 0;
        tree.n_branches    = 1;

        if (budget <= 0 || block_size <= 1) {
            tree.parents.push_back(-1);
            tree.child_maps.emplace_back();
            tree.visibility.assign(1, 1);
            return;
        }

        tree.parents.push_back(-1);
        tree.child_maps.emplace_back();

        const int chain_target = std::min(block_size - 1, budget);
        {
            int parent = 0;
            for (int d = 1; d <= chain_target; ++d) {
                const llama_token tok = (llama_token) topk_tokens[(size_t) d * topk_K + 0];
                const int idx = tree.n_nodes + 1;
                tree.tokens.push_back(tok);
                tree.parents.push_back(parent);
                tree.depths.push_back(d);
                tree.branch_ids.push_back(0);
                tree.child_maps.emplace_back();
                tree.child_maps[parent][tok] = idx;
                tree.n_nodes++;
                parent = idx;
            }
            tree.main_path_len = tree.n_nodes;
        }

        if (topk_K >= 2 && tree.n_nodes < budget) {
            // Cumulative-path-probability best-first expansion (ported from
            // lucebox's build_ddtree / Ringel & Romano arXiv:2604.12989).
            //
            // During chain pre-seed above, accumulate per-depth cumulative
            // log-prob of the main chain (top-1 at each depth). Then seed
            // a max-heap with the rank-1 sibling at each chain depth, scored
            // by the cumulative log-prob of the *path from root* that leads
            // to that sibling (= chain cumul up to parent + sibling's local
            // log-prob). Expansion pops the highest-path-prob candidate,
            // places it in the tree, and pushes its children (next-depth
            // siblings along the same branch) back onto the heap.
            //
            // This concentrates sibling nodes at positions where the draft's
            // uncertainty is highest relative to the path probability, which
            // produces ~1.3 higher AL than the flat per-position sort used
            // previously. Reference: lucebox RESULTS.md "Chain pre-seed in
            // build_ddtree" row (+5 AL).

            // Compute cumulative chain log-prob at each depth (for scoring).
            std::vector<float> chain_cum_logp(chain_target + 1, 0.0f);
            for (int d = 1; d <= chain_target; ++d) {
                chain_cum_logp[d] = chain_cum_logp[d - 1]
                    + topk_logprobs[(size_t) d * topk_K + 0];
            }

            struct HeapEntry {
                float   neg_path_logp; // negated for min-heap → max-path-prob
                int     parent_idx;    // index in the flat tree (0 = root)
                int     depth;         // 1-based depth in the tree
                int32_t tok;
            };
            auto heap_cmp = [](const HeapEntry & a, const HeapEntry & b) {
                return a.neg_path_logp > b.neg_path_logp;
            };
            std::priority_queue<HeapEntry, std::vector<HeapEntry>,
                                decltype(heap_cmp)> heap(heap_cmp);

            // Seed heap: for each chain depth d, push the rank-1 sibling
            // with path score = chain_cum_logp[d-1] + sibling_local_logp.
            for (int d = 1; d <= chain_target; ++d) {
                for (uint32_t rank = 1; rank < topk_K; ++rank) {
                    const size_t pos_idx = (size_t) d * topk_K + rank;
                    const int32_t tok = topk_tokens[pos_idx];
                    if (tok < 0) continue;
                    const float path_logp = chain_cum_logp[d - 1]
                        + topk_logprobs[pos_idx];
                    heap.push({ -path_logp, /*parent_idx=*/ d - 1,
                                /*depth=*/ d, tok });
                }
            }

            int next_branch = 1;
            while (!heap.empty() && tree.n_nodes < budget) {
                const HeapEntry entry = heap.top();
                heap.pop();

                // Skip if this parent already has this token as a child
                // (can happen if multiple heap paths converge on the same
                // (parent, tok) pair via different scoring routes).
                if (tree.child_maps[entry.parent_idx].count(entry.tok)) {
                    continue;
                }

                const int idx = tree.n_nodes + 1;
                tree.tokens.push_back((llama_token) entry.tok);
                tree.parents.push_back(entry.parent_idx);
                tree.depths.push_back(entry.depth);
                tree.branch_ids.push_back(next_branch);
                tree.child_maps.emplace_back();
                tree.child_maps[entry.parent_idx][(llama_token) entry.tok] = idx;
                tree.n_nodes++;
                next_branch++;

                // Push this node's children (next-depth candidates along the
                // same branch) back onto the heap if they exist in the
                // topk_logprobs array. The child at next depth inherits this
                // node's path_logp as its parent-path score.
                const int child_depth = entry.depth + 1;
                if (child_depth <= chain_target) {
                    const float parent_path_logp = -entry.neg_path_logp;
                    for (uint32_t rank = 0; rank < topk_K; ++rank) {
                        const size_t pos_idx = (size_t) child_depth * topk_K + rank;
                        const int32_t child_tok = topk_tokens[pos_idx];
                        if (child_tok < 0) continue;
                        const float child_path_logp = parent_path_logp
                            + topk_logprobs[pos_idx];
                        heap.push({ -child_path_logp, /*parent_idx=*/ idx,
                                    /*depth=*/ child_depth, child_tok });
                    }
                }
            }
            tree.n_branches = next_branch;
        }

        const int n = tree.n_nodes + 1;
        tree.visibility.assign((size_t) n * n, 0);
        tree.visibility[0 * n + 0] = 1;
        for (int i = 1; i < n; ++i) {
            const int parent = tree.parents[i];
            for (int j = 0; j < i; ++j) {
                tree.visibility[(size_t) i * n + j] = tree.visibility[(size_t) parent * n + j];
            }
            tree.visibility[(size_t) i * n + i] = 1;
        }
    }

    // uniform tree shape: chain seed (top-1 at depths 1..bs-1, capped at
    // budget) + round-robin sibling expansion (rank 1..K-1 across all depths)
    // until budget is exhausted. Each alt is a leaf hanging
    // off the main chain at its depth and gets a unique branch_id.
    static void build_ddtree_tree(const int32_t * draft_topk,
                                  int             block_size,
                                  uint32_t        topk_K,
                                  int             budget,
                                  common_speculative_tree & tree) {
        tree.tokens.clear();
        tree.parents.clear();
        tree.depths.clear();
        tree.branch_ids.clear();
        tree.child_maps.clear();
        tree.visibility.clear();
        tree.n_nodes       = 0;
        tree.main_path_len = 0;
        tree.n_branches    = 1;

        if (budget <= 0 || block_size <= 1) {
            // Degenerate: return empty tree (just the implicit root).
            tree.parents.push_back(-1);
            tree.child_maps.emplace_back();
            tree.visibility.assign(1, 1);
            return;
        }

        // Root.
        tree.parents.push_back(-1);
        tree.child_maps.emplace_back();

        // ----- chain seed (main path) -----
        const int chain_target = std::min(block_size - 1, budget);
        {
            int parent = 0;
            for (int d = 1; d <= chain_target; ++d) {
                const llama_token tok = (llama_token) draft_topk[(size_t) d * topk_K + 0];
                const int idx = tree.n_nodes + 1;
                tree.tokens.push_back(tok);
                tree.parents.push_back(parent);
                tree.depths.push_back(d);
                tree.branch_ids.push_back(0); // main branch
                tree.child_maps.emplace_back();
                tree.child_maps[parent][tok] = idx;
                tree.n_nodes++;
                parent = idx;
            }
            tree.main_path_len = tree.n_nodes;
        }

        // ----- alt leaves (uniform expansion) -----
        if (topk_K >= 2 && tree.n_nodes < budget) {
            int next_branch = 1;
            for (uint32_t rank = 1; rank < topk_K && tree.n_nodes < budget; ++rank) {
                for (int d = 1; d <= chain_target && tree.n_nodes < budget; ++d) {
                    const int alt_parent_idx = d - 1; // root or main_path[d-1]
                    const llama_token alt_tok =
                        (llama_token) draft_topk[(size_t) d * topk_K + rank];
                    if (alt_tok < 0) continue;
                    if (tree.child_maps[alt_parent_idx].count(alt_tok)) continue;

                    const int idx = tree.n_nodes + 1;
                    tree.tokens.push_back(alt_tok);
                    tree.parents.push_back(alt_parent_idx);
                    tree.depths.push_back(d);
                    tree.branch_ids.push_back(next_branch);
                    tree.child_maps.emplace_back();
                    tree.child_maps[alt_parent_idx][alt_tok] = idx;
                    tree.n_nodes++;
                    next_branch++;
                }
            }
            tree.n_branches = next_branch;
        }

        // ----- visibility matrix (used for the verify-side tree mask) -----
        const int n = tree.n_nodes + 1;
        tree.visibility.assign((size_t) n * n, 0);
        tree.visibility[0 * n + 0] = 1;
        for (int i = 1; i < n; ++i) {
            const int parent = tree.parents[i];
            for (int j = 0; j < i; ++j) {
                tree.visibility[(size_t) i * n + j] = tree.visibility[(size_t) parent * n + j];
            }
            tree.visibility[(size_t) i * n + i] = 1;
        }
    }
};

struct common_speculative {
    std::vector<std::unique_ptr<common_speculative_state>> impls; // list of implementations to use and their states

    common_speculative_state * curr_impl = nullptr; // current implementation in use (for stats)
};

static common_ngram_map get_common_ngram_map(
        common_speculative_type type,
        const common_params_speculative_ngram_map & config) {
    uint16_t size_key   = config.size_n;
    uint16_t size_value = config.size_m;
    bool     key_only   = type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K;
    uint16_t min_hits   = config.min_hits;

    return common_ngram_map(size_key, size_value, key_only, min_hits);
}

static common_speculative_state_ngram_cache create_state_ngram_cache(
        const std::string & path_static, const std::string & path_dynamic,
        const common_speculative_config & config) {
    uint16_t n_draft = 8; // TODO get from config?

    // TODO bool param in common/common.h to set save_static/save_dynamic?
    bool save_static = false;
    bool save_dynamic = false;

    common_speculative_state_ngram_cache state(config.type, path_static, path_dynamic, n_draft, save_static, save_dynamic);

    return state;
}

std::string common_speculative_type_name_str() {
    std::string result;
    for (size_t i = 0; i < common_speculative_types.size(); i++) {
        if (i > 0) {
            result += ", ";
        }
        result += common_speculative_type_to_str(common_speculative_types[i]);
    }
    return result;
}

std::string common_speculative_type_to_str(enum common_speculative_type type) {
    switch (type) {
        case COMMON_SPECULATIVE_TYPE_NONE:          return "none";
        case COMMON_SPECULATIVE_TYPE_DRAFT:         return "draft";
        case COMMON_SPECULATIVE_TYPE_EAGLE3:        return "eagle3";
        case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE:  return "ngram_simple";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:   return "ngram_map_k";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: return "ngram_map_k4v";
        case COMMON_SPECULATIVE_TYPE_NGRAM_MOD:     return "ngram_mod";
        case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE:   return "ngram_cache";
        case COMMON_SPECULATIVE_TYPE_DFLASH:        return "dflash";
        default:                                    return "unknown";
    }
}

enum common_speculative_type common_speculative_type_from_name(const std::string & name) {
    const auto it = common_speculative_type_from_name_map.find(name);
    if (it == common_speculative_type_from_name_map.end()) {
        return COMMON_SPECULATIVE_TYPE_COUNT;
    }
    return it->second;
}

// initialization of the speculative decoding system
//
common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt) {
    llama_context * ctx_dft = nullptr;

    // Detect DFlash drafts before context creation: if the user explicitly
    // requested DFlash, OR if the loaded draft GGUF carries DFlash metadata
    // and no other type was requested, the target's tok_embd / lm_head
    // must be bound into the draft model BEFORE `llama_init_from_model`
    // (graph_reserve runs at context creation and reads target_tok_embd /
    // target_output via the bound model.target_* fields).
    const bool draft_is_dflash =
        params.draft.model != nullptr &&
        llama_model_dflash_block_size(params.draft.model) > 0;

    const bool want_dflash =
        (params.type == COMMON_SPECULATIVE_TYPE_DFLASH) ||
        (draft_is_dflash && params.type == COMMON_SPECULATIVE_TYPE_NONE);

    if (want_dflash) {
        if (!draft_is_dflash) {
            LOG_ERR("%s: --draft-type dflash requires a DFlash draft GGUF "
                    "(missing dflash.block_size metadata)\n", __func__);
            return nullptr;
        }
        const llama_model * model_tgt = llama_get_model(ctx_tgt);
        if (!llama_dflash_bind_target(params.draft.model, model_tgt)) {
            LOG_WRN("%s: llama_dflash_bind_target returned false; the draft "
                    "graph will fall back to its self-contained tensors\n",
                    __func__);
        }

        // inline-encoder plumbing: populate non-owning pointers on the target
        // model that reference the draft's encoder weights (dflash_fc,
        // dflash_hidden_norm, per-layer wk/wv/k_norm). When
        // --dflash-inline-encoder is set, the graph builder hooks these into
        // the target's main decode graph after the final captured layer.
        if (params.dflash_inline_encoder) {
            if (!llama_dflash_bind_encoder(
                    const_cast<llama_model *>(model_tgt),
                    params.draft.model)) {
                LOG_WRN("%s: llama_dflash_bind_encoder returned false; the "
                        "inline encoder will be unavailable\n", __func__);
            }
        }
    }

    if (params.draft.model) {
        ctx_dft = llama_init_from_model(params.draft.model, params.draft.cparams);
        if (ctx_dft == nullptr) {
            LOG_ERR("%s", "failed to create draft context\n");
            return nullptr;
        }
    }

    // inline-encoder: bind draft's side-store K/V tensors onto the target
    // context so the target's inline encoder ops can set_rows into them.
    // Done here (after draft context creation) because the side
    // store is allocated in the draft context's constructor.
    if (want_dflash && params.dflash_inline_encoder && ctx_dft != nullptr) {
        if (!llama_dflash_bind_inline_side_store(ctx_tgt, ctx_dft)) {
            LOG_WRN("%s: llama_dflash_bind_inline_side_store returned false; "
                    "inline encoder will fall back to draft-side path\n",
                    __func__);
        }
    }

    // Compute the implementations to use based on the config and their order of preference
    std::vector<common_speculative_config> configs = {}; // list of speculative configs to try
    {
        bool has_draft = !params.draft.mparams.path.empty();
        bool has_draft_eagle3 = false; // TODO PR-18039: if params.speculative.eagle3

        bool has_ngram_cache   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_CACHE);
        bool has_ngram_simple  = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE);
        bool has_ngram_map_k   = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K);
        bool has_ngram_map_k4v = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V);
        bool has_ngram_mod     = (params.type == COMMON_SPECULATIVE_TYPE_NGRAM_MOD);
        bool has_dflash        = want_dflash;

        // In a more complex implementation we could use the same implementation but with different parameters.
        // This was initially used in PR-18471 but removed to simplify the code.
        if (has_ngram_simple) {
            // This implementation can guess a lot of tokens without any draft model.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE, params));
        }
        if (has_ngram_map_k) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K, params));
        }
        if (has_ngram_map_k4v) {
            // This implementation can guess tokens with high acceptance rate but is more expensive.
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V, params));
        }
        if (has_ngram_mod) {
            auto & sparams = params.ngram_mod;

            if (!sparams.obj) {
                sparams.obj = std::make_shared<common_ngram_mod>(sparams.n_match, 4*1024*1024);

                LOG_INF("%s: initialized ngram_mod with n_match=%d, size=%zu (%.3f MB)\n", __func__,
                        sparams.n_match, sparams.obj->size(), (float)(sparams.obj->size_bytes())/1024/1024);

                if (sparams.n_match < 16) {
                    LOG_WRN("%s: ngram_mod n_match=%d is too small - poor quality is possible, "
                            "see: https://github.com/ggml-org/llama.cpp/pull/19164\n", __func__, sparams.n_match);
                }
            }

            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_MOD, params));
        }
        if (has_ngram_cache) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_NGRAM_CACHE, params));
        }
        if (has_dflash) {
            // DFlash takes precedence over the regular DRAFT path when the
            // draft model is a DFlash GGUF; the two are mutually exclusive
            // (DFlash binds target tensors into the draft model, which a
            // regular small-draft path doesn't expect).
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DFLASH, params));
        } else if (has_draft) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_DRAFT, params));
        }
        if (has_draft_eagle3) {
            configs.push_back(common_speculative_config(COMMON_SPECULATIVE_TYPE_EAGLE3, params));
        }
    }

    std::vector<std::unique_ptr<common_speculative_state>> impls = {};

    for (const common_speculative_config & config : configs) {
        LOG_DBG("%s: adding implementation %s\n", __func__, common_speculative_type_to_str(config.type).c_str());
        switch (config.type) {
            case COMMON_SPECULATIVE_TYPE_NONE:
                break;
            case COMMON_SPECULATIVE_TYPE_DRAFT: {
                const bool use_ckpt = common_context_can_seq_rm(ctx_dft) == COMMON_CONTEXT_SEQ_RM_TYPE_FULL;

                impls.push_back(std::make_unique<common_speculative_state_draft>(config.type,
                    /* .ctx_tgt      = */ ctx_tgt,
                    /* .ctx_dft      = */ ctx_dft,
                    /* .replacements = */ params.draft.replacements,
                    /* .use_ckpt     = */ use_ckpt
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_EAGLE3: {
                impls.push_back(std::make_unique<common_speculative_state_eagle3>(config.type));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE: {
                common_ngram_map ngram_map = get_common_ngram_map(config.type, config.params.ngram_simple);

                uint16_t ngram_size_key   = ngram_map.size_key;
                uint16_t mgram_size_value = ngram_map.size_value;

                auto config_simple = common_ngram_simple_config {
                    /* .size_ngram = */ ngram_size_key,
                    /* .size_mgram = */ mgram_size_value
                };
                auto state = std::make_unique<common_speculative_state_ngram_simple>(
                    /* .type  = */ config.type,
                    /* .state = */ config_simple
                );
                impls.push_back(std::move(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K:
            case COMMON_SPECULATIVE_TYPE_NGRAM_MAP_K4V: {
                impls.push_back(std::make_unique<common_speculative_state_ngram_map_k>(
                    (config.type),
                    get_common_ngram_map(config.type, config.params.ngram_map_k)
                ));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_MOD: {
                GGML_ASSERT(config.params.ngram_mod.obj);
                impls.push_back(std::make_unique<common_speculative_state_ngram_mod>(config.type, *config.params.ngram_mod.obj));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_NGRAM_CACHE: {
                auto state = create_state_ngram_cache(params.ngram_cache.lookup_cache_static, params.ngram_cache.lookup_cache_dynamic, config);
                impls.push_back(std::make_unique<common_speculative_state_ngram_cache>(state));
                break;
            }
            case COMMON_SPECULATIVE_TYPE_DFLASH: {
                impls.push_back(std::make_unique<common_speculative_state_dflash>(
                    config.type, ctx_tgt, ctx_dft, params.dflash_inline_encoder));
                break;
            }
            default:
                break;
        }
    }

    if (impls.empty()) {
        LOG_WRN("%s", "no implementations specified for speculative decoding\n");
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .impls     = */ std::move(impls),
        /* .curr_impl = */ nullptr,
    };

    return result;
}

void common_speculative_free(common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    delete spec;
}

void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt) {
    if (spec == nullptr) {
        return;
    }

    for (auto & impl : spec->impls) {
        common_time_meas tm(impl->t_begin_us, !impl->gen_perf);
        impl->begin(prompt);
        impl->n_call_begin++;
    }
}

llama_tokens common_speculative_draft(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt, // specified in target model vocab
        llama_token id_last) {
    llama_tokens result;

    spec->curr_impl = nullptr; // reset current implementation

    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft(params, prompt_tgt, id_last, result);
            impl->n_call_draft++;
        }

        {
            const int n_min = impl->n_min(params);

            if (!result.empty() && (int) result.size() < n_min) {
                LOG_DBG("%s: ignoring small draft: %d < %d\n", __func__, (int) result.size(), n_min);
                result.clear();
            }
        }

        if (!result.empty()) {
            LOG_DBG("%s: called impl %s, hist size = %zu, call_count = %zu, gen = %zu\n", __func__,
                    common_speculative_type_to_str(impl.get()->type).c_str(), prompt_tgt.size(),
                    impl.get()->n_call_draft, result.size());

            spec->curr_impl = impl.get(); // set current implementation for stats
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.size();

            break; // we have a draft, so break out of the loop and return it.
        }
    }

    return result;
}

bool common_speculative_tree::write_parent_ids(int32_t * out,
                                               int n_tokens,
                                               int n_seqs) const {
    if (out == nullptr) return false;
    if (n_tokens != n_nodes + 1) return false;
    if (n_seqs <= 0) return false;

    // Seq 0 carries the actual DFS-flattened tree. parents[i] is already
    // in batch-slot terms: 0 = root, j = tree node j (= batch slot j).
    // GGML_GDN_TREE_ROOT_PARENT (= -1) for the root signals
    // "reload from curr_state" in the kernel; for tree nodes whose
    // parent is the root (parents[i] == 0) we keep the 0 so the kernel
    // sees parent_t=0=t-1 for t=1 (sequential continuation), and for
    // deeper-DFS root-children parent_t=0 != t-1 triggers a reload from
    // inter[0] (state-after-root).
    out[0] = -1;
    for (int i = 1; i <= n_nodes; ++i) {
        out[i] = parents[i];
    }

    // Seqs > 0 are unused at the spec-driver level for the single-seq
    // DFS-flattened tree layout, but the kernel still reads them (its
    // n_seqs comes from the ubatch's recurrent dim). Fill with -1 so
    // any per-seq state slabs that aren't actually populated this iter
    // get treated as "reload from curr_state" — i.e. the pre-block
    // recurrent state — and don't corrupt anything.
    for (int s = 1; s < n_seqs; ++s) {
        for (int t = 0; t < n_tokens; ++t) {
            out[(size_t) s * n_tokens + t] = -1;
        }
    }

    return true;
}

common_speculative_tree common_speculative_draft_tree(
        common_speculative * spec,
        const common_params_speculative & params,
        const llama_tokens & prompt_tgt,
        llama_token id_last) {
    common_speculative_tree result;

    if (spec == nullptr) {
        return result;
    }

    spec->curr_impl = nullptr;

    // Tree-mode is currently DFlash-only; iterate impls and use the first
    // one that produces a non-empty tree.
    for (auto & impl : spec->impls) {
        {
            common_time_meas tm(impl->t_draft_us, !impl->gen_perf);
            impl->draft_tree(params, prompt_tgt, id_last, result);
            impl->n_call_draft++;
        }

        if (result.n_nodes > 0) {
            spec->curr_impl = impl.get();
            impl->n_gen_drafts++;
            impl->n_gen_tokens += result.n_nodes;
            break;
        }
    }

    return result;
}

void common_speculative_record_alt_accept(
        common_speculative * spec,
        int                  alt_capture_idx,
        int                  alt_depth) {
    if (spec == nullptr) {
        return;
    }
    for (auto & impl : spec->impls) {
        impl->record_alt_accept(alt_capture_idx, alt_depth);
    }
}

void common_speculative_accept(common_speculative * spec, uint16_t n_accepted) {
    if (n_accepted == 0) {
        return;
    }

    common_speculative_state * impl = spec->curr_impl;

    GGML_ASSERT(impl);

    {
        common_time_meas tm(impl->t_accept_us, !impl->gen_perf);
        if (n_accepted > 0) {
            impl->n_acc_drafts++;
            impl->n_acc_tokens += n_accepted;
        }

        impl->accept(n_accepted);
        impl->n_call_accept++;
    }
}

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }

    int32_t n_max = 0;
    for (const auto & impl : spec->impls) {
        n_max = std::max(n_max, impl->n_max(params));
    }

    return n_max;
}

int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params) {
    if (spec == nullptr) {
        return 0;
    }

    int32_t n_min = 0;
    for (const auto & impl : spec->impls) {
        n_min = std::max(n_min, impl->n_min(params));
    }

    return n_min;
}

void common_speculative_print_stats(const common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }

    for (const auto & impl : spec->impls) {
        std::string str_perf;
        if (impl->gen_perf) {
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(3) << impl->t_begin_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_draft_us / 1000.0 << ", ";
            oss << std::fixed << std::setprecision(3) << impl->t_accept_us / 1000.0;
            str_perf = ", dur(b,g,a) = " + oss.str() + " ms";
        } else {
            str_perf = "";
        }

        LOG_INF("statistics %s: #calls(b,g,a) = %zu %zu %zu, #gen drafts = %zu, #acc drafts = %zu, #gen tokens = %zu, #acc tokens = %zu%s\n",
                common_speculative_type_to_str(impl->type).c_str(),
                impl->n_call_begin, impl->n_call_draft, impl->n_call_accept,
                impl->n_gen_drafts,
                impl->n_acc_drafts,
                impl->n_gen_tokens,
                impl->n_acc_tokens,
                str_perf.c_str());
    }
}
