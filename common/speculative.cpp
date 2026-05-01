#include "speculative.h"

#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "common.h"
#include "sampling.h"

#include <cstring>
#include <algorithm>
#include <map>
#include <memory>

#define SPEC_VOCAB_MAX_SIZE_DIFFERENCE  128
#define SPEC_VOCAB_CHECK_START_TOKEN_ID 5

// =============================================================================
// Polymorphic speculative-decoder state
// =============================================================================
//
// `common_speculative` is a thin wrapper holding an algorithm-specific state
// (DRAFT or DFLASH) and a few cross-algorithm bits (vocab compatibility flag,
// vocab replacement map). Each algorithm subclass owns its own batches,
// samplers, and book-keeping.
//
// The original llama.cpp `common_speculative` was an unparameterized
// draft-model implementation. That implementation moved verbatim into
// `common_speculative_state_draft`; the new DFlash implementation lives
// in `common_speculative_state_dflash`.
//

struct common_speculative_state {
    explicit common_speculative_state(common_speculative_type type) : type(type) {}
    virtual ~common_speculative_state() = default;

    common_speculative_type type;

    // One-shot prompt prefill on ctx_tgt. Returns true on success.
    virtual bool target_prefill(const llama_tokens & prompt) = 0;

    // Generate draft tokens for the next verify step. The returned tokens
    // are appended to the verify batch on top of `id_last`.
    virtual llama_tokens gen_draft(
            const common_speculative_params & params,
            const llama_tokens &              prompt_tgt,
            llama_token                       id_last) = 0;
};

// =============================================================================
// COMMON_SPECULATIVE_TYPE_DRAFT — small, vocab-compatible draft model
// =============================================================================
//
// The original llama.cpp speculative implementation: the draft model is a
// regular causal LM (Qwen3-0.6B drafting Qwen3-8B etc.); each call to
// gen_draft() greedy-samples up to n_draft tokens, stopping early when the
// draft's confidence drops below `p_min`.

struct common_speculative_state_draft : public common_speculative_state {
    common_speculative_state_draft(llama_context * ctx_tgt, llama_context * ctx_dft)
        : common_speculative_state(COMMON_SPECULATIVE_TYPE_DRAFT)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
        , batch(llama_batch_init(llama_n_batch(ctx_dft), 0, 1))
    {
        common_params_sampling sparams;
        sparams.no_perf  = false;
        sparams.top_k    = 10;
        sparams.samplers = { COMMON_SAMPLER_TYPE_TOP_K };

        smpl = common_sampler_init(llama_get_model(ctx_dft), sparams);
    }

    ~common_speculative_state_draft() override {
        common_sampler_free(smpl);
        llama_batch_free(batch);
    }

    bool target_prefill(const llama_tokens & prompt) override {
        // The DRAFT path matches the original llama.cpp speculative behaviour:
        // decode prompt[0..n-1] on the target with `batch_get_one`, leaving the
        // last prompt token to be consumed by the first verify batch.
        if (prompt.empty()) {
            return true;
        }
        const int n_to_decode = (int) prompt.size() - 1;
        if (n_to_decode <= 0) {
            return true;
        }
        return llama_decode(ctx_tgt,
                            llama_batch_get_one(const_cast<llama_token *>(prompt.data()), n_to_decode)) == 0;
    }

    llama_tokens gen_draft(
            const common_speculative_params & params,
            const llama_tokens &              prompt_tgt_main_model,
            llama_token                       id_last) override {
        // -----------------------------------------------------------------
        // Verbatim port of the legacy `common_speculative_gen_draft`:
        // build a draft batch on the draft context, optionally retokenise
        // when vocabs differ, sample up to n_draft tokens. The loop reuses
        // as much of the draft's KV cache as the prefix overlap allows.
        // -----------------------------------------------------------------
        auto * mem_dft = llama_get_memory(ctx_dft);

        int reuse_i = 0;
        int reuse_n = 0;

        const int n_ctx = llama_n_ctx(ctx_dft) - params.n_draft;

        llama_tokens prompt_tgt_draft_model;
        if (!vocab_dft_compatible) {
            std::string text;
            text = common_detokenize(ctx_tgt, prompt_tgt_main_model, true);
            text = replace_to_dft(text);
            LOG_DBG("%s: main->draft detokenized string: '%s'\n", __func__, text.c_str());
            prompt_tgt_draft_model = common_tokenize(ctx_dft, text, false, true);

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

        const llama_tokens & prompt_tgt =
            vocab_dft_compatible ? prompt_tgt_main_model : prompt_tgt_draft_model;

        const int i_start = std::max<int>(0, (int) prompt_tgt.size() - n_ctx);

        for (int i = 0; i < (int) prompt_dft.size(); ++i) {
            int cur = 0;
            while (i_start + cur < (int) prompt_tgt.size() &&
                   i       + cur < (int) prompt_dft.size() &&
                   prompt_tgt[i_start + cur] == prompt_dft[i + cur]) {
                cur++;
            }

            if ((cur >= params.n_reuse || n_ctx >= (int) prompt_tgt.size()) && cur > reuse_n) {
                reuse_i = i;
                reuse_n = cur;
            }
        }

        LOG_DBG("%s: reuse_i = %d, reuse_n = %d, prompt = %d\n", __func__, reuse_i, reuse_n, (int) prompt_dft.size());

        llama_tokens result;
        result.reserve(params.n_draft);

        if (reuse_n == 0) {
            llama_memory_clear(mem_dft, false);
            prompt_dft.clear();
        } else {
            // a previous draft was discarded but the target agreed with it —
            // pass back the saved suffix to save compute.
            if (reuse_i + reuse_n < (int) prompt_dft.size() && prompt_dft[reuse_i + reuse_n] == id_last) {
                for (int i = reuse_i + reuse_n + 1; i < (int) prompt_dft.size(); ++i) {
                    result.push_back(prompt_dft[i]);
                    if (params.n_draft <= (int) result.size()) {
                        break;
                    }
                }
                return result;
            }

            if (reuse_i > 0) {
                llama_memory_seq_rm (mem_dft, 0, 0, reuse_i);
                llama_memory_seq_add(mem_dft, 0, reuse_i, -1, -reuse_i);
                prompt_dft.erase(prompt_dft.begin(), prompt_dft.begin() + reuse_i);
            }

            if (reuse_n < (int) prompt_dft.size()) {
                llama_memory_seq_rm(mem_dft, 0, reuse_n, -1);
                prompt_dft.erase(prompt_dft.begin() + reuse_n, prompt_dft.end());
            }
        }

        common_batch_clear(batch);
        for (size_t i = i_start + reuse_n; i < prompt_tgt.size(); ++i) {
            common_batch_add(batch, prompt_tgt[i], i - i_start, { 0 }, false);
            prompt_dft.push_back(prompt_tgt[i]);
        }

        if (batch.n_tokens > 0) {
            llama_decode(ctx_dft, batch);
        }

        const llama_pos n_past = prompt_dft.size();
        LOG_DBG("%s: n_past = %d\n", __func__, n_past);

        common_batch_clear(batch);
        common_batch_add  (batch, id_last, n_past, { 0 }, true);
        prompt_dft.push_back(id_last);

        LOG_DBG("%s: draft prompt: %s\n", __func__, string_from(ctx_dft, prompt_dft).c_str());

        llama_decode(ctx_dft, batch);

        common_sampler_reset(smpl);

        for (int i = 0; i < params.n_draft; ++i) {
            common_batch_clear(batch);

            common_sampler_sample(smpl, ctx_dft, 0, true);
            const auto * cur_p = common_sampler_get_candidates(smpl, true);

            for (int k = 0; k < std::min(3, (int) cur_p->size); ++k) {
                LOG_DBG(" - draft candidate %3d, pos %3d: %6d (%8.3f) '%s'\n",
                        k, i, cur_p->data[k].id, cur_p->data[k].p, common_token_to_piece(ctx_dft, cur_p->data[k].id).c_str());
            }

            const llama_token id = cur_p->data[0].id;
            common_sampler_accept(smpl, id, true);
            result.push_back(id);

            if (params.n_draft <= (int) result.size()) {
                break;
            }
            if (cur_p->data[0].p < params.p_min) {
                break;
            }

            common_batch_add(batch, id, n_past + i + 1, { 0 }, true);
            llama_decode(ctx_dft, batch);
            prompt_dft.push_back(id);
        }

        if (!vocab_dft_compatible) {
            std::string detokenized = common_detokenize(ctx_dft, result, true);
            detokenized = replace_to_tgt(detokenized);
            LOG_DBG("draft->main detokenized string: '%s'\n", detokenized.c_str());
            result = common_tokenize(ctx_tgt, detokenized, false, true);
            if (result.size() > (size_t) params.n_draft) {
                result.resize(params.n_draft);
            }
        }

        return result;
    }

    // Allow the wrapping common_speculative to plumb the vocab-compat result
    // and the replacement map into the state (the wrapper does the cross-
    // algorithm bookkeeping, the draft state does the actual replacement).
    void set_vocab_compatible(bool v)                                                 { vocab_dft_compatible = v; }
    void set_replacements(const std::map<std::string, std::string> & replacements) { tgt_dft_replacements = replacements; }

private:
    std::string replace_to_dft(const std::string & input) const {
        std::string result = input;
        for (const auto & pair : tgt_dft_replacements) {
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
        for (const auto & pair : tgt_dft_replacements) {
            size_t pos = result.find(pair.second);
            while (pos != std::string::npos) {
                result.replace(pos, pair.second.length(), pair.first);
                pos = result.find(pair.second, pos + pair.first.length());
            }
        }
        return result;
    }

    llama_context * ctx_tgt;
    llama_context * ctx_dft;

    common_sampler * smpl   = nullptr;
    llama_batch      batch  {};
    llama_tokens     prompt_dft;

    bool                                vocab_dft_compatible = true;
    std::map<std::string, std::string>  tgt_dft_replacements;
};

// =============================================================================
// COMMON_SPECULATIVE_TYPE_DFLASH — block-parallel draft via the DFlash drafter
// =============================================================================
//
// Algorithm (paper §4 + our K/V side-store optimisation from §4.1):
//
//   target_prefill(prompt):
//     1. reset the draft's per-layer K/V side store
//     2. decode prompt[0..n-1] on the target with logits at every position
//        (DFlash captures hidden states only at output positions, so we have
//        to mark every position as an output)
//     3. extend the side store with all n captures at positions [0..n-1]
//     4. record n_committed_total = n - 1  (matches the spec_simple model
//        where prompt.back() is the still-uncommitted id_last)
//
//   gen_draft(params, prompt_tgt, id_last):
//     1. n_to_extend = prompt_tgt.size() - n_committed_total
//        (= the number of newly-accepted tokens since the previous gen_draft)
//     2. if n_to_extend > 0: pull captures from the most recent target verify
//        decode and extend the side store with the first n_to_extend of them,
//        starting at side-store position n_committed_total
//     3. n_committed_total = prompt_tgt.size()
//     4. run the draft on a block of [id_last, MASK, ..., MASK] at positions
//        [n_committed_total, n_committed_total+1, ..., n_committed_total+bs-1]
//     5. argmax-decode the per-position draft logits to get bs-1 tokens
//        (positions 1..bs-1 are the predictions; position 0 is the anchor)
//
// Side-store invariant after each gen_draft:
//   side store contains pre-projected K/V for positions [0..n_committed-1],
//   where n_committed is the number of tokens that have been COMMITTED on
//   the target *and* whose target hidden states have been captured.
//   The current `id_last` is at position n_committed and its capture has
//   *not* been pushed to the side store yet — the draft uses its own
//   embedding/layers to produce its representation, exactly as in the
//   stand-alone driver.

struct common_speculative_state_dflash : public common_speculative_state {
    common_speculative_state_dflash(llama_context * ctx_tgt, llama_context * ctx_dft)
        : common_speculative_state(COMMON_SPECULATIVE_TYPE_DFLASH)
        , ctx_tgt(ctx_tgt)
        , ctx_dft(ctx_dft)
        , batch_tgt(llama_batch_init(llama_n_batch(ctx_tgt), 0, 1))
        , batch_dft(llama_batch_init(llama_n_batch(ctx_dft), 0, 1))
    {
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

        // Paper §4.2: bind target's tok_embd / lm_head into the draft. This
        // is idempotent and harmless if the driver already bound them before
        // creating ctx_dft (which it must, because graph_reserve runs at
        // context construction). Calling here too is purely defensive.
        if (!llama_dflash_bind_target(const_cast<llama_model *>(model_dft), model_tgt)) {
            LOG_DBG("%s: llama_dflash_bind_target returned false (draft is not LLM_ARCH_DFLASH "
                    "or one of the models is null). The graph builder will fall back to "
                    "self-contained tensors.\n", __func__);
        }

        // Install per-layer hidden-state capture on the target context so every
        // decode tees out the layers the draft was trained against.
        if (!target_layer_ids.empty()) {
            llama_set_dflash_capture(ctx_tgt,
                                     target_layer_ids.data(),
                                     target_layer_ids.size(),
                                     n_embd_target);
        }

        LOG_INF("%s: dflash spec initialised: block_size=%u, mask_token=%d, n_features=%lld, "
                "target_layer_ids=[", __func__,
                block_size_default, (int) mask_token_id, (long long) n_features);
        for (size_t i = 0; i < target_layer_ids.size(); ++i) {
            LOG_INF("%s%d", i == 0 ? "" : ",", (int) target_layer_ids[i]);
        }
        LOG_INF("]\n");
    }

    ~common_speculative_state_dflash() override {
        if (ctx_tgt != nullptr) {
            llama_set_dflash_capture(ctx_tgt, nullptr, 0, 0);
        }
        llama_batch_free(batch_tgt);
        llama_batch_free(batch_dft);
    }

    bool target_prefill(const llama_tokens & prompt) override {
        if (target_layer_ids.empty()) {
            LOG_ERR("%s: draft has no target_layer_ids — cannot run DFlash speculation\n", __func__);
            return false;
        }
        if (prompt.empty()) {
            n_committed_total = 0;
            return true;
        }

        // Paper §4.1: start each prompt with a clean K/V side store.
        llama_dflash_reset_ctx_kv(ctx_dft);

        // Decode prompt[0..n-1] on the target with logits at every position.
        // This is the only place in the entire DFlash pipeline that needs an
        // all-positions-logits batch — the spec_simple loop already requests
        // logits at every verify position, so capture coverage holds there.
        const int n_to_decode = (int) prompt.size() - 1;
        if (n_to_decode <= 0) {
            // Single-token prompt — nothing to commit, the verify loop will
            // decode it. (We still need n_committed_total == 0 so the next
            // gen_draft picks up captures correctly.)
            n_committed_total = 0;
            return true;
        }

        common_batch_clear(batch_tgt);
        for (int i = 0; i < n_to_decode; ++i) {
            common_batch_add(batch_tgt, prompt[i], (llama_pos) i, { 0 }, /*logits=*/true);
        }
        if (llama_decode(ctx_tgt, batch_tgt) != 0) {
            LOG_ERR("%s: target prompt prefill failed\n", __func__);
            return false;
        }

        // Push the prompt-wide captures through the DFlash encoder so the
        // draft can read pre-projected K_ctx / V_ctx for [0..n_to_decode-1]
        // from its side store on the very first gen_draft.
        if (!extend_side_store(n_to_decode, /*pos_start=*/0)) {
            return false;
        }
        n_committed_total = n_to_decode;
        return true;
    }

    llama_tokens gen_draft(
            const common_speculative_params & params,
            const llama_tokens &              prompt_tgt,
            llama_token                       id_last) override {
        llama_tokens result;
        if (target_layer_ids.empty()) {
            LOG_ERR("%s: draft has no target_layer_ids — cannot run DFlash speculation\n", __func__);
            return result;
        }

        const int block_size = resolve_block_size(params.n_draft);
        if (block_size <= 1) {
            return result;
        }

        // ----------------------------------------------------------------
        // Step 1: bring the K/V side store up to date with whatever the
        // caller's verify loop accepted since the previous call. The
        // captures from the most recent ctx_tgt decode cover all bs verify
        // positions; we keep the first `n_to_extend` (= the accepted prefix
        // including the still-uncommitted id_last that anchored it).
        // ----------------------------------------------------------------
        const int64_t n_committed_after = (int64_t) prompt_tgt.size();
        const int64_t n_to_extend       = n_committed_after - n_committed_total;
        if (n_to_extend < 0) {
            LOG_ERR("%s: prompt_tgt shrank below n_committed_total (%lld < %lld) — caller "
                    "must not roll back without recreating the spec state\n",
                    __func__, (long long) n_committed_after, (long long) n_committed_total);
            return result;
        }
        if (n_to_extend > 0) {
            if (!extend_side_store(n_to_extend, /*pos_start=*/n_committed_total)) {
                return result;
            }
            n_committed_total = n_committed_after;
        }

        // ----------------------------------------------------------------
        // Step 2: run the draft on a block [id_last, MASK, ..., MASK] at
        // absolute positions [n_committed_total..n_committed_total+bs-1].
        // The DFlash decoder cross-attends to the side store (positions
        // 0..n_committed_total-1) and self-attends bidirectionally inside
        // the block. We reset the draft's KV cache between blocks because
        // the side store already encodes the committed prefix; a stale
        // draft KV would otherwise leak rejected positions into the next
        // block's attention.
        // ----------------------------------------------------------------
        crop_kv_cache(ctx_dft, /*pos_min=*/0);

        const llama_pos pos_block_start = (llama_pos) n_committed_total;

        common_batch_clear(batch_dft);
        for (int i = 0; i < block_size; ++i) {
            const llama_token tok = (i == 0) ? id_last : mask_token_id;
            common_batch_add(batch_dft, tok, pos_block_start + i, { 0 }, /*logits=*/true);
        }
        if (llama_decode(ctx_dft, batch_dft) != 0) {
            LOG_ERR("%s: draft llama_decode failed\n", __func__);
            return result;
        }

        // DFlash uses bidirectional intra-block attention (paper §3.1): the
        // draft logits at intra-block position i predict the token AT
        // position i (not position i+1 as in a regular causal LM). Position
        // 0 is the anchor (input token == id_last == prediction at pos 0),
        // so we read predictions at positions [1..bs-1] and return bs-1
        // tokens.
        //
        // The verify batch built by the spec_simple driver places id_last at
        // ctx position n_past and the i-th draft token at ctx position
        // n_past+1+i. The target's CAUSAL logits at ctx position n_past+i
        // therefore predict the token at ctx position n_past+i+1 — which is
        // what the i-th draft token claims. The off-by-one between the
        // standalone DFlash driver (which puts id_last at intra-block 0 and
        // draft_tokens[1..bs-1] at intra-block [1..bs-1] in BOTH the draft
        // and verify batches) and our spec-simple integration is therefore
        // absorbed into the indexing here.
        //
        // We read the predictions from the in-graph greedy argmax exposed
        // by `llama_get_dflash_draft_argmax(ctx_dft)` (one int32 per draft
        // position). The decoder graph (`llm_build_dflash`) intentionally
        // skips the float-logits read-back, so `llama_get_logits_ith()`
        // would return nullptr here.
        int64_t n_argmax_outputs = 0;
        const int32_t * draft_argmax = llama_get_dflash_draft_argmax(ctx_dft, &n_argmax_outputs);
        if (draft_argmax == nullptr || n_argmax_outputs < block_size) {
            LOG_ERR("%s: draft argmax buffer missing or too small (got %lld, need >= %d)\n",
                    __func__, (long long) n_argmax_outputs, block_size);
            return result;
        }
        result.reserve(block_size - 1);
        for (int i = 1; i < block_size; ++i) {
            result.push_back((llama_token) draft_argmax[i]);
        }

        // Reset the draft cache so the next iteration starts from a clean
        // slate (the side store carries the persistent state).
        crop_kv_cache(ctx_dft, /*pos_min=*/0);
        return result;
    }

private:
    // Clamp the user-requested block_size against the draft's trained value.
    // Per paper §5.4.4 we refuse `inference_block_size > trained_block_size`
    // (the un-generalisable direction).
    int resolve_block_size(int requested) const {
        const int trained = (int) block_size_default;
        if (requested <= 0) {
            return trained;
        }
        if (requested > trained) {
            LOG_ERR("%s: requested DFlash block_size=%d exceeds trained block_size=%d. "
                    "Per paper §5.4.4 this is the un-generalisable direction; use "
                    "--draft-max %d or smaller, or train a larger DFlash draft.\n",
                    __func__, requested, trained, trained);
            return -1;
        }
        return requested;
    }

    // Read freshly-captured target features from ctx_tgt and push the first
    // `n_keep` of them into the draft's per-layer K/V side store at offset
    // `pos_start`. Wraps llama_dflash_extend with a friendlier error path.
    bool extend_side_store(int64_t n_keep, int64_t pos_start) {
        int64_t n_outputs = 0;
        const float * captures = llama_get_dflash_captured_features(ctx_tgt, &n_outputs);
        if (captures == nullptr || n_outputs == 0) {
            LOG_ERR("%s: target captured no features (capture not installed or no outputs requested)\n", __func__);
            return false;
        }
        if (n_keep > n_outputs) {
            LOG_ERR("%s: asked to keep %lld captures but target produced only %lld\n",
                    __func__, (long long) n_keep, (long long) n_outputs);
            return false;
        }
        const int32_t rc = llama_dflash_extend(ctx_dft, captures, n_keep, pos_start);
        if (rc != 0) {
            LOG_ERR("%s: llama_dflash_extend failed (rc=%d)\n", __func__, rc);
            return false;
        }
        return true;
    }

    static void crop_kv_cache(llama_context * ctx, llama_pos pos_min) {
        llama_memory_seq_rm(llama_get_memory(ctx), 0, pos_min, -1);
    }

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
    // the DFlash encoder into the draft's per-layer K/V side store. Equal
    // to `prompt_tgt.size()` after each successful gen_draft.
    int64_t               n_committed_total  = 0;
};

// =============================================================================
// common_speculative wrapper
// =============================================================================

struct common_speculative {
    llama_context *                              ctx_tgt;
    llama_context *                              ctx_dft;
    std::unique_ptr<common_speculative_state>    state;

    bool                                         vocab_dft_compatible = true;
    std::map<std::string, std::string>           tgt_dft_replacements;
};

bool common_speculative_are_compatible(
    const struct llama_context * ctx_tgt,
    const struct llama_context * ctx_dft) {
    const struct llama_model * model_tgt = llama_get_model(ctx_tgt);
    const struct llama_model * model_dft = llama_get_model(ctx_dft);

    const struct llama_vocab * vocab_tgt = llama_model_get_vocab(model_tgt);
    const struct llama_vocab * vocab_dft = llama_model_get_vocab(model_dft);

    const bool vocab_type_tgt = llama_vocab_type(vocab_tgt);
    LOG_DBG("%s: vocab_type tgt: %d\n", __func__, vocab_type_tgt);

    const bool vocab_type_dft = llama_vocab_type(vocab_dft);
    LOG_DBG("%s: vocab_type dft: %d\n", __func__, vocab_type_dft);

    if (vocab_type_tgt != vocab_type_dft) {
        LOG_DBG("%s: draft model vocab type must match target model to use speculation but ", __func__);
        LOG_DBG("vocab_type_dft = %d while vocab_type_tgt = %d\n", vocab_type_dft, vocab_type_tgt);
        return false;
    }

    if (
        llama_vocab_get_add_bos(vocab_tgt) != llama_vocab_get_add_bos(vocab_dft) ||
        llama_vocab_get_add_eos(vocab_tgt) != llama_vocab_get_add_eos(vocab_dft) ||
        llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)
    ) {
        LOG_DBG("%s: draft model special tokens must match target model to use speculation\n", __func__);
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
                        common_token_to_piece(ctx_tgt, i).c_str(),
                        common_token_to_piece(ctx_dft, i).c_str());
                return false;
            }
        }
    }

    return true;
}

struct common_speculative * common_speculative_init_typed(
        struct llama_context *      ctx_tgt,
        struct llama_context *      ctx_dft,
        enum common_speculative_type type) {
    if (ctx_tgt == nullptr || ctx_dft == nullptr) {
        LOG_ERR("%s: ctx_tgt and ctx_dft must both be non-null\n", __func__);
        return nullptr;
    }

    // Resolve AUTO. Pick DFLASH iff the draft GGUF carries DFlash metadata.
    enum common_speculative_type resolved = type;
    const llama_model * model_dft = llama_get_model(ctx_dft);
    const bool draft_is_dflash = llama_model_dflash_block_size(model_dft) > 0;

    if (resolved == COMMON_SPECULATIVE_TYPE_AUTO) {
        resolved = draft_is_dflash
            ? COMMON_SPECULATIVE_TYPE_DFLASH
            : COMMON_SPECULATIVE_TYPE_DRAFT;
        LOG_INF("%s: --draft-type auto resolved to '%s' "
                "(draft model %s DFlash GGUF metadata)\n", __func__,
                resolved == COMMON_SPECULATIVE_TYPE_DFLASH ? "dflash" : "draft",
                draft_is_dflash ? "carries" : "does not carry");
    }

    if (resolved == COMMON_SPECULATIVE_TYPE_DFLASH && !draft_is_dflash) {
        LOG_ERR("%s: --draft-type dflash requires a DFlash draft GGUF "
                "(missing dflash.block_size metadata)\n", __func__);
        return nullptr;
    }

    auto * result = new common_speculative {
        /* .ctx_tgt              = */ ctx_tgt,
        /* .ctx_dft              = */ ctx_dft,
        /* .state                = */ nullptr,
        /* .vocab_dft_compatible = */ false,
        /* .tgt_dft_replacements = */ {},
    };

    result->vocab_dft_compatible = common_speculative_are_compatible(ctx_tgt, ctx_dft);
    LOG_DBG("vocab_dft_compatible = %d\n", result->vocab_dft_compatible);

    if (resolved == COMMON_SPECULATIVE_TYPE_DFLASH) {
        if (!result->vocab_dft_compatible) {
            LOG_ERR("%s: --draft-type dflash requires a vocab-compatible draft "
                    "(retokenisation between target and DFlash draft is not supported)\n", __func__);
            delete result;
            return nullptr;
        }
        result->state = std::make_unique<common_speculative_state_dflash>(ctx_tgt, ctx_dft);
    } else {
        auto draft_state = std::make_unique<common_speculative_state_draft>(ctx_tgt, ctx_dft);
        // Plumb the cross-algorithm bits into the DRAFT state so it can do
        // its retokenisation work the way the legacy implementation did.
        draft_state->set_vocab_compatible(result->vocab_dft_compatible);
        result->state = std::move(draft_state);
    }

    return result;
}

struct common_speculative * common_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft) {
    return common_speculative_init_typed(ctx_tgt, ctx_dft, COMMON_SPECULATIVE_TYPE_AUTO);
}

void common_speculative_free(struct common_speculative * spec) {
    if (spec == nullptr) {
        return;
    }
    delete spec;
}

void common_speculative_add_replacement_tgt_dft(
        struct common_speculative * spec,
        const char * source, const char * dest) {
    spec->tgt_dft_replacements[source] = dest;
    // Forward to the DRAFT state so its detokenize/retokenize loop sees them.
    if (spec->state && spec->state->type == COMMON_SPECULATIVE_TYPE_DRAFT) {
        static_cast<common_speculative_state_draft *>(spec->state.get())
            ->set_replacements(spec->tgt_dft_replacements);
    }
}

bool common_speculative_target_prefill(
        struct common_speculative * spec,
        const llama_tokens &        prompt) {
    GGML_ASSERT(spec != nullptr && spec->state != nullptr);
    return spec->state->target_prefill(prompt);
}

enum common_speculative_type common_speculative_get_type(
        const struct common_speculative * spec) {
    GGML_ASSERT(spec != nullptr && spec->state != nullptr);
    return spec->state->type;
}

llama_tokens common_speculative_gen_draft(
        struct common_speculative * spec,
        struct common_speculative_params params,
        const llama_tokens & prompt_tgt_main_model,
        llama_token id_last) {
    GGML_ASSERT(spec != nullptr && spec->state != nullptr);
    return spec->state->gen_draft(params, prompt_tgt_main_model, id_last);
}
