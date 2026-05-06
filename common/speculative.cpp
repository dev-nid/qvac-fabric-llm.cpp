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

    // DDTree (DFlash Phase 2 Stage B): tree-shaped variant of gen_draft.
    // Default impl returns an empty tree (= "tree mode unsupported on this
    // speculative state"). Overridden by COMMON_SPECULATIVE_TYPE_DFLASH.
    virtual common_speculative_tree gen_draft_tree(
            const common_speculative_params & params,
            const llama_tokens &              prompt_tgt,
            llama_token                       id_last) {
        (void) params;
        (void) prompt_tgt;
        (void) id_last;
        return {};
    }

    // DDTree Phase 2 Stage C — alt-accept fast path (commit 35).
    // Default impl is a no-op (only DFLASH overrides). See public C wrapper
    // `common_speculative_record_alt_accept` in `speculative.h` for full
    // documentation.
    virtual void record_alt_accept(int alt_capture_idx, int alt_depth) {
        (void) alt_capture_idx;
        (void) alt_depth;
    }
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
        // We read the predictions from the in-graph top-K exposed by
        // `llama_get_dflash_draft_topk(ctx_dft, &n, &K)` (K int32s per
        // draft position, sorted descending so [i*K+0] is the argmax).
        // The decoder graph (`llm_build_dflash`) intentionally skips the
        // float-logits read-back, so `llama_get_logits_ith()` would return
        // nullptr here.
        //
        // For Phase 1 (this commit) we only consume the [i*K+0] column
        // (the argmax) regardless of K, so chain-mode behavior is bit-
        // identical to before. Phase 2 (DDTree) will use the K-1 alternate
        // candidates per position to build K parallel verify chains.
        int64_t  n_topk_outputs = 0;
        uint32_t topk_K         = 0;
        const int32_t * draft_topk = llama_get_dflash_draft_topk(ctx_dft, &n_topk_outputs, &topk_K);
        if (draft_topk == nullptr || n_topk_outputs < block_size || topk_K < 1) {
            LOG_ERR("%s: draft top-K buffer missing or too small (got n=%lld K=%u, need n>=%d K>=1)\n",
                    __func__, (long long) n_topk_outputs, topk_K, block_size);
            return result;
        }
        result.reserve(block_size - 1);
        for (int i = 1; i < block_size; ++i) {
            result.push_back((llama_token) draft_topk[(size_t) i * topk_K + 0]);
        }

        // Reset the draft cache so the next iteration starts from a clean
        // slate (the side store carries the persistent state).
        crop_kv_cache(ctx_dft, /*pos_min=*/0);
        return result;
    }

    // DDTree Phase 2 Stage B: produce a tree of likely continuations
    // built from the draft's per-position top-K. Same draft compute as
    // gen_draft (one DFlash forward pass over [id_last, MASK*15]); only
    // the post-processing differs (tree builder consumes draft_topk
    // directly, doesn't flatten to a chain).
    common_speculative_tree gen_draft_tree(
            const common_speculative_params & params,
            const llama_tokens &              prompt_tgt,
            llama_token                       id_last) override {
        common_speculative_tree tree;
        if (target_layer_ids.empty()) {
            LOG_ERR("%s: draft has no target_layer_ids — cannot run DFlash speculation\n", __func__);
            return tree;
        }

        const int block_size = resolve_block_size(params.n_draft);
        if (block_size <= 1) {
            return tree;
        }

        // Bring the K/V side store up to date with whatever was committed
        // since the previous gen_draft / gen_draft_tree call. Mirrors the
        // first half of gen_draft (above) verbatim — same invariants, same
        // error reporting.
        const int64_t n_committed_after = (int64_t) prompt_tgt.size();
        const int64_t n_to_extend       = n_committed_after - n_committed_total;
        if (n_to_extend < 0) {
            LOG_ERR("%s: prompt_tgt shrank below n_committed_total (%lld < %lld)\n",
                    __func__, (long long) n_committed_after, (long long) n_committed_total);
            return tree;
        }
        if (n_to_extend > 0) {
            if (!extend_side_store(n_to_extend, /*pos_start=*/n_committed_total)) {
                return tree;
            }
            n_committed_total = n_committed_after;
        }

        // Run the draft once on [id_last, MASK, ..., MASK].
        crop_kv_cache(ctx_dft, /*pos_min=*/0);

        const llama_pos pos_block_start = (llama_pos) n_committed_total;
        common_batch_clear(batch_dft);
        for (int i = 0; i < block_size; ++i) {
            const llama_token tok = (i == 0) ? id_last : mask_token_id;
            common_batch_add(batch_dft, tok, pos_block_start + i, { 0 }, /*logits=*/true);
        }
        if (llama_decode(ctx_dft, batch_dft) != 0) {
            LOG_ERR("%s: draft llama_decode failed\n", __func__);
            return tree;
        }

        // Read the top-K candidates per draft position. K must be >= 2 for
        // any tree shape with alternates to be buildable; if the caller
        // forgot to set --dflash-topk we still return a degenerate
        // chain-only tree (which the spec-simple branch will then verify
        // exactly like chain mode, just with the tree-mask plumbing
        // exercised).
        int64_t  n_topk_outputs = 0;
        uint32_t topk_K         = 0;
        const int32_t * draft_topk = llama_get_dflash_draft_topk(ctx_dft, &n_topk_outputs, &topk_K);
        if (draft_topk == nullptr || n_topk_outputs < block_size || topk_K < 1) {
            LOG_ERR("%s: draft top-K buffer missing or too small (got n=%lld K=%u, need n>=%d K>=1)\n",
                    __func__, (long long) n_topk_outputs, topk_K, block_size);
            return tree;
        }

        // Resolve the tree budget. budget == 0 (default when --dflash-tree
        // is set without --dflash-tree-budget B) → Stage B default (chain
        // seed + 1 alt at depth 1 = block_size nodes).
        const int budget = (params.dflash_tree_budget > 0)
                         ? params.dflash_tree_budget
                         : block_size;
        build_ddtree_tree(draft_topk, block_size, topk_K, budget, tree);

        crop_kv_cache(ctx_dft, /*pos_min=*/0);
        return tree;
    }

    // DDTree Phase 2 Stage C — alt-accept fast path (commit 35).
    //
    // Stash the (capture_idx, depth) hint for the next extend_side_store
    // so it can remap captures `[0..d-1, capture_idx]` instead of the
    // linear `[0..d]` slice. See speculative.h for the rationale.
    //
    // Idempotent overwrite: if a hint is already pending (the previous
    // iteration didn't consume it via gen_draft_tree → extend_side_store),
    // the new one wins. In the standard driver flow each tree iteration
    // calls gen_draft_tree → extend_side_store at the start, so the hint
    // set at the end of the previous iter is consumed before any new one
    // could be set; the override branch is defensive only.
    void record_alt_accept(int alt_capture_idx, int alt_depth) override {
        pending_alt_capture_idx = alt_capture_idx;
        pending_alt_depth       = alt_depth;
    }

private:
    // Stage C tree shape: chain seed (top-1 at depths 1..bs-1, capped at
    // budget) + uniform round-robin sibling expansion (rank 1, 2, ...,
    // K-1 across all depths) until budget is exhausted. Each alt is a
    // LEAF hanging off the main chain at its depth (parent = main path
    // node at depth-1) and gets a unique branch_id starting at 1; the
    // main chain is branch_id 0.
    //
    // Indexing (0 = implicit root):
    //   0                  : root (id_last); parents[0] = -1; branch=N/A.
    //   1..main_path_len   : main chain. parents[i] = i-1, depth = i,
    //                        branch_id = 0.
    //   main_path_len+1..  : alt leaves. Each alt at depth d has
    //                        parent = main path's depth-(d-1) node
    //                        = index d-1; branch_id = 1, 2, ...
    //
    // Round-robin order is rank-major then depth-major: rank 1 across all
    // depths first, then rank 2 across all depths, etc. Same effective
    // shape as buun's "uniform K alts per depth" fallback, just expressed
    // as a sequence of single-leaf insertions so we can stop exactly at
    // budget instead of always growing in K-1 chunks.
    //
    // For Stage B equivalence (budget = block_size = 16 on the b16
    // checkpoint): chain seed of 15 nodes + 1 alt at depth 1 = 16 nodes,
    // n_branches = 2. Identical to the Stage B builder this replaces.
    //
    // Visibility matrix is computed for compatibility with Stage A's
    // tree-mask path even though the shipped multi-seq Stage B/C path
    // doesn't use it.
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
        tree.n_branches    = 1; // main only; bumped as alts get added

        if (budget <= 0 || block_size <= 1) {
            // Degenerate: return the empty tree (just the implicit root).
            tree.parents.push_back(-1);
            tree.child_maps.emplace_back();
            tree.visibility.assign(1, 1); // root sees itself
            return;
        }

        // Root.
        tree.parents.push_back(-1);
        tree.child_maps.emplace_back();

        // ----- chain seed (main path) -----
        // top-1 at each draft position 1..bs-1, capped at budget.
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

        // ----- alt leaves (Stage C uniform expansion) -----
        // Round-robin add ranks 1..K-1 across depths 1..main_path_len
        // until budget is reached. Each alt is a leaf, parent =
        // main_path[d-1] (= node index d-1 in our 0=root indexing).
        //
        // Skip alts whose token would collide with an existing child of
        // the same parent (== that parent already has this token in the
        // tree at this depth — happens when top-K has duplicates or when
        // top-rank > 0 happens to equal top-0). The check keeps the
        // child_maps insert one-to-one so the accept walk can lookup the
        // unique child by token.
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

        // ----- visibility matrix (Stage A tree-mask compatibility) -----
        const int n = tree.n_nodes + 1;
        tree.visibility.assign((size_t) n * n, 0);
        tree.visibility[0 * n + 0] = 1; // root sees itself
        for (int i = 1; i < n; ++i) {
            const int parent = tree.parents[i];
            for (int j = 0; j < i; ++j) {
                tree.visibility[(size_t) i * n + j] = tree.visibility[(size_t) parent * n + j];
            }
            tree.visibility[(size_t) i * n + i] = 1;
        }
    }

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
    //
    // Stage C alt-accept fast path (commit 35): when a `record_alt_accept`
    // hint is pending, the captures buffer holds the previous tree-decode
    // outputs at indices [0..main_path_len, alt_1, ..., alt_n] but we want
    // to push `[0, 1, ..., d-1, alt_capture_idx]` into the side store
    // (= replace the rejected m_d capture with the accepted alt's at
    // depth d). The remap is done host-side into a small scratch buffer
    // before the llama_dflash_extend call. After consumption, the hint is
    // cleared so subsequent extends use the linear (chain-style) path.
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

        const float * extend_buf = captures;

        if (pending_alt_capture_idx >= 0) {
            // The hint says: replace row (alt_depth) of the linear range
            // with row (alt_capture_idx) from the captures buffer. The
            // first alt_depth rows ([0..d-1]) stay as the chain prefix
            // [id_last, m_1, ..., m_{d-1}].
            const int     d                = pending_alt_depth;
            const int     alt_idx          = pending_alt_capture_idx;
            const int64_t expected_n_keep  = (int64_t) d + 1;
            const size_t  row_floats       = (size_t) n_features;

            if (n_keep != expected_n_keep) {
                LOG_ERR("%s: alt-accept remap: n_keep=%lld but alt_depth+1=%lld (out of sync; "
                        "driver must not call extend_side_store with non-matching n_keep "
                        "after record_alt_accept)\n",
                        __func__, (long long) n_keep, (long long) expected_n_keep);
                return false;
            }
            if (alt_idx >= n_outputs) {
                LOG_ERR("%s: alt-accept remap: alt_capture_idx=%d but only %lld outputs in "
                        "captures buffer\n", __func__, alt_idx, (long long) n_outputs);
                return false;
            }
            if (row_floats == 0) {
                LOG_ERR("%s: alt-accept remap: n_features=0 (capture not installed?)\n", __func__);
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

            // Consume the hint: subsequent extend calls use the linear path.
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

    // DDTree Phase 2 Stage C — alt-accept fast path (commit 35).
    //
    // When the driver's tree-verify accept walk descends into an alt
    // branch, it calls `record_alt_accept(alt_capture_idx, alt_depth)`
    // INSTEAD OF redecoding `[id_last, m_1, ..., m_{d-1}, alt_token]` in
    // seq 0 to repopulate KV+captures. The fields below stash the hint;
    // the next gen_draft_tree → extend_side_store consumes them by
    // remapping the captures buffer offsets. -1 means "no hint pending"
    // (the linear chain-prefix path applies). See extend_side_store for
    // the consumption logic.
    int                   pending_alt_capture_idx = -1;
    int                   pending_alt_depth       = 0;

    // Scratch buffer used by extend_side_store to materialise the
    // remapped captures rows for an alt-accept extend. Sized lazily to
    // `(alt_depth + 1) * n_features` floats; reused across iterations.
    std::vector<float>    alt_remap_buf;
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

common_speculative_tree common_speculative_gen_draft_tree(
        struct common_speculative * spec,
        struct common_speculative_params params,
        const llama_tokens & prompt_tgt_main_model,
        llama_token id_last) {
    GGML_ASSERT(spec != nullptr && spec->state != nullptr);
    return spec->state->gen_draft_tree(params, prompt_tgt_main_model, id_last);
}

void common_speculative_record_alt_accept(
        struct common_speculative * spec,
        int                          alt_capture_idx,
        int                          alt_depth) {
    GGML_ASSERT(spec != nullptr && spec->state != nullptr);
    spec->state->record_alt_accept(alt_capture_idx, alt_depth);
}
