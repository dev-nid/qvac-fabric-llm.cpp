#include "dflash-speculative.h"

#include "ggml.h"
#include "llama.h"
#include "log.h"
#include "common.h"
#include "sampling.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <vector>

// -------------------------------------------------------------------------
// Internal state
// -------------------------------------------------------------------------

struct common_dflash_speculative {
    struct llama_context * ctx_tgt = nullptr;
    struct llama_context * ctx_dft = nullptr;

    // DFlash hparams (cached from the draft model's GGUF)
    uint32_t              block_size_default = 0;
    llama_token           mask_token_id      = LLAMA_TOKEN_NULL;
    std::vector<int32_t>  target_layer_ids;

    // Per-token feature dimension = n_target_layer_ids * n_embd_target.
    // Cached to avoid recomputing on every loop iteration.
    int64_t n_features      = 0;
    int64_t n_embd_target   = 0;

    // Persistent host-side accumulator of captured target hidden states for
    // every committed token (prompt + accepted draft tokens + bonus tokens).
    // Layout: row-major, [committed_n_tokens, n_features]
    //   accumulated[i_token * n_features + i_feat]
    // Grows as tokens are committed; truncated when rejected drafts are
    // dropped. Passed to the draft via llama_set_dflash_input.
    std::vector<float> accumulated;
    int64_t            accumulated_n = 0;

    llama_batch batch_tgt {};
    llama_batch batch_dft {};
};

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

static double now_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

static void crop_kv_cache(llama_context * ctx, llama_pos pos_min) {
    llama_memory_seq_rm(llama_get_memory(ctx), 0, pos_min, -1);
}

static llama_token argmax_logits(const float * logits, int32_t n_vocab) {
    int32_t best_id = 0;
    float   best_lp = logits[0];
    for (int32_t v = 1; v < n_vocab; ++v) {
        if (logits[v] > best_lp) {
            best_lp = logits[v];
            best_id = v;
        }
    }
    return (llama_token) best_id;
}

// Append the first `n_tokens_keep` rows of the freshly-captured features
// (which sit on ctx_tgt after each decode) onto our persistent accumulator.
//
// The captures cover every "output" position in the most recent decode; the
// driver always requests logits at every position so that's all of them, in
// the same order as the input batch.
static bool append_captures(common_dflash_speculative * spec, int64_t n_tokens_keep) {
    int64_t n_outputs = 0;
    const float * captures = llama_get_dflash_captured_features(spec->ctx_tgt, &n_outputs);
    if (captures == nullptr || n_outputs == 0) {
        LOG_ERR("%s: target captured no features (capture not installed or no outputs requested)\n", __func__);
        return false;
    }
    if (n_tokens_keep > n_outputs) {
        LOG_ERR("%s: asked to keep %lld captures but target produced only %lld\n",
                __func__, (long long) n_tokens_keep, (long long) n_outputs);
        return false;
    }

    const size_t n_features = (size_t) spec->n_features;
    const size_t old_size   = spec->accumulated.size();

    spec->accumulated.resize(old_size + (size_t) n_tokens_keep * n_features);
    std::memcpy(spec->accumulated.data() + old_size,
                captures,
                (size_t) n_tokens_keep * n_features * sizeof(float));
    spec->accumulated_n += n_tokens_keep;
    return true;
}

// Roll the accumulator back to exactly `n_keep` tokens.
static void truncate_accumulator(common_dflash_speculative * spec, int64_t n_keep) {
    GGML_ASSERT(n_keep <= spec->accumulated_n);
    spec->accumulated.resize((size_t) n_keep * spec->n_features);
    spec->accumulated_n = n_keep;
}

// -------------------------------------------------------------------------
// Init / free
// -------------------------------------------------------------------------

struct common_dflash_speculative * common_dflash_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft) {
    auto * spec = new common_dflash_speculative();
    spec->ctx_tgt = ctx_tgt;
    spec->ctx_dft = ctx_dft;

    const auto * model_dft = llama_get_model(ctx_dft);
    const auto * model_tgt = llama_get_model(ctx_tgt);

    spec->block_size_default = llama_model_dflash_block_size(model_dft);
    spec->mask_token_id      = llama_model_dflash_mask_token_id(model_dft);
    spec->n_embd_target      = llama_model_n_embd(model_tgt);

    const int n_tlid = llama_model_dflash_n_target_layer_ids(model_dft);
    spec->target_layer_ids.reserve(n_tlid > 0 ? n_tlid : 0);
    for (int i = 0; i < n_tlid; ++i) {
        spec->target_layer_ids.push_back(
            llama_model_dflash_target_layer_id(model_dft, i));
    }
    spec->n_features = (int64_t) spec->target_layer_ids.size() * spec->n_embd_target;

    spec->batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, 1);
    spec->batch_dft = llama_batch_init(llama_n_batch(ctx_dft), 0, 1);

    // Paper §4.2: the DFlash draft shares tok_embd and lm_head with the
    // target. Bind them now; harmless if the draft GGUF was built
    // self-contained (the bind sets the pointers, but the graph builder
    // prefers them only when needed).
    //
    // Note: `model_dft` is non-const here even though it came from a
    // const llama_get_model, because llama_dflash_bind_target mutates the
    // draft's external-tensor pointers. Both reference impls do the same.
    if (!llama_dflash_bind_target(const_cast<llama_model *>(model_dft), model_tgt)) {
        LOG_DBG("%s: llama_dflash_bind_target returned false (draft is not LLM_ARCH_DFLASH "
                "or one of the models is null). Falling back to self-contained tensors.\n",
                __func__);
    }

    // Install per-layer hidden-state capture on the target context so every
    // decode tees out the layers the draft was trained against.
    if (!spec->target_layer_ids.empty()) {
        llama_set_dflash_capture(ctx_tgt,
                                 spec->target_layer_ids.data(),
                                 spec->target_layer_ids.size(),
                                 spec->n_embd_target);
    }

    LOG_INF("%s: dflash draft initialised: block_size=%u, mask_token=%d, n_features=%lld, target_layer_ids=[",
        __func__, spec->block_size_default, (int) spec->mask_token_id, (long long) spec->n_features);
    for (size_t i = 0; i < spec->target_layer_ids.size(); ++i) {
        LOG_INF("%s%d", i == 0 ? "" : ",", (int) spec->target_layer_ids[i]);
    }
    LOG_INF("]\n");

    return spec;
}

void common_dflash_speculative_free(struct common_dflash_speculative * spec) {
    if (spec == nullptr) {
        return;
    }
    // Detach capture from the target context.
    if (spec->ctx_tgt != nullptr) {
        llama_set_dflash_capture(spec->ctx_tgt, nullptr, 0, 0);
    }
    llama_batch_free(spec->batch_tgt);
    llama_batch_free(spec->batch_dft);
    delete spec;
}

bool common_dflash_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft) {
    const auto * model_tgt = llama_get_model(ctx_tgt);
    const auto * model_dft = llama_get_model(ctx_dft);

    if (llama_model_dflash_block_size(model_dft) == 0) {
        LOG_ERR("%s: draft model does not appear to be a DFlash draft "
                "(missing dflash.block_size metadata)\n", __func__);
        return false;
    }

    const auto * vocab_tgt = llama_model_get_vocab(model_tgt);
    const auto * vocab_dft = llama_model_get_vocab(model_dft);

    if (llama_vocab_n_tokens(vocab_tgt) != llama_vocab_n_tokens(vocab_dft)) {
        LOG_ERR("%s: vocab size mismatch: tgt=%d, dft=%d\n", __func__,
            llama_vocab_n_tokens(vocab_tgt), llama_vocab_n_tokens(vocab_dft));
        return false;
    }

    if (llama_vocab_bos(vocab_tgt) != llama_vocab_bos(vocab_dft) ||
        llama_vocab_eos(vocab_tgt) != llama_vocab_eos(vocab_dft)) {
        LOG_ERR("%s: bos/eos token mismatch between target and draft vocabs\n", __func__);
        return false;
    }

    return true;
}

// -------------------------------------------------------------------------
// Generate
// -------------------------------------------------------------------------

llama_tokens common_dflash_speculative_generate(
        struct common_dflash_speculative * spec,
        struct common_dflash_speculative_params params,
        const llama_tokens & prompt,
        const std::vector<llama_token> & eos_ids,
        struct common_sampler * smpl_tgt,
        struct common_dflash_speculative_callbacks cbs,
        struct common_dflash_speculative_stats * stats_out) {
    GGML_ASSERT(spec != nullptr);
    GGML_ASSERT(smpl_tgt != nullptr);

    auto * ctx_tgt = spec->ctx_tgt;
    auto * ctx_dft = spec->ctx_dft;

    if (spec->target_layer_ids.empty()) {
        LOG_ERR("%s: draft has no target_layer_ids — cannot run DFlash speculation\n", __func__);
        return prompt;
    }

    const int block_size = params.block_size > 0 ? params.block_size : (int) spec->block_size_default;
    if (block_size <= 1) {
        LOG_ERR("%s: invalid block_size %d (must be >= 2)\n", __func__, block_size);
        return prompt;
    }

    // Paper §5.4.4 ablation:
    //   "DFlash models trained with larger block sizes generalize well to
    //    smaller inference-time block sizes. ... However, the reverse does
    //    not hold."
    //
    // Concretely: training at block 16 + inference at block 8 ≈ same as
    // training at block 8 + inference at block 8 (so downsizing is fine);
    // but training at block 8 + inference at block 16 measurably degrades
    // acceptance length. Refuse this case rather than silently producing
    // bad outputs.
    if ((uint32_t) block_size > spec->block_size_default) {
        LOG_ERR("%s: requested inference block_size=%d exceeds the trained block_size=%u "
                "of the DFlash draft. Per the DFlash paper §5.4.4, scaling block size "
                "*up* at inference does not generalize and yields shorter acceptance "
                "lengths than training at that size directly. Use --draft-max %u or smaller, "
                "or train a draft at the desired block size.\n",
                __func__, block_size, spec->block_size_default, spec->block_size_default);
        return prompt;
    }
    if ((uint32_t) block_size < spec->block_size_default) {
        LOG_INF("%s: using block_size=%d (< trained block_size=%u). This is the supported "
                "scaling direction per paper §5.4.4 — acceptance length will be slightly "
                "lower than the trained value but not pathologically so.\n",
                __func__, block_size, spec->block_size_default);
    }

    const int n_max_predict  = params.n_max_predict;
    const auto * model_tgt   = llama_get_model(ctx_tgt);
    const auto * vocab_tgt   = llama_model_get_vocab(model_tgt);
    const int    n_vocab     = llama_vocab_n_tokens(vocab_tgt);
    const llama_token mask_id = spec->mask_token_id;

    llama_tokens output = prompt;
    output.reserve(prompt.size() + n_max_predict + block_size);

    auto eos_or_stop = [&](llama_token id) {
        if (llama_vocab_is_eog(vocab_tgt, id)) {
            return true;
        }
        return std::find(eos_ids.begin(), eos_ids.end(), id) != eos_ids.end();
    };

    // ---------------------------------------------------------------------
    // Step 1: Prefill the target on the prompt.
    // We request logits at *every* position so that DFlash capture covers
    // the whole prompt (n_outputs == n_tokens).
    // ---------------------------------------------------------------------
    spec->accumulated.clear();
    spec->accumulated_n = 0;

    LOG_DBG("%s: prefill target on %d tokens\n", __func__, (int) prompt.size());
    {
        common_batch_clear(spec->batch_tgt);
        for (size_t i = 0; i < prompt.size(); ++i) {
            common_batch_add(spec->batch_tgt, prompt[i], (llama_pos) i, { 0 }, /*logits=*/true);
        }
        if (llama_decode(ctx_tgt, spec->batch_tgt) != 0) {
            LOG_ERR("%s: prefill target llama_decode failed\n", __func__);
            return output;
        }
    }

    // Pull the prompt-wide captures into the persistent accumulator.
    if (!append_captures(spec, (int64_t) prompt.size())) {
        return output;
    }
    GGML_ASSERT(spec->accumulated_n == (int64_t) prompt.size());

    // Sample the very first generated token from the last prompt position.
    llama_token first_id = common_sampler_sample(smpl_tgt, ctx_tgt, /*idx=*/(int) prompt.size() - 1);
    common_sampler_accept(smpl_tgt, first_id, /*accept_grammar=*/true);
    output.push_back(first_id);
    int n_predict = 1;

    if (cbs.on_token && !cbs.on_token(first_id, cbs.user_data)) {
        return output;
    }
    if (eos_or_stop(first_id)) {
        return output;
    }

    // We haven't decoded `first_id` on the target yet — its features will be
    // captured as part of the next verify decode (which always re-decodes
    // the most recently committed token at position 0 of the block).

    // ---------------------------------------------------------------------
    // Step 2: Decode loop.
    // ---------------------------------------------------------------------
    const double t_decode_start = now_seconds();
    int n_blocks  = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // n_committed_post_prompt counts tokens generated past the prompt that are
    // already in the target's KV cache. Initially it's 0 because `first_id`
    // was sampled but has not been re-decoded as a regular position yet.
    // (The next block's verify step will be its first appearance in
    // ctx_tgt's KV cache.)
    int n_committed_post_prompt = 0;

    std::vector<llama_token> draft_tokens(block_size);
    std::vector<llama_token> target_tokens(block_size);

    while (n_predict < n_max_predict) {
        const int bs = std::min<int>(block_size, n_max_predict - n_predict + 1);
        if (bs <= 1) {
            break;
        }

        const int64_t n_committed_total = (int64_t) prompt.size() + n_committed_post_prompt;
        const llama_pos pos_block_start = (llama_pos) n_committed_total;

        // -----------------------------------------------------------------
        // Step 2a: Run the draft on a block of [last_committed, MASK, ..., MASK].
        // The draft KV cache is reset between blocks (the cross-context K/V
        // already encode the committed prefix; a stale draft KV would
        // otherwise leak rejected positions into the next block).
        // -----------------------------------------------------------------
        crop_kv_cache(ctx_dft, 0);

        // Stage the persistent target_hidden buffer on the draft for this block.
        llama_set_dflash_input(ctx_dft,
                               spec->accumulated.empty() ? nullptr : spec->accumulated.data(),
                               spec->n_features,
                               spec->accumulated_n,
                               bs);

        common_batch_clear(spec->batch_dft);
        for (int i = 0; i < bs; ++i) {
            const llama_token tok = (i == 0) ? output.back() : mask_id;
            // Logits at every position so the per-position draft logits are
            // returned (positions 1..bs-1 are the predicted draft tokens).
            common_batch_add(spec->batch_dft, tok, pos_block_start + i, { 0 }, /*logits=*/true);
        }

        if (llama_decode(ctx_dft, spec->batch_dft) != 0) {
            LOG_ERR("%s: draft llama_decode failed\n", __func__);
            break;
        }

        for (int i = 1; i < bs; ++i) {
            const float * dft_logits_i = llama_get_logits_ith(ctx_dft, i);
            if (dft_logits_i == nullptr) {
                LOG_ERR("%s: draft logits missing at position %d\n", __func__, i);
                return output;
            }
            draft_tokens[i] = argmax_logits(dft_logits_i, n_vocab);
        }
        n_drafted += bs - 1;

        // Reset the draft cache; the next iteration will rerun on a fresh slate.
        crop_kv_cache(ctx_dft, 0);

        // -----------------------------------------------------------------
        // Step 2b: Verify the block on the target.
        // We request logits AND captures at every position.
        // -----------------------------------------------------------------
        common_batch_clear(spec->batch_tgt);
        common_batch_add(spec->batch_tgt, output.back(), pos_block_start + 0, { 0 }, /*logits=*/true);
        for (int i = 1; i < bs; ++i) {
            common_batch_add(spec->batch_tgt, draft_tokens[i], pos_block_start + i, { 0 }, /*logits=*/true);
        }

        if (llama_decode(ctx_tgt, spec->batch_tgt) != 0) {
            LOG_ERR("%s: verify llama_decode failed\n", __func__);
            break;
        }

        for (int i = 0; i < bs; ++i) {
            const float * tgt_logits_i = llama_get_logits_ith(ctx_tgt, i);
            if (tgt_logits_i == nullptr) {
                LOG_ERR("%s: target logits missing at position %d\n", __func__, i);
                return output;
            }
            target_tokens[i] = argmax_logits(tgt_logits_i, n_vocab);
        }

        // Acceptance: longest prefix where draft_tokens[i+1] == target_tokens[i].
        int accepted = 0;
        for (int i = 0; i < bs - 1; ++i) {
            if (draft_tokens[i + 1] == target_tokens[i]) {
                accepted++;
            } else {
                break;
            }
        }
        n_accept += accepted;
        n_blocks++;

        // -----------------------------------------------------------------
        // Step 2c: Commit accepted draft tokens + the bonus target token.
        // -----------------------------------------------------------------
        const llama_token bonus_id = target_tokens[accepted];

        // The total number of new tokens that will land in ctx_tgt's KV: the
        // (accepted + 1) leading positions plus the bonus is the bonus.
        // Wait — actually, we've ALREADY decoded all bs positions on the
        // target. positions [0..accepted] are valid (they match the
        // accepted-prefix proposals); position [accepted] is the
        // first-rejection position whose target prediction becomes our
        // bonus. So the cache should keep [0..accepted] (= accepted+1 tokens).
        const int n_keep_in_block = accepted + 1;

        for (int i = 1; i <= accepted; ++i) {
            const llama_token id = draft_tokens[i];
            output.push_back(id);
            ++n_predict;
            ++n_committed_post_prompt;
            common_sampler_accept(smpl_tgt, id, /*accept_grammar=*/true);

            if (cbs.on_token && !cbs.on_token(id, cbs.user_data)) {
                goto done;
            }
            if (eos_or_stop(id)) {
                goto done;
            }
            if (n_predict >= n_max_predict) {
                goto done;
            }
        }

        // Commit the bonus.
        output.push_back(bonus_id);
        ++n_predict;
        ++n_committed_post_prompt;
        common_sampler_accept(smpl_tgt, bonus_id, /*accept_grammar=*/true);

        if (cbs.on_token && !cbs.on_token(bonus_id, cbs.user_data)) {
            goto done;
        }
        if (eos_or_stop(bonus_id)) {
            goto done;
        }

        // -----------------------------------------------------------------
        // Step 2d: Update KV cache + accumulator with what we accepted.
        // -----------------------------------------------------------------
        // Target KV: drop the rejected-draft tail. We keep positions
        //   [0..pos_block_start + n_keep_in_block - 1].
        // After cropping at `pos_block_start + n_keep_in_block` we drop
        // everything from that position onwards.
        crop_kv_cache(ctx_tgt, pos_block_start + n_keep_in_block);

        // Append the captured features for the accepted positions only.
        // The verify decode produced captures for all bs positions in the
        // same input order; we keep the first n_keep_in_block.
        if (!append_captures(spec, n_keep_in_block)) {
            goto done;
        }
        GGML_ASSERT(spec->accumulated_n == (int64_t) prompt.size() + n_committed_post_prompt);

        LOG_DBG("%s: block #%d: bs=%d, accepted=%d (+1 bonus), n_predict=%d, accumulated_n=%lld\n",
            __func__, n_blocks, bs, accepted, n_predict, (long long) spec->accumulated_n);
    }
done:

    if (stats_out) {
        stats_out->n_blocks    = n_blocks;
        stats_out->n_drafted   = n_drafted;
        stats_out->n_accept    = n_accept;
        stats_out->n_predict   = n_predict;
        stats_out->t_decode_s  = now_seconds() - t_decode_start;
        stats_out->n_input     = (int) prompt.size();
    }

    // Silence unused-variable warning if truncation isn't used in this build path.
    (void) truncate_accumulator;

    return output;
}
