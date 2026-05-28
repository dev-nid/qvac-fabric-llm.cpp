// DFlash speculative-decoding draft model graph builder.
//
// DFlash is a small Qwen3-shaped decoder that drafts an entire block of
// tokens in a single forward pass. The decisive architectural difference
// vs. plain Qwen3 is the per-layer attention block: keys and values are
// the concatenation of (a) per-layer K_ctx / V_ctx pre-projected from
// the *committed* target hidden states (read as zero-copy views of the
// per-layer K/V side store populated by the encoder graph
// `llama_model_dflash::encode_graph` via `llama_dflash_extend`) and (b)
// projections of the current draft tokens. Queries come only from the
// draft tokens. Attention spans the [ctx | proposal] segments and is
// non-causal within the proposal block (matches the python reference's
// is_causal=False on the bidirectional draft attention).
//
// The K/V side store is the paper §4.1 reuse optimisation: the
// `fc + hidden_norm + per-layer wk/wv (+k_norm +RoPE)` chain runs ONCE
// when each block of target features is committed (in the encoder
// graph), and every subsequent draft decode reads pre-projected
// post-RoPE K and post-projection V via zero-copy views.

#include "models.h"

void llama_model_dflash::load_arch_hparams(llama_model_loader & ml) {
    ml.get_key(LLM_KV_ATTENTION_LAYERNORM_RMS_EPS, hparams.f_norm_rms_eps);

    // DFlash-specific KVs (optional so the synthetic test-llama-archs
    // generator's GGUFs that lack these keys can still be loaded; real
    // DFlash GGUFs always include them via the Python converter).
    ml.get_key(LLM_KV_DFLASH_BLOCK_SIZE, hparams.dflash_block_size, false);

    // mask_token_id arrives as uint32 in GGUF; cast into signed token type
    {
        uint32_t mask_id = 0;
        if (ml.get_key(LLM_KV_DFLASH_MASK_TOKEN_ID, mask_id, false)) {
            hparams.dflash_mask_token_id = (llama_token) mask_id;
        }
    }

    ml.get_key(LLM_KV_DFLASH_NUM_TARGET_LAYERS, hparams.dflash_num_target_layers, false);

    // Read target_layer_ids array (variable length, capped at LLAMA_DFLASH_MAX_TARGET_LAYERS)
    {
        std::array<int32_t, LLAMA_DFLASH_MAX_TARGET_LAYERS> ids{};
        ids.fill(-1);
        if (ml.get_arr(LLM_KV_DFLASH_TARGET_LAYER_IDS, ids, false)) {
            // Count the valid entries (non-negative).
            uint32_t n = 0;
            for (size_t i = 0; i < ids.size(); ++i) {
                if (ids[i] >= 0) { n++; } else { break; }
            }
            hparams.dflash_target_layer_ids = ids;
            hparams.dflash_n_target_layer_ids = n;
        } else {
            hparams.dflash_target_layer_ids.fill(-1);
            hparams.dflash_n_target_layer_ids = 0;
        }
    }

    // Gemma-family DFlash drafts carry extra hparams. All optional so
    // non-Gemma DFlash GGUFs (Qwen3, Qwen3.5, LLaMA-3.1 etc.) load unchanged.
    //
    // final_logit_softcapping (scalar): consumed by the draft graph after lm_head
    // sliding_window (scalar):          window width for SWA layers
    // sliding_window_pattern (array):   per-layer SWA layer-type marker
    ml.get_key(LLM_KV_FINAL_LOGIT_SOFTCAPPING, hparams.f_final_logit_softcapping, false);

    uint32_t n_swa_local = 0;
    if (ml.get_key(LLM_KV_ATTENTION_SLIDING_WINDOW, n_swa_local, false) && n_swa_local > 0) {
        hparams.n_swa    = n_swa_local;
        hparams.swa_type = LLAMA_SWA_TYPE_STANDARD;

        // sliding_window_pattern can be either a length-n_layer boolean
        // array (Gemma-4-style per-layer types) or absent. We try the
        // full per-layer read first and on any error fall back to "all
        // layers SWA" so abbreviated / mismatched GGUFs still load. The
        // SWA mask itself is a no-op for prompts within n_swa, so the
        // fallback only affects long-context correctness.
        try {
            ml.get_key_or_arr(LLM_KV_ATTENTION_SLIDING_WINDOW_PATTERN,
                              hparams.swa_layers, hparams.n_layer, false);
        } catch (const std::runtime_error & e) {
            LLAMA_LOG_WARN("%s: dflash SWA pattern array shape mismatch, "
                           "marking all layers SWA (%s)\n", __func__, e.what());
            for (uint32_t il = 0; il < hparams.n_layer; ++il) {
                hparams.swa_layers[il] = 1;
            }
        }

        // Keep the invariant llama_model::create_memory expects:
        // swa_type != NONE iff is_swa_any(). Some GGUFs carry the
        // sliding_window key without a usable per-layer pattern (or
        // with a pattern that ends up all-zero); in that case revert
        // to NONE so the non-SWA KV cache path is selected and the
        // assertion at llama-model.cpp doesn't fire.
        if (!hparams.is_swa_any()) {
            hparams.n_swa    = 0;
            hparams.swa_type = LLAMA_SWA_TYPE_NONE;
        }
    }

    // type tag (use Qwen3-ish heuristics; not load-bearing for DFlash drafts)
    switch (hparams.n_layer) {
        case 1:  type = LLM_TYPE_UNKNOWN; break;
        case 2:  type = LLM_TYPE_UNKNOWN; break;
        default: type = LLM_TYPE_UNKNOWN;
    }
}

void llama_model_dflash::load_arch_tensors(llama_model_loader &) {
    LLAMA_LOAD_LOCALS;

    // Paper §4.2: a paper-faithful DFlash GGUF does NOT carry tok_embd /
    // output. They are bound at runtime by the speculative driver via
    // llama_dflash_bind_target() (target_tok_embd / target_output fields).
    // For self-contained drafts (e.g. older / debug GGUFs) we still allow
    // tok_embd + output to be present.
    tok_embd = create_tensor(tn(LLM_TENSOR_TOKEN_EMBD,  "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);
    output_norm = create_tensor(tn(LLM_TENSOR_OUTPUT_NORM, "weight"), {n_embd}, 0);
    output = create_tensor(tn(LLM_TENSOR_OUTPUT,        "weight"), {n_embd, n_vocab}, TENSOR_NOT_REQUIRED);

    // DFlash globals (TENSOR_NOT_REQUIRED so the synthetic test-llama-archs
    // generator's GGUFs without these tensors can still load; real DFlash
    // GGUFs always include them via the Python converter and the graph
    // builder asserts on their presence at decode time).
    {
        const int n_target_ids = (int) hparams.dflash_n_target_layer_ids;
        // dflash_fc projects [n_target_ids * n_embd] → [n_embd]
        const int n_in = n_target_ids > 0 ? n_target_ids * n_embd : n_embd;
        dflash_fc          = create_tensor(tn(LLM_TENSOR_DFLASH_FC,          "weight"), {n_in, n_embd}, TENSOR_NOT_REQUIRED);
        dflash_hidden_norm = create_tensor(tn(LLM_TENSOR_DFLASH_HIDDEN_NORM, "weight"), {n_embd}, TENSOR_NOT_REQUIRED);
    }

    for (int i = 0; i < n_layer; ++i) {
        auto & layer = layers[i];

        layer.attn_norm = create_tensor(tn(LLM_TENSOR_ATTN_NORM, "weight", i), {n_embd}, 0);

        create_tensor_qkv(layer, i, n_embd, n_embd_head_k * n_head, n_embd_gqa, n_embd_gqa, 0);
        layer.wo = create_tensor(tn(LLM_TENSOR_ATTN_OUT, "weight", i), {n_embd_head_k * n_head, n_embd}, 0);

        layer.attn_k_norm = create_tensor(tn(LLM_TENSOR_ATTN_K_NORM, "weight", i), {n_embd_head_k}, 0);
        layer.attn_q_norm = create_tensor(tn(LLM_TENSOR_ATTN_Q_NORM, "weight", i), {n_embd_head_k}, 0);

        layer.ffn_norm = create_tensor(tn(LLM_TENSOR_FFN_NORM, "weight", i), {n_embd}, 0);
        layer.ffn_gate = create_tensor(tn(LLM_TENSOR_FFN_GATE, "weight", i), {n_embd, n_ff}, 0);
        layer.ffn_down = create_tensor(tn(LLM_TENSOR_FFN_DOWN, "weight", i), {n_ff,   n_embd}, 0);
        layer.ffn_up   = create_tensor(tn(LLM_TENSOR_FFN_UP,   "weight", i), {n_embd, n_ff}, 0);
    }
}

std::unique_ptr<llm_graph_context> llama_model_dflash::build_arch_graph(const llm_graph_params & params) const {
    return std::make_unique<graph>(*this, params);
}

llama_model_dflash::graph::graph(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == (int64_t) hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);

    // Synthetic test-llama-archs models may lack dflash_fc/dflash_hidden_norm.
    // Emit a minimal graph (just an output tensor) so the test harness's
    // graph-build sweep can complete without crashing. Real DFlash GGUFs
    // always have these tensors via the Python converter.
    if (model.dflash_fc == nullptr || model.dflash_hidden_norm == nullptr ||
        dflash == nullptr || dflash->ctx_K.empty() || dflash->ctx_V.empty()) {
        ggml_tensor * inpL = build_inp_embd(model.tok_embd);
        if (inpL && model.output_norm) {
            inpL = build_norm(inpL, model.output_norm, NULL, LLM_NORM_RMS, -1);
        }
        if (inpL && model.output) {
            ggml_tensor * out = build_lora_mm(model.output, inpL);
            cb(out, "result_output", -1);
            res->t_logits = out;
            ggml_build_forward_expand(gf, out);
        }
        return;
    }

    // ---------- DFlash drafter inputs ----------
    auto * inp_dflash = build_inp_dflash();
    ggml_tensor * kq_mask     = inp_dflash->kq_mask_cnv;
    ggml_tensor * kq_mask_swa = inp_dflash->kq_mask_swa_cnv;

    const int64_t n_ctx_dft = dflash->n_ctx;

    // ---------- noise embeddings + block positions ----------
    // Paper §4.2: a paper-faithful DFlash GGUF does not carry its own tok_embd
    // or lm_head — they're shared with the target. If llama_dflash_bind_target
    // has been called, model.target_tok_embd points at the target's embedding
    // and we use that. Otherwise we fall back to a self-contained draft GGUF
    // that loaded its own tok_embd.
    ggml_tensor * tok_embd_use = model.target_tok_embd ? model.target_tok_embd : model.tok_embd;
    GGML_ASSERT(tok_embd_use != nullptr
                && "DFlash drafter: no token embedding found.");
    ggml_tensor * inpL = build_inp_embd(tok_embd_use);
    GGML_ASSERT(inpL != nullptr);

    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // ---------- decoder stack ----------
    // Force F32 accumulation on the lm_head matmul (the final projection that
    // produces logits → argmax → draft token choices). Intermediate decoder
    // layers stay at default precision (F16 on Vulkan coopmat, BF16 on CUDA
    // tensor cores) for throughput. Only the lm_head determines argmax, so
    // forcing F32 there is sufficient for deterministic draft predictions
    // across driver versions. Cost: ~2% on the lm_head matmul only (one of
    // ~15 total matmuls in the 5-layer decoder; negligible).
    auto build_lora_mm_f32 = [&](ggml_tensor * w, ggml_tensor * cur_in) -> ggml_tensor * {
        ggml_tensor * r = build_lora_mm(w, cur_in);
        ggml_mul_mat_set_prec(r, GGML_PREC_F32);
        return r;
    };

    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // ---------- attention with cross-context K/V ----------
        {
            // Q: from the proposal block only.
            ggml_tensor * Qcur = build_lora_mm(model.layers[il].wq, cur);
            Qcur = ggml_reshape_3d(ctx0, Qcur, n_embd_head, n_head, n_tokens);
            Qcur = build_norm(Qcur, model.layers[il].attn_q_norm, NULL, LLM_NORM_RMS, il);
            Qcur = ggml_rope_ext(
                    ctx0, Qcur, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Qcur, "Qcur", il);

            // K from proposal (noise) tokens.
            ggml_tensor * Kcur_n = build_lora_mm(model.layers[il].wk, cur);
            Kcur_n = ggml_reshape_3d(ctx0, Kcur_n, n_embd_head, n_head_kv, n_tokens);
            Kcur_n = build_norm(Kcur_n, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            Kcur_n = ggml_rope_ext(
                    ctx0, Kcur_n, inp_pos, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_n, "Kcur_noise", il);

            // K from target context: zero-copy view of the per-layer side store.
            const int64_t n_ctx_eff = std::max<int64_t>(n_ctx_dft, 1);
            ggml_tensor * dst_K = dflash->ctx_K[il];
            GGML_ASSERT(dst_K != nullptr);
            GGML_ASSERT(dst_K->ne[0] == n_embd_head * n_head_kv);
            const size_t row_size_K = ggml_row_size(dst_K->type, n_embd_head);
            ggml_tensor * Kcur_c = ggml_view_3d(
                ctx0, dst_K,
                n_embd_head, n_head_kv, n_ctx_eff,
                /*nb1=*/ row_size_K,
                /*nb2=*/ dst_K->nb[1],
                /*offset=*/ 0);
            cb(Kcur_c, "Kcur_ctx", il);

            // V from proposal (noise) tokens.
            ggml_tensor * Vcur_n = build_lora_mm(model.layers[il].wv, cur);
            Vcur_n = ggml_reshape_3d(ctx0, Vcur_n, n_embd_head, n_head_kv, n_tokens);
            cb(Vcur_n, "Vcur_noise", il);

            // V from target context: zero-copy view of the per-layer side store.
            ggml_tensor * dst_V = dflash->ctx_V[il];
            GGML_ASSERT(dst_V != nullptr);
            GGML_ASSERT(dst_V->ne[0] == n_embd_head * n_head_kv);
            const size_t row_size_V = ggml_row_size(dst_V->type, n_embd_head);
            ggml_tensor * Vcur_c = ggml_view_3d(
                ctx0, dst_V,
                n_embd_head, n_head_kv, n_ctx_eff,
                /*nb1=*/ row_size_V,
                /*nb2=*/ dst_V->nb[1],
                /*offset=*/ 0);
            cb(Vcur_c, "Vcur_ctx", il);

            // Cast both segments to the noise side's type so concat is well-defined.
            if (Kcur_c->type != Kcur_n->type) {
                Kcur_c = ggml_cast(ctx0, Kcur_c, Kcur_n->type);
                cb(Kcur_c, "Kcur_ctx_cast", il);
            }
            if (Vcur_c->type != Vcur_n->type) {
                Vcur_c = ggml_cast(ctx0, Vcur_c, Vcur_n->type);
                cb(Vcur_c, "Vcur_ctx_cast", il);
            }

            // Concatenate along the sequence dimension: [ctx | proposal].
            ggml_tensor * Kcur = ggml_concat(ctx0, Kcur_c, Kcur_n, 2);
            ggml_tensor * Vcur = ggml_concat(ctx0, Vcur_c, Vcur_n, 2);
            cb(Kcur, "Kcur_concat", il);
            cb(Vcur, "Vcur_concat", il);

            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            // Add a stream dim: build_attn_mha expects k->ne[3] = n_stream (always 1 for drafter).
            ggml_tensor * Q4 = ggml_reshape_4d(ctx0, Qcur, Qcur->ne[0], Qcur->ne[1], Qcur->ne[2], 1);
            ggml_tensor * K4 = ggml_reshape_4d(ctx0, Kcur, Kcur->ne[0], Kcur->ne[1], Kcur->ne[2], 1);
            ggml_tensor * V4 = ggml_reshape_4d(ctx0, Vcur, Vcur->ne[0], Vcur->ne[1], Vcur->ne[2], 1);

            // Gemma-family drafts interleave SWA (local) and full
            // (global) attention layers; pick the matching mask per
            // layer. Non-SWA drafts have kq_mask_swa == nullptr and
            // always use the dense mask.
            ggml_tensor * mask_use = (kq_mask_swa && hparams.is_swa(il)) ? kq_mask_swa : kq_mask;

            cur = build_attn_mha(Q4, K4, V4,
                                 /*kq_b=*/   nullptr,
                                 /*kq_mask=*/ mask_use,
                                 /*sinks=*/  nullptr,
                                 /*v_mla=*/  nullptr,
                                 /*kq_scale=*/ 1.0f / sqrtf(float(n_embd_head)),
                                 il);
            cb(cur, "kqv_out", il);

            // o_proj
            cur = build_lora_mm(model.layers[il].wo, cur);
            if (model.layers[il].wo_b) {
                cur = ggml_add(ctx0, cur, model.layers[il].wo_b);
            }
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // attention residual
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // ---------- FFN (Qwen3-style SwiGLU, inlined for F32 prec) ----------
        cur = build_norm(ffn_inp,
                model.layers[il].ffn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "ffn_norm", il);

        cur = build_ffn(cur,
                model.layers[il].ffn_up,   NULL, NULL,
                model.layers[il].ffn_gate, NULL, NULL,
                model.layers[il].ffn_down, NULL, NULL,
                NULL,
                LLM_FFN_SILU, LLM_FFN_PAR, il);
        cb(cur, "ffn_out", il);

        cur = ggml_add(ctx0, cur, ffn_inp);
        cur = build_cvec(cur, il);
        cb(cur, "l_out", il);

        inpL = cur;
    }

    // ---------- final norm + lm_head ----------
    ggml_tensor * cur = inpL;
    cur = build_norm(cur, model.output_norm, NULL, LLM_NORM_RMS, -1);
    cb(cur, "result_norm", -1);
    res->t_embd = cur;

    // Paper §4.2: lm_head is shared with the target.
    ggml_tensor * lm_head_use = model.target_output ? model.target_output : model.output;
    GGML_ASSERT(lm_head_use != nullptr && "DFlash drafter: no lm_head found.");
    ggml_tensor * logits = build_lora_mm_f32(lm_head_use, cur);
    cb(logits, "result_output", -1);

    // Gemma-family targets apply a tanh-based final-logit softcap after lm_head.
    // The shared lm_head matmul gives the same raw logits as the target's,
    // but softcapping is a separate post-process - without it the draft
    // produces uncapped logits while the target produces capped ones.
    // Monotonic, so chain-mode argmax / top-K selection is unchanged at T=0,
    // but it does change absolute logit values used by T>0 sampling and
    // log-prob paths.
    if (hparams.f_final_logit_softcapping) {
        logits = ggml_scale(ctx0, logits, 1.0f / hparams.f_final_logit_softcapping);
        logits = ggml_tanh(ctx0, logits);
        logits = ggml_scale(ctx0, logits, hparams.f_final_logit_softcapping);
        cb(logits, "result_output_softcapped", -1);
    }

    // In-graph top-K to eliminate the bs * n_vocab * 4 byte PCIe transfer per block.
    const uint32_t K = cparams.dflash_topk == 0 ? 1 : cparams.dflash_topk;
    ggml_tensor * topk;
    ggml_tensor * topk_argmax = nullptr;
    if (K == 1) {
        // Chain mode: K=1 is just argmax. ggml_top_k(x, 1) routes through the
        // CUDA argsort path which on CCCL < 3.2 falls back to
        // DeviceSegmentedSortFallbackKernel — a full sort of [vocab × n_tokens]
        // costing ~1.3 ms/round per nsys profiling on Qwen 3.6-27B Q5_K_S
        // (HumanEval/66, May 2026). Using ggml_argmax instead dispatches to a
        // dedicated single-pass reduction kernel (~10× faster). Output layout
        // is byte-compatible with top_k(.,1): n_outputs × i32, contiguous, so
        // the host readback at llama-context.cpp::~3071 is unchanged.
        topk = ggml_argmax(ctx0, logits);
        cb(topk, "result_output_argmax", -1);
    } else {
        topk = ggml_top_k(ctx0, logits, (int) K);
        cb(topk, "result_output_topk", -1);

        // Same reasoning as the K == 1 branch above: an argmax tensor
        // computed via ggml_argmax avoids the cub::DeviceSegmentedSort
        // fallback that ggml_top_k(.,1) routes through on CCCL < 3.2.
        // Used by the tree-mode best-first expansion path to find the
        // per-position argmax cheaply alongside the K-best candidates.
        topk_argmax = ggml_argmax(ctx0, logits);
        cb(topk_argmax, "result_output_topk_argmax", -1);
    }
    // Default: skip the bs * n_vocab * 4 byte logits readback (chain mode and
    // uniform-expansion tree mode never need it). When dflash_emit_logits is
    // set (best-first tree expansion), emit the full logits so the host can do
    // softmax + log-prob top-K for the heap-based tree builder.
    res->t_logits             = cparams.dflash_emit_logits ? logits : nullptr;
    res->t_dflash_topk        = topk;
    res->t_dflash_topk_argmax = topk_argmax;

    ggml_build_forward_expand(gf, topk);
    if (topk_argmax) {
        ggml_build_forward_expand(gf, topk_argmax);
    }
    if (cparams.dflash_emit_logits) {
        ggml_build_forward_expand(gf, logits);
    }
}

// llama_model_dflash::encode_graph
//
// One-shot graph that projects newly-committed target features through
// `fc + hidden_norm + per-layer wk/wv (+k_norm +RoPE for K)` and scatters
// the results into the persistent DFlash K/V side store. Called from
// llama_context::dflash_extend().

class llm_graph_input_dflash_encode : public llm_graph_input_i {
public:
    llm_graph_input_dflash_encode(int64_t n_features, int64_t n_new)
        : n_features(n_features), n_new(n_new) {}
    virtual ~llm_graph_input_dflash_encode() = default;

    void set_input(const llama_ubatch * ubatch) override { GGML_UNUSED(ubatch); }

    ggml_tensor * target_hidden_new = nullptr; // F32 [n_features, n_new]
    ggml_tensor * pos_new           = nullptr; // I32 [n_new]
    ggml_tensor * pos_idx           = nullptr; // I64 [n_new]

    int64_t n_features;
    int64_t n_new;
};

llama_model_dflash::encode_graph::encode_graph(const llama_model    & model,
                                               const llm_graph_params & params)
    : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v();

    GGML_ASSERT(n_embd_head == (int64_t) hparams.n_embd_head_k());
    GGML_ASSERT(n_embd_head == n_rot);
    GGML_ASSERT(model.dflash_fc          != nullptr);
    GGML_ASSERT(model.dflash_hidden_norm != nullptr);
    GGML_ASSERT(dflash != nullptr && "DFlash encode graph requires the dflash struct");
    GGML_ASSERT((int64_t) dflash->ctx_K.size() == (int64_t) n_layer);
    GGML_ASSERT((int64_t) dflash->ctx_V.size() == (int64_t) n_layer);

    const int64_t n_new      = (int64_t) n_tokens;
    const int64_t n_features = dflash->n_features;
    GGML_ASSERT(n_new      >  0 && "encoder: n_new must be > 0");
    GGML_ASSERT(n_features >  0 && "encoder: n_features must be > 0");

    // ---------- inputs ----------
    auto inp = std::make_unique<llm_graph_input_dflash_encode>(n_features, n_new);

    inp->target_hidden_new = ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_features, n_new);
    ggml_set_input(inp->target_hidden_new);
    cb(inp->target_hidden_new, "dflash_enc_target_hidden_new", -1);

    inp->pos_new = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, n_new);
    ggml_set_input(inp->pos_new);
    cb(inp->pos_new, "dflash_enc_pos_new", -1);

    inp->pos_idx = ggml_new_tensor_1d(ctx0, GGML_TYPE_I64, n_new);
    ggml_set_input(inp->pos_idx);
    cb(inp->pos_idx, "dflash_enc_pos_idx", -1);

    ggml_tensor * target_hidden_new = inp->target_hidden_new;
    ggml_tensor * pos_new           = inp->pos_new;
    ggml_tensor * pos_idx           = inp->pos_idx;

    // ---------- shared projection: h_proj = hidden_norm(fc(target_hidden_new)) ----------
    // target_hidden_new is the concat of unnormalized hidden states from
    // several target layers; its magnitude is unbounded and the fc weight is
    // F16. On CUDA the default mat-mul kernel for F16 weights uses an F16
    // accumulator, which overflows for output magnitudes > ~6.5e4 and produces
    // +inf values. Those infs poison the following RMSNorm (scale becomes 0)
    // and cascade into NaN downstream, killing speculative acceptance on
    // medium/long prompts. Force F32 accumulation here; the RMSNorm right
    // after still normalizes the result, so the fix is free in terms of
    // numerics but pays a small CUDA mat-mul cost for higher precision.
    ggml_tensor * h_fc = build_lora_mm(model.dflash_fc, target_hidden_new);
    ggml_mul_mat_set_prec(h_fc, GGML_PREC_F32);
    cb(h_fc, "dflash_enc_h_fc", -1);
    ggml_tensor * h_proj = build_norm(h_fc, model.dflash_hidden_norm, NULL, LLM_NORM_RMS, -1);
    cb(h_proj, "dflash_enc_h_proj", -1);

    // ---------- per-layer K/V projection + scatter into side store via set_rows ----------
    // Same F16-accumulator concern as h_fc above. The wk/wv weights are F16
    // and on backends with fp16-capable devices the default mat-mul accumulator
    // is F16. The side-store K/V written here is the only path the captured
    // target features take into the spec block, so any precision loss in this
    // projection feeds straight into draft acceptance. Force F32 accumulation
    // on both projections; cost is a small per-mat-mul precision tax, gain is
    // the captured features arriving at draft attention without F16 truncation.
    for (int il = 0; il < n_layer; ++il) {
        // K_new = wk · h_proj  → [n_embd_head, n_head_kv, n_new]
        ggml_tensor * K_new = build_lora_mm(model.layers[il].wk, h_proj);
        ggml_mul_mat_set_prec(K_new, GGML_PREC_F32);
        K_new = ggml_reshape_3d(ctx0, K_new, n_embd_head, n_head_kv, n_new);
        K_new = build_norm(K_new, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        K_new = ggml_rope_ext(
                ctx0, K_new, pos_new, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(K_new, "dflash_enc_K_new", il);

        // V_new = wv · h_proj  (no norm, no RoPE)
        ggml_tensor * V_new = build_lora_mm(model.layers[il].wv, h_proj);
        ggml_mul_mat_set_prec(V_new, GGML_PREC_F32);
        V_new = ggml_reshape_3d(ctx0, V_new, n_embd_head, n_head_kv, n_new);
        cb(V_new, "dflash_enc_V_new", il);

        ggml_tensor * dst_K = dflash->ctx_K[il];
        ggml_tensor * dst_V = dflash->ctx_V[il];
        GGML_ASSERT(dst_K != nullptr && dst_V != nullptr);
        GGML_ASSERT(dst_K->ne[0] == n_embd_head * n_head_kv);
        GGML_ASSERT(dst_V->ne[0] == n_embd_head * n_head_kv);

        GGML_ASSERT(ggml_row_size(K_new->type, n_embd_head) == K_new->nb[1]);
        GGML_ASSERT(ggml_row_size(V_new->type, n_embd_head) == V_new->nb[1]);
        ggml_tensor * K_new_2d = ggml_view_2d(
            ctx0, K_new, n_embd_head * n_head_kv, n_new, K_new->nb[2], 0);
        ggml_tensor * V_new_2d = ggml_view_2d(
            ctx0, V_new, n_embd_head * n_head_kv, n_new, V_new->nb[2], 0);

        ggml_tensor * scatter_K = ggml_set_rows(ctx0, dst_K, K_new_2d, pos_idx);
        ggml_tensor * scatter_V = ggml_set_rows(ctx0, dst_V, V_new_2d, pos_idx);

        ggml_build_forward_expand(gf, scatter_K);
        ggml_build_forward_expand(gf, scatter_V);
    }

    res->add_input(std::move(inp));
}
