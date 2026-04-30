// DFlash speculative-decoding draft model graph builder.
//
// DFlash is a small Qwen3-shaped decoder that drafts an entire block of
// tokens in a single forward pass. The decisive architectural difference
// vs. plain Qwen3 is the per-layer attention block: keys and values are the
// concatenation of (a) projections of *frozen target hidden states* -- the
// committed-prefix context staged on the context via llama_set_dflash_input
// -- and (b) projections of the current draft tokens. Queries come only from
// the draft tokens. Attention spans the [ctx | proposal] segments and is
// non-causal within the proposal block (matches the python reference's
// is_causal=False on the bidirectional draft attention).
//
// References:
//   dflash/dflash/model.py      — Qwen3DFlashAttention.forward
//   dflash/dflash/model_mlx.py  — DFlashAttention.__call__
//   core_architecture/01_architecture_analysis.md (this repo) — full analysis

#include "models.h"

llm_build_dflash::llm_build_dflash(const llama_model & model, const llm_graph_params & params) : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);
    GGML_ASSERT(model.dflash_fc          != nullptr);
    GGML_ASSERT(model.dflash_hidden_norm != nullptr);
    GGML_ASSERT(dflash != nullptr && "DFlash drafter graph requires the dflash drafter input");

    // ---------- DFlash drafter inputs ----------
    auto * inp_dflash = build_inp_dflash();
    ggml_tensor * target_hidden = inp_dflash->target_hidden;
    ggml_tensor * pos_ctx       = inp_dflash->pos_ctx;
    ggml_tensor * kq_mask       = inp_dflash->kq_mask_cnv;

    const int64_t n_ctx_dft = dflash->n_ctx;
    // dflash->n_block is implicit in n_tokens (the ubatch size). Asserted here
    // for clarity of intent; the graph reads sizes from the standard inputs.
    GGML_ASSERT(dflash->n_block == (int64_t) n_tokens && "DFlash: ubatch size must equal dflash->n_block");

    // ---------- noise embeddings + block positions ----------
    // The driver constructs the batch as [last_committed, MASK, MASK, ..., MASK]
    // so the standard token-embedding input gives us the noise embedding.
    //
    // Paper §4.2: a paper-faithful DFlash GGUF does not carry its own tok_embd
    // or lm_head — they're shared with the target. If llama_dflash_bind_target
    // has been called, model.target_tok_embd points at the target's embedding
    // and we use that. Otherwise we fall back to a self-contained draft GGUF
    // that loaded its own tok_embd.
    ggml_tensor * tok_embd_use = model.target_tok_embd ? model.target_tok_embd : model.tok_embd;
    GGML_ASSERT(tok_embd_use != nullptr
                && "DFlash drafter: no token embedding found. Either load a self-contained "
                   "DFlash GGUF that includes tok_embd, or call llama_dflash_bind_target() "
                   "with the target model before running speculative decoding.");
    ggml_tensor * inpL = build_inp_embd(tok_embd_use);
    GGML_ASSERT(inpL != nullptr);

    // Block positions: [n_ctx, n_ctx+1, ..., n_ctx+n_block-1].
    // The standard build_inp_pos() reads ubatch->pos which the driver fills
    // with these absolute positions.
    ggml_tensor * inp_pos = build_inp_pos();

    ggml_tensor * inp_out_ids = build_inp_out_ids();

    // ---------- DFlash globals: project the concatenated target hidden states ----------
    // dflash_fc: [n_features, n_embd]
    // dflash_hidden_norm: [n_embd]
    //
    // h_ctx_proj has shape [n_embd, n_ctx_real].
    // The non-real tail of target_hidden is zero (set_input zero-fills it),
    // so projecting it through fc + hidden_norm produces a benign value that
    // the kq_mask -inf will block during attention.
    ggml_tensor * h_ctx_proj = build_lora_mm(model.dflash_fc, target_hidden); // [n_embd, n_ctx]
    h_ctx_proj = build_norm(h_ctx_proj, model.dflash_hidden_norm, NULL, LLM_NORM_RMS, -1);
    cb(h_ctx_proj, "dflash_h_ctx_proj", -1);

    // ---------- decoder stack ----------
    for (int il = 0; il < n_layer; ++il) {
        ggml_tensor * inpSA = inpL;

        ggml_tensor * cur = build_norm(inpL,
                model.layers[il].attn_norm, NULL,
                LLM_NORM_RMS, il);
        cb(cur, "attn_norm", il);

        // ---------- attention with cross-context K/V ----------
        {
            // Q: from the proposal block only (Qwen3-style q_proj + q_norm + RoPE).
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

            // K from target context — same wk projection but applied to h_ctx_proj.
            const int64_t n_ctx_eff = std::max<int64_t>(n_ctx_dft, 1); // matches build_inp_dflash
            ggml_tensor * Kcur_c = build_lora_mm(model.layers[il].wk, h_ctx_proj);
            Kcur_c = ggml_reshape_3d(ctx0, Kcur_c, n_embd_head, n_head_kv, n_ctx_eff);
            Kcur_c = build_norm(Kcur_c, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
            Kcur_c = ggml_rope_ext(
                    ctx0, Kcur_c, pos_ctx, nullptr,
                    n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                    ext_factor, attn_factor, beta_fast, beta_slow);
            cb(Kcur_c, "Kcur_ctx", il);

            // V from proposal (noise) tokens.
            ggml_tensor * Vcur_n = build_lora_mm(model.layers[il].wv, cur);
            Vcur_n = ggml_reshape_3d(ctx0, Vcur_n, n_embd_head, n_head_kv, n_tokens);
            cb(Vcur_n, "Vcur_noise", il);

            // V from target context.
            ggml_tensor * Vcur_c = build_lora_mm(model.layers[il].wv, h_ctx_proj);
            Vcur_c = ggml_reshape_3d(ctx0, Vcur_c, n_embd_head, n_head_kv, n_ctx_eff);
            cb(Vcur_c, "Vcur_ctx", il);

            // Concatenate along the sequence dimension (dim=2 of the
            // [head_dim, n_head_kv, n_tokens] layout): [ctx | proposal].
            ggml_tensor * Kcur = ggml_concat(ctx0, Kcur_c, Kcur_n, 2);
            ggml_tensor * Vcur = ggml_concat(ctx0, Vcur_c, Vcur_n, 2);
            cb(Kcur, "Kcur_concat", il);
            cb(Vcur, "Vcur_concat", il);

            // Pin the K/V graph so the scheduler keeps the concat results
            // around for build_attn_mha (which views/permutes them).
            ggml_build_forward_expand(gf, Qcur);
            ggml_build_forward_expand(gf, Kcur);
            ggml_build_forward_expand(gf, Vcur);

            // Add a stream dim: build_attn_mha expects k->ne[3] = n_stream
            // (here always 1 for the drafter — single sequence).
            ggml_tensor * Q4 = ggml_reshape_4d(ctx0, Qcur, Qcur->ne[0], Qcur->ne[1], Qcur->ne[2], 1);
            ggml_tensor * K4 = ggml_reshape_4d(ctx0, Kcur, Kcur->ne[0], Kcur->ne[1], Kcur->ne[2], 1);
            ggml_tensor * V4 = ggml_reshape_4d(ctx0, Vcur, Vcur->ne[0], Vcur->ne[1], Vcur->ne[2], 1);

            cur = build_attn_mha(Q4, K4, V4,
                                 /*kq_b=*/   nullptr,
                                 /*kq_mask=*/ kq_mask,
                                 /*sinks=*/  nullptr,
                                 /*v_mla=*/  nullptr,
                                 /*kq_scale=*/ 1.0f / sqrtf(float(n_embd_head)),
                                 il);
            cb(cur, "kqv_out", il);

            // o_proj
            cur = build_lora_mm(model.layers[il].wo, cur);
            if (model.layers[il].bo) {
                cur = ggml_add(ctx0, cur, model.layers[il].bo);
            }
        }

        if (il == n_layer - 1 && inp_out_ids) {
            cur   = ggml_get_rows(ctx0,   cur, inp_out_ids);
            inpSA = ggml_get_rows(ctx0, inpSA, inp_out_ids);
        }

        // attention residual
        ggml_tensor * ffn_inp = ggml_add(ctx0, cur, inpSA);
        cb(ffn_inp, "ffn_inp", il);

        // ---------- FFN (Qwen3-style SwiGLU) ----------
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

    // Paper §4.2: lm_head is shared with the target. Prefer the bound
    // target_output; fall back to model.output for self-contained GGUFs.
    // Either must be present, otherwise the driver can't sample anything.
    ggml_tensor * lm_head_use = model.target_output ? model.target_output : model.output;
    GGML_ASSERT(lm_head_use != nullptr
                && "DFlash drafter: no lm_head found. Either load a self-contained "
                   "DFlash GGUF that includes output.weight, or call llama_dflash_bind_target() "
                   "before running speculative decoding.");
    cur = build_lora_mm(lm_head_use, cur);
    cb(cur, "result_output", -1);
    res->t_logits = cur;

    ggml_build_forward_expand(gf, cur);
}
