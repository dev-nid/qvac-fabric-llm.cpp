// DFlash speculative-decoding draft model graph builder.
//
// DFlash is a small Qwen3-shaped decoder that drafts an entire block of
// tokens in a single forward pass. The decisive architectural difference
// vs. plain Qwen3 is the per-layer attention block: keys and values are
// the concatenation of (a) per-layer K_ctx / V_ctx pre-projected from
// the *committed* target hidden states (read as zero-copy views of the
// per-layer K/V side store populated by the encoder graph
// `llm_build_dflash_encode` via `llama_dflash_extend`) and (b)
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
    GGML_ASSERT(dflash != nullptr && "DFlash drafter graph requires the dflash drafter state");
    GGML_ASSERT(!dflash->ctx_K.empty() && !dflash->ctx_V.empty()
                && "DFlash drafter graph requires the per-layer K/V side store "
                   "to have been allocated by llama_context's constructor");

    // ---------- DFlash drafter inputs ----------
    auto * inp_dflash = build_inp_dflash();
    ggml_tensor * kq_mask = inp_dflash->kq_mask_cnv;

    // n_ctx_dft is the K_ctx column count seen by attention. We read it
    // from `dflash->n_ctx`, which holds:
    //   * `ctx_capacity` during graph_reserve (worst-case sizing — the
    //     constructor seeds it to cparams.n_ctx_seq before any
    //     graph_reserve is invoked, so the compute buffer is reserved for
    //     the largest possible attention matmul);
    //   * `ctx_filled`   at runtime (kept in sync by `dflash_extend`).
    // The decoder graph is rebuilt each time `n_ctx` changes (the kq_mask
    // shape depends on it via `build_inp_dflash`), so the per-decode view
    // is always exactly the right size and fits comfortably inside the
    // worst-case compute buffer.
    const int64_t n_ctx_dft = dflash->n_ctx;
    // The proposal-window length always equals n_tokens (the ubatch size).
    // build_inp_dflash() above derives it from n_tokens directly, and the
    // n_ubatch clamp in llama_context's constructor pins n_tokens to the
    // model's `dflash_block_size` for any graph_reserve worst case as well.

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

    // The shared `fc + hidden_norm + per-layer wk/wv (+ k_norm + RoPE)` chain
    // that legacy DFlash implementations recompute on every block lives in
    // `llm_build_dflash_encode` instead — it runs once per accepted block,
    // writes the post-projection K and V into the per-layer side store, and
    // the decoder below just reads zero-copy views of it.

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

            // K from target context: zero-copy view of the per-layer side
            // store, already post `wk + k_norm + RoPE` (the encoder graph
            // applied all of that when the features were committed).
            //
            // ne[2] uses the live `n_ctx_dft` (= dflash->n_ctx, which
            // tracks ctx_capacity at graph_reserve time and ctx_filled at
            // runtime), clamped to 1 because ggml does not allow zero-size
            // dimensions even when the ctx is empty (e.g. the very first
            // block before any extend has run — though in practice
            // target_prefill always extends before the first gen_draft).
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

            // V from target context: zero-copy view of the per-layer side
            // store (no norm, no RoPE — V is just `wv · h_ctx_proj`).
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

            // ggml_concat requires matching types. The side store is allocated
            // at `params.type_k` / `params.type_v` (default F16); the noise
            // K/V come out of build_lora_mm + RoPE in whatever type the
            // matmul + RoPE chain produces (often F32). Cast both segments
            // to the noise side's type so the concat is well-defined and the
            // downstream attention math sees a uniform tensor.
            if (Kcur_c->type != Kcur_n->type) {
                Kcur_c = ggml_cast(ctx0, Kcur_c, Kcur_n->type);
                cb(Kcur_c, "Kcur_ctx_cast", il);
            }
            if (Vcur_c->type != Vcur_n->type) {
                Vcur_c = ggml_cast(ctx0, Vcur_c, Vcur_n->type);
                cb(Vcur_c, "Vcur_ctx_cast", il);
            }

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
    ggml_tensor * logits = build_lora_mm(lm_head_use, cur);
    cb(logits, "result_output", -1);

    // In-graph argmax (paper §3 + buun fork's GGML_DFLASH_INLINE_ARGMAX).
    // The DFlash speculative driver only ever needs the greedy top-1 token
    // per draft position, never the full vocab distribution. Doing the
    // argmax inside the graph and reading back I32 [n_outputs] (~64 bytes
    // per block) instead of F32 [n_outputs * n_vocab] (~9.7 MB per block
    // for Qwen3 vocab=151,936 and bs=16) eliminates the dominant draft-
    // side PCIe transfer.
    //
    // Note that we deliberately set `res->t_logits = nullptr` so the
    // generic logits read-back path in llama_context::decode() skips the
    // float copy entirely — the lm_head matmul still computes, but its
    // output is a graph-internal intermediate (no ggml_set_output) and
    // ggml-alloc is free to reuse the storage as soon as ggml_argmax is
    // done with it.
    ggml_tensor * argmax = ggml_argmax(ctx0, logits);
    cb(argmax, "result_output_argmax", -1);
    res->t_logits        = nullptr;
    res->t_logits_argmax = argmax;

    ggml_build_forward_expand(gf, argmax);
}

// =================================================================
// llm_build_dflash_encode
// =================================================================
//
// One-shot graph that projects newly-committed target features through
// `fc + hidden_norm + per-layer wk/wv (+k_norm +RoPE for K)` and scatters
// the results into the persistent DFlash K/V side store at the columns
// specified by the runtime input `pos_idx`. The caller
// (llama_context::dflash_extend) is responsible for advancing
// `dflash->ctx_filled` after compute returns.
//
// This is the central performance optimisation from paper §4.1:
// per-layer K_ctx / V_ctx are computed ONCE when their corresponding
// target features are committed, and reused across every subsequent
// drafting iteration.
//
// Graph-reuse contract: the topology depends ONLY on `n_new` (the
// `n_tokens` derived from the stub ubatch). The destination columns
// are runtime inputs via `pos_idx`, so `dflash_extend` can cache the
// built graph by `n_new` and reuse it on consecutive calls with the
// same `n_new` (skipping `res->reset` + rebuild +
// `ggml_backend_sched_alloc_graph`). For most workloads consecutive
// extends share the same `n_new` (= accepted_drafts + 1 stays
// roughly constant within a phase of generation), so the cache hit
// rate is high.
//
// Required inputs (allocated by this builder, filled by the caller
// pre-compute via `ggml_backend_tensor_set`):
//   target_hidden_new : F32 [n_features, n_new]
//   pos_new           : I32 [n_new]   (= absolute sequence positions for RoPE)
//   pos_idx           : I64 [n_new]   (= side-store column indices to write to,
//                                        commonly = [write_offset..write_offset+n_new-1])

class llm_graph_input_dflash_encode : public llm_graph_input_i {
public:
    llm_graph_input_dflash_encode(int64_t n_features, int64_t n_new)
        : n_features(n_features), n_new(n_new) {}
    virtual ~llm_graph_input_dflash_encode() = default;

    // No ubatch; caller fills the tensors directly.
    void set_input(const llama_ubatch * ubatch) override { GGML_UNUSED(ubatch); }

    ggml_tensor * target_hidden_new = nullptr; // F32 [n_features, n_new]
    ggml_tensor * pos_new           = nullptr; // I32 [n_new]
    ggml_tensor * pos_idx           = nullptr; // I64 [n_new]

    int64_t n_features;
    int64_t n_new;
};

llm_build_dflash_encode::llm_build_dflash_encode(const llama_model    & model,
                                                 const llm_graph_params & params)
    : llm_graph_context(params) {
    const int64_t n_embd_head = hparams.n_embd_head_v;

    GGML_ASSERT(n_embd_head == hparams.n_embd_head_k);
    GGML_ASSERT(n_embd_head == hparams.n_rot);
    GGML_ASSERT(model.dflash_fc          != nullptr);
    GGML_ASSERT(model.dflash_hidden_norm != nullptr);
    GGML_ASSERT(dflash != nullptr && "DFlash encode graph requires the dflash struct");
    GGML_ASSERT((int64_t) dflash->ctx_K.size() == (int64_t) n_layer);
    GGML_ASSERT((int64_t) dflash->ctx_V.size() == (int64_t) n_layer);

    // The "new" extent comes from n_tokens, set by the caller of build_graph.
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
    ggml_tensor * h_proj = build_lora_mm(model.dflash_fc, target_hidden_new);   // [n_embd, n_new]
    h_proj = build_norm(h_proj, model.dflash_hidden_norm, NULL, LLM_NORM_RMS, -1);
    cb(h_proj, "dflash_enc_h_proj", -1);

    // ---------- per-layer K/V projection + scatter into side store via set_rows ----------
    for (int il = 0; il < n_layer; ++il) {
        // K_new = wk · h_proj  → [n_embd_head, n_head_kv, n_new]
        ggml_tensor * K_new = build_lora_mm(model.layers[il].wk, h_proj);
        K_new = ggml_reshape_3d(ctx0, K_new, n_embd_head, n_head_kv, n_new);
        K_new = build_norm(K_new, model.layers[il].attn_k_norm, NULL, LLM_NORM_RMS, il);
        K_new = ggml_rope_ext(
                ctx0, K_new, pos_new, nullptr,
                n_rot, rope_type, n_ctx_orig, freq_base, freq_scale,
                ext_factor, attn_factor, beta_fast, beta_slow);
        cb(K_new, "dflash_enc_K_new", il);

        // V_new = wv · h_proj  → [n_embd_head, n_head_kv, n_new]    (no norm, no RoPE)
        ggml_tensor * V_new = build_lora_mm(model.layers[il].wv, h_proj);
        V_new = ggml_reshape_3d(ctx0, V_new, n_embd_head, n_head_kv, n_new);
        cb(V_new, "dflash_enc_V_new", il);

        // Scatter K_new / V_new into the per-layer side store at the rows
        // indicated by pos_idx. Side-store layout: dst_K[il] is 2D
        // [n_embd_k_gqa, ctx_capacity] (ne[1] == ctx_capacity is the row
        // dim that pos_idx indexes into). K_new / V_new are 3D
        // [n_embd_head, n_head_kv, n_new]; we view them as 2D
        // [n_embd_head*n_head_kv, n_new] for set_rows. This matches the
        // pattern used by `llama_kv_cache::cpy_k` (see
        // `src/llama-kv-cache.cpp`) — same op, same convention.
        ggml_tensor * dst_K = dflash->ctx_K[il];
        ggml_tensor * dst_V = dflash->ctx_V[il];
        GGML_ASSERT(dst_K != nullptr && dst_V != nullptr);
        GGML_ASSERT(dst_K->ne[0] == n_embd_head * n_head_kv);
        GGML_ASSERT(dst_V->ne[0] == n_embd_head * n_head_kv);

        // Merge the [head_dim, n_head_kv] dims of K_new / V_new into
        // a single [n_embd_k_gqa] dim so the source rank matches the
        // 2D dst. The `ggml_row_size(K_new->type, n_embd_head) ==
        // K_new->nb[1]` invariant is required by `ggml_view_2d` here
        // (and is the same precondition `cpy_k` checks); it holds
        // because the head dim is contiguous after reshape_3d + RoPE.
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

    // The encoder graph has no t_logits / t_embd; its outputs are the
    // ggml_set_rows nodes built above, which scatter straight into the
    // side store. Add the input class so the scheduler keeps the inputs
    // alive until the graph is computed.
    res->add_input(std::move(inp));
}
