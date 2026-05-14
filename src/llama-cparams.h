#pragma once

#include "llama.h"

#include <cstdint>

#define LLAMA_MAX_SEQ 256

struct llama_cparams {
    uint32_t n_ctx;           // context size used during inference
    uint32_t n_ctx_seq;       // context for a single sequence
    uint32_t n_batch;
    uint32_t n_ubatch;
    uint32_t n_seq_max;
    int32_t  n_threads;       // number of threads to use for generation
    int32_t  n_threads_batch; // number of threads to use for batch processing

    float rope_freq_base;
    float rope_freq_scale;

    uint32_t n_ctx_orig_yarn;
    // These hyperparameters are not exposed in GGUF, because all
    // existing YaRN models use the same values for them.
    float yarn_ext_factor;
    float yarn_attn_factor;
    float yarn_beta_fast;
    float yarn_beta_slow;

    bool embeddings;
    bool causal_attn;
    bool offload_kqv;
    bool flash_attn;
    bool auto_fa;
    bool fused_gdn_ar;       // use fused gated delta net (autoregressive)
    bool fused_gdn_ch;       // use fused gated delta net (chunked)
    bool auto_fgdn;
    bool no_perf;
    bool warmup;
    bool op_offload;
    bool kv_unified;
    bool pipeline_parallel;

    enum llama_pooling_type pooling_type;

    // Sliding-window cap on the DFlash drafter's per-layer K/V side store
    // (ignored for non-DFlash drafts). -1 = auto-scale (default), 0 = uncapped,
    // >0 = explicit cap. Resolved at side-store allocation time in
    // llama-context.cpp. Signed so the -1 sentinel survives forwarding from
    // the public `llama_context_params::dflash_max_ctx`. See the docstring on
    // llama_context_params::dflash_max_ctx for full semantics.
    int32_t dflash_max_ctx;

    // Number of top-K candidate tokens the DFlash drafter emits per output
    // position. 1 = chain mode (cheap argmax kernel). >=2 = tree mode
    // (ggml_argsort_top_k); consumed by the speculative driver to build K
    // parallel verify chains.
    // See llama_context_params::dflash_topk for full semantics.
    uint32_t dflash_topk;

    // When true, the DFlash draft graph emits full-vocab logits so the host
    // can compute per-position softmax / log-probs for best-first tree
    // expansion. Default false (keeps the bs * n_vocab * 4 byte readback
    // optimisation that ships with chain-mode and uniform-expansion tree
    // mode). Only consulted on LLM_ARCH_DFLASH drafts. Enabled automatically
    // when --dflash-tree-best-first is passed.
    bool dflash_emit_logits;

    // Inline DFlash encoder. When true on the TARGET context, the target's
    // graph builder runs the encoder K/V projection inline after the final
    // captured layer using non-owning pointers populated by
    // llama_dflash_bind_encoder(). The companion sizing fields below are
    // needed at graph_reserve time before the draft model has been bound.
    // All four sizing fields must be > 0 if dflash_inline_encoder is true,
    // or initialisation fails.
    //
    //   n_embd_dft        : draft embedding dim
    //   n_head_kv_dft     : draft n_head_kv (K/V heads count)
    //   n_embd_head_dft   : draft n_embd_head_v == n_embd_head_k
    //   n_target_layers   : len(draft target_layer_ids), one K/V side-store
    //                       layer per draft layer
    //
    // Default: all zero, dflash_inline_encoder = false.
    bool     dflash_inline_encoder;
    uint32_t dflash_inline_n_embd_dft;
    uint32_t dflash_inline_n_head_kv_dft;
    uint32_t dflash_inline_n_embd_head_dft;
    uint32_t dflash_inline_n_target_layers;

    // GatedDeltaNet history kernel. When true on a TARGET context with GDN
    // layers (Qwen3.5 family), the GDN op uses the
    // GGML_OP_GATED_DELTA_NET_WITH_HISTORY variant which writes per-token
    // recurrent state to a persistent per-layer buffer. The speculative
    // driver picks state[K-1] post-sampler to roll back the recurrent state
    // on partial acceptance, replacing the FULL-seq_rm checkpoint+re-verify
    // path. Throws at init on LLM_ARCH_DFLASH contexts (target-only).
    // CUDA-only at runtime. Default: false.
    bool dflash_gdn_history;

    // GDN history persistent buffer dtype. When true, the per-layer
    // dflash.gdn_history[il] tensor is allocated as GGML_TYPE_F16 instead of
    // GGML_TYPE_F32. This halves the per-layer footprint of the persistent
    // state buffer.
    //
    // Picked up at runtime by the GDN op kernel: the InterT template parameter
    // is selected from src_persist_inter->type. Ignored when dflash_gdn_history
    // is false. Default false.
    bool dflash_gdn_history_f16;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
