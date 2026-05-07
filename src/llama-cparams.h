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
    // position. 1 = chain mode (cheap argmax kernel; byte-exact-equivalent
    // to the pre-DDTree behavior). >=2 = tree mode (ggml_argsort_top_k);
    // consumed by the speculative driver to build K parallel verify chains.
    // See llama_context_params::dflash_topk for full semantics.
    uint32_t dflash_topk;

    ggml_backend_sched_eval_callback cb_eval;
    void * cb_eval_user_data;
};
