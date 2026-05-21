#pragma once

#include "llama.h"
#include "llama-ext.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama-adapter.h"
#include "llama-impl.h"

#include "ggml-cpp.h"
#include "ggml-opt.h"

#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// stores copy of the memory in device buffer. used for fast state save/load
struct llama_memory_buffer {
    int n_tensors = 0;
    size_t total_size = 0;

    ggml_backend_buffer_ptr buf;

    ggml_context_ptr ctx;

    std::vector<ggml_tensor *> org;
    std::vector<ggml_tensor *> cpy;
};

using llama_memory_buffers = std::map<ggml_backend_buffer_type_t, llama_memory_buffer>;

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

    // reserve a new backend scheduler (if needed)
    // for example, when:
    //   - changing loras
    //   - changing samplers
    //   - changing attention type
    //   - etc.
    void sched_reserve();

    void synchronize();

    const llama_model   & get_model()   const;
    const llama_cparams & get_cparams() const;

    ggml_backend_sched_t get_sched() const;

    uint32_t n_ctx()     const;
    uint32_t n_ctx_seq() const;
    uint32_t n_batch()   const;
    uint32_t n_ubatch()  const;
    uint32_t n_seq_max() const;

    uint32_t n_threads()       const;
    uint32_t n_threads_batch() const;

    llama_memory_t get_memory() const;

    // return true if the memory was updated
    bool memory_update(bool optimize);

    enum llama_pooling_type pooling_type() const;

    float * get_logits();
    float * get_logits_ith(int32_t i);

    float * get_embeddings();
    float * get_embeddings_ith(int32_t i);
    float * get_embeddings_seq(llama_seq_id seq_id);

    // DFlash: capture target hidden states for the drafter
    void set_dflash_capture(const int32_t * layer_ids,
                            size_t          n_layer_ids,
                            int64_t         n_embd_target);

    const float * get_dflash_captured_features(int64_t * n_outputs_out) const;

    // After decode on a draft context (LLM_ARCH_DFLASH), returns top-K candidates.
    const int32_t * get_dflash_draft_topk(int64_t * n_outputs_out, uint32_t * topk_out) const;

    // Idempotent O(n_outputs * K) post-pass to swap argmax into slot 0 for K>=2.
    void dflash_finalize_draft_topk();

    // Read access for the graph builders.
    const llama_dflash * get_dflash() const;

    // Append `n_new` newly-committed target captures to the K/V side store.
    int32_t dflash_extend(const float * target_hidden_new,
                          int64_t       n_new,
                          int64_t       pos_start);

    // Device-to-device variant of dflash_extend: source is a packed-captures
    // ggml_tensor produced by a target context's most recent llama_decode().
    // Skips the H2D bounce. Internally builds the same encoder graph as the
    // host-pointer variant; only the t_target_hidden_new input is populated
    // via ggml_backend_tensor_copy_async from src_captures instead of
    // ggml_backend_tensor_set from a host buffer. Returns 0 on success.
    int32_t dflash_extend_from_tensor(ggml_tensor * src_captures,
                                      int64_t       src_row_offset,
                                      int64_t       n_keep,
                                      int64_t       pos_start);

    // inline encoder (target-side execution): same encoder graph contents
    // as dflash_extend_from_tensor but executed on TARGET's scheduler instead
    // of DRAFT's. Called as
    //   target_ctx->dflash_inline_encode_from_ctx(draft_ctx, ...)
    // with src_captures resident in target's own buffer (no cross-context
    // D2D needed for the captures). The graph contents reference the draft
    // model's encoder weights and the draft context's side-store K/V
    // tensors directly; ggml-backend dispatches each op to whichever
    // backend instance owns the relevant buffer. Cross-context tensor
    // access (read for weights, write for ctx_K/V) requires both contexts
    // to share the same ggml_backend_t pointers — true for the standard
    // llama.cpp init path where backends are CUDA singletons per device.
    //
    // Returns 0 on success; negative on error. Falls back is the caller's
    // responsibility — speculative driver retries via the legacy path if
    // this returns < 0.
    int32_t dflash_inline_encode_from_ctx(llama_context * draft_ctx,
                                          ggml_tensor *   src_captures,
                                          int64_t         src_row_offset,
                                          int64_t         n_keep,
                                          int64_t         pos_start);

    // Reset the K/V side store to empty.
    void dflash_reset_ctx_kv();

    // skip the per-decode D2H of captured_features. Consumers must use
    // dflash_extend_from_tensor (or llama_dflash_extend_from_ctx) after
    // this is set.
    void set_dflash_skip_host_readback(bool skip) {
        dflash.skip_host_readback = skip;
    }

    // most recent packed-captures tensor produced by decode(),
    // valid until the next decode on this context.
    ggml_tensor * get_dflash_last_packed_captures() const {
        return dflash.last_packed_captures;
    }

    // One-shot D2H readback of the last_packed_captures into
    // dflash.captured_features. Used by consumers running in
    // skip_host_readback mode that occasionally need host bytes (alt
    // remap). Returns 0 on success, negative if no captures available.
    int32_t dflash_force_host_readback();

    // Non-const accessor to the dflash struct. Used by the inline-encoder
    // binding APIs (llama_dflash_bind_inline_side_store /
    // llama_dflash_set_inline_encode_state) to populate target-side state
    // from outside the class without needing friend access.
    llama_dflash & get_dflash_mut() { return dflash; }

    // Drop the cached decode-graph result so the next decode rebuilds it
    // from scratch. Called from llama_dflash_inline_advance_ctx_filled
    // when n_ctx_dft grows so the draft graph picks up the new side-store
    // size (matches the equivalent reset in dflash_extend()).
    void dflash_invalidate_graph_cache() {
        if (gf_res_prev) {
            gf_res_prev->reset();
        }
    }

    // Internal: drop the oldest `n_drop` columns of the side store.
    bool dflash_slide_left(int64_t n_drop);

    // DFlash tree-shaped attention mask
    void set_tree_mask(const uint8_t * visibility, int n_tree_tokens);
    void clear_tree_mask();

    llama_token * get_sampled_tokens() const;
    llama_token   get_sampled_token_ith(int32_t idx);

    float * get_sampled_logits_ith(int32_t idx);
    size_t  get_sampled_logits_count(int32_t idx);

    float * get_sampled_probs_ith(int32_t idx);
    size_t  get_sampled_probs_count(int32_t idx);

    const llama_token * get_sampled_candidates_ith(int32_t idx);
    size_t get_sampled_candidates_count(int32_t idx);

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales);

    bool set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end);

    // process a single ubatch with a specific graph type
    // if memory_context is provided, it will be applied first to the context's memory
    // ret contains the status of the graph computation
    // returns nullptr only if ret != GGML_STATUS_SUCCESS
    llm_graph_result * process_ubatch(
                const llama_ubatch & ubatch,
                    llm_graph_type   gtype,
            llama_memory_context_i * mctx,
                       ggml_status & ret);

    int encode(const llama_batch & batch_inp);
    int decode(const llama_batch & batch_inp);

    //
    // state save/load
    //

    size_t state_get_size();
    size_t state_get_data(      uint8_t * dst, size_t size);
    size_t state_set_data(const uint8_t * src, size_t size);

    size_t state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags);

    size_t state_seq_get_data(llama_seq_id seq_id,       uint8_t * dst, size_t size, llama_state_seq_flags flags);
    size_t state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags);

    bool state_load_file(
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    bool state_save_file(
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    size_t state_seq_load_file(
          llama_seq_id   seq_id,
            const char * filepath,
           llama_token * tokens_out,
                size_t   n_token_capacity,
                size_t * n_token_count_out);

    size_t state_seq_save_file(
          llama_seq_id   seq_id,
            const char * filepath,
     const llama_token * tokens,
                size_t   n_token_count);

    //
    // perf
    //

    llama_perf_context_data perf_get_data() const;
    void perf_reset();

    llama_memory_breakdown memory_breakdown() const;

    //
    // training
    //

    void opt_init(struct llama_model * model, struct llama_opt_params lopt_params);

    // TODO: more flexible combinations of logical/physical batch size and context size
    void opt_epoch(
            ggml_opt_dataset_t      dataset,
            ggml_opt_result_t       result_train,
            ggml_opt_result_t       result_eval,
            int64_t                 idata_split,
            ggml_opt_epoch_callback callback_train,
            ggml_opt_epoch_callback callback_eval);

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            llama_batch                    & batch,
            ggml_opt_epoch_callback          callback,
            bool                             train,
            int64_t                          idata_in_loop,
            int64_t                          ndata_in_loop,
            int64_t                          t_loop_start);

private:
    //
    // output
    //

    // Make sure enough space is available for outputs.
    // Returns max number of outputs for which space was reserved.
    uint32_t output_reserve(int32_t n_outputs);

    void output_reorder();

    // map the output row index `i` to batch index
    int64_t output_resolve_row(int32_t i) const;

    //
    // graph
    //

public:
    uint32_t graph_max_nodes(uint32_t n_tokens) const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // overload that runs the graph on a caller-provided scheduler. Used by
    // `dflash_extend` so the encoder graph runs on one of its dedicated
    // `sched_dflash_encode_slots[n_keep_pad]` rather than the regular
    // decode `sched`.
    ggml_status graph_compute(ggml_backend_sched_t sched_use, ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false, size_t * sizes = nullptr);

    bool set_sampler(llama_seq_id seq_id, llama_sampler * sampler);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    // Returns the per-tensor graph callback. Names tensors and pins
    // per-layer "norm" tensors to their layer's device on `sched_for_cb`,
    // so callers must pass the scheduler the graph will actually run on:
    // the main `sched` for the regular decoder, the chosen
    // `sched_dflash_encode_slots[n]` / `sched_dflash_encode_fallback`
    // for the DFlash encoder paths, and `sched_dflash_inline_encode`
    // for the inline encoder.
    llm_graph_cb graph_get_cb(ggml_backend_sched_t sched_for_cb) const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams cparams;

    llama_adapter_cvec_ptr  cvec;
    llama_adapter_loras_ptr loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    buffer_view<float> logits = {nullptr, 0};

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    buffer_view<float> embd = {nullptr, 0};

    struct sampling_info {
        // !samplers.empty() to check if any samplers are active
        std::map<llama_seq_id, llama_sampler *> samplers;

        buffer_view<float>       logits     = {nullptr, 0};
        buffer_view<llama_token> sampled    = {nullptr, 0};
        buffer_view<float>       probs      = {nullptr, 0};
        buffer_view<llama_token> candidates = {nullptr, 0};

        std::vector<uint32_t> logits_count;
        std::vector<uint32_t> probs_count;
        std::vector<uint32_t> candidates_count;

        // optimization
        std::vector<llama_token> token_ids_full_vocab;
    };

    sampling_info sampling;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // reuse the batch_allocr to avoid unnecessary memory allocations
    std::unique_ptr<llama_batch_allocr> balloc;

    uint32_t n_outputs = 0; // number of actually-used outputs in the current ubatch or last logical batch

    std::vector<int32_t> output_ids; // map batch token positions to ids of the logits and embd buffers

    struct swap_info {
        uint32_t i0;
        uint32_t i1;
    };

    std::vector<swap_info> output_swaps;

    ggml_backend_sched_ptr sched;

    bool sched_need_reserve = true;

    // Dedicated scheduler(s) for the DFlash encoder graph. Allocated
    // alongside `sched` for DFlash-arch contexts so the encoder + decoder
    // don't clobber each other's compute-buffer slot assignments. See
    // `gf_res_dflash_encode_slots` below for the per-`n_keep_pad`
    // cache-hit fast path that depends on this isolation.
    //
    // The vector is indexed by n_keep_pad (the per-call extend width).
    // Index 0 is unused; valid indices are [1, dflash_block_size]. Slots
    // are lazily populated by ensure_dflash_encode_slot() and pre-warmed
    // for all valid indices in the constructor (see the warmup loop near
    // sched_reserve()) so the steady-state extend never rebuilds.
    std::vector<ggml_backend_sched_ptr> sched_dflash_encode_slots;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // pointers and buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;
    std::vector<size_t>                     backend_buf_exp_size; // expected buffer sizes

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // DFlash drafter cross-context state. Populated by set_dflash_capture()
    // (capture install) and dflash_extend() (per-layer K/V side store).
    llama_dflash dflash;

    // DFlash tree-mode custom attention mask. Inactive (active==false) by default.
    llama_tree_mask tree_mask;

    // ggml contexts + backend buffers backing the DFlash K/V side store.
    // Only populated when the model arch is LLM_ARCH_DFLASH; empty otherwise.
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> dflash_kv_ctxs_bufs;

    // ggml contexts + backend buffers backing the per-GDN-layer state
    // history tensors. Populated on the TARGET context when
    // cparams.dflash_gdn_history is set and the model has recurrent
    // (GDN) layers; empty otherwise.
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> dflash_gdn_history_ctxs_bufs;

    // Encoder-graph result holders, one per `n_keep_pad` slot. Combined
    // with `sched_dflash_encode_slots`, this enables per-shape graph
    // caching: a slot is built once for its specific n_keep_pad and reused
    // across every subsequent extend call with the same width. Required
    // in chain mode where n_keep_pad varies between 1 and dflash_block_size
    // each iter (= 1 + accept_count) so consecutive extends typically have
    // different shapes.
    //
    // Index 0 is unused; valid indices are [1, dflash_block_size]. A slot
    // pointer being null means it has not been built yet. The vector is
    // sized once at the same time as the side-store K/V buffers (in
    // sched_reserve(), guarded by model.arch == LLM_ARCH_DFLASH).
    std::vector<llm_graph_result_ptr> gf_res_dflash_encode_slots;

    // Fallback single-slot encoder cache for `n_new > dflash_block_size`,
    // used by full-prompt prefill (which extends with n_new = prompt_length,
    // typically much larger than the per-iter block_size and a one-shot
    // call). Cache hits only when consecutive calls share the same n_new
    // value, otherwise rebuild. Lazy-init on first oversized extend.
    ggml_backend_sched_ptr sched_dflash_encode_fallback;
    llm_graph_result_ptr   gf_res_dflash_encode_fallback;
    int64_t                gf_res_dflash_encode_fallback_n_new = -1;

    // inline encoder (target-side). When cparams.dflash_inline_encoder is set,
    // the speculative driver runs the encoder graph on TARGET's scheduler
    // instead of the DRAFT's. These members live on the target context;
    // they mirror the sched_dflash_encode_slots[] / gf_res_dflash_encode_slots[]
    // pair above but are allocated using the target's backend_ptrs and
    // reference the draft model's encoder weights + draft context's side
    // store (ctx_K/V) cross-context. Single-slot, lazy-init on first call
    // to dflash_inline_encode_from_ctx().
    ggml_backend_sched_ptr sched_dflash_inline_encode;
    llm_graph_result_ptr   gf_res_dflash_inline_encode;
    int64_t                gf_res_dflash_inline_encode_n_new = -1;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

    // keep copies of the per-sequence memory on the device
    std::map<llama_seq_id, llama_memory_buffers> mem_storage;

    bool has_evaluated_once = false;

    // env: LLAMA_GRAPH_REUSE_DISABLE
    bool graph_reuse_disable = false;

    // perf
    mutable int64_t t_start_us  = 0;
    mutable int64_t t_load_us   = 0;
    mutable int64_t t_p_eval_us = 0;
    mutable int64_t t_eval_us   = 0;

    mutable int64_t t_compute_start_us = 0;
    mutable int64_t n_queued_tokens    = 0;

    mutable int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)
    mutable int32_t n_eval   = 0; // number of eval calls

    mutable int32_t n_reused = 0; // number of times the previous graph was reused
};
