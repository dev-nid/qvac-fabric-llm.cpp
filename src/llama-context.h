#pragma once

#include "ggml-cpp.h"
#include "ggml-opt.h"
#include "llama-adapter.h"
#include "llama-cparams.h"
#include "llama-graph.h"
#include "llama.h"

#include <atomic>
#include <map>
#include <vector>

struct llama_model;
class llama_batch_allocr;

class llama_io_read_i;
class llama_io_write_i;

// "memory" as in abstract memory for the context
struct llama_memory_i;
struct llama_memory_context_i;

// "memory" as in physical memory for a buffer type, in bytes
struct llama_memory_breakdown_data {
    size_t model   = 0; // memory allocated for the model
    size_t context = 0; // memory allocated for the context
    size_t compute = 0; // memory allocated for temporary compute buffers
};

struct llama_context {
    // init scheduler and compute buffers, reserve worst-case graphs
    llama_context(
            const llama_model & model,
                  llama_context_params params);

    ~llama_context();

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

    // -----------------------------------------------------------------------
    // DFlash: capture target hidden states for the drafter
    // -----------------------------------------------------------------------
    // Tee out target hidden states from the listed layer ids on every
    // llama_decode(). The capture is read by the speculative-decoding
    // driver via get_dflash_captured_features() and then pushed through
    // the encoder graph (dflash_extend) into the draft's K/V side store.
    void set_dflash_capture(const int32_t * layer_ids,
                            size_t          n_layer_ids,
                            int64_t         n_embd_target);

    // After decode, returns the captured features and (via out param) the
    // number of token positions covered. Returns nullptr if capture inactive.
    const float * get_dflash_captured_features(int64_t * n_outputs_out) const;

    // After decode on a draft context (LLM_ARCH_DFLASH), returns the
    // K candidate token IDs per output position from the in-graph top-K
    // (= ggml_argmax for K=1 / ggml_argsort_top_k for K>=2 over lm_head,
    // emitted by llm_build_dflash). Saves the per-decode
    // `bs * n_vocab * 4` byte float-logits PCIe transfer (~9.7 MiB per
    // block on Qwen3 vocab=151,936) in favour of a `K * bs * 4` byte
    // int32 read-back (~64 bytes per block at K=1, ~256 at K=4).
    // Returns nullptr on a non-DFlash context. The buffer is row-major
    // [n_outputs, K]: position i's K candidates live at indices
    // [i*K .. i*K+K-1], sorted descending by logit. *n_outputs_out is
    // filled with the number of positions covered (= n_outputs_all of
    // the most recent decode); *topk_out is filled with K.
    const int32_t * get_dflash_draft_topk(int64_t * n_outputs_out, uint32_t * topk_out) const;

    // Commit-33 helper: ggml_top_k returns the K largest indices in
    // unspecified order. Our consumers expect dflash.draft_topk[i*K+0]
    // to be the argmax. The decoder graph also emits a companion
    // ggml_argmax into dflash.draft_topk_argmax; this method runs a
    // tiny O(n_outputs * K) in-place pass to swap the argmax into slot
    // 0 of each row. No-op when K==1 (the K=1 fast path is plain
    // ggml_argmax — already at slot 0). Idempotent: calling it twice
    // performs only no-op swaps the second time around.
    //
    // Called from the public `llama_get_dflash_draft_topk` C wrapper
    // immediately after the existing `synchronize()` call, so the
    // async readbacks queued in `decode()` are guaranteed complete
    // before the swap runs. CPU-bound work (pointer chasing in a small
    // [n_outputs * K] array); ~microseconds even for K=4 + bs=16.
    void dflash_finalize_draft_topk();

    // Read access for the graph builders (passed via llm_graph_params.dflash).
    const llama_dflash * get_dflash() const;

    // Append `n_new` newly-committed target captures to the persistent K/V
    // side store. `target_hidden_new` is host memory laid out
    // [n_new, n_features] row-major; `pos_start` is the absolute position
    // of the first new token (positions [pos_start..pos_start+n_new-1] are
    // used for K's RoPE).
    //
    // Runs the DFlash encoder graph (llm_build_dflash_encode) once, which
    // applies fc + hidden_norm + per-layer wk/wv (+k_norm +RoPE on K) and
    // ggml_cpy's the result into ctx_K[il] / ctx_V[il] at column offset
    // dflash.ctx_filled. On success, advances dflash.ctx_filled by n_new.
    //
    // Returns 0 on success, non-zero on failure.
    int32_t dflash_extend(const float * target_hidden_new,
                          int64_t       n_new,
                          int64_t       pos_start);

    // Reset the K/V side store to empty (e.g. on a new prompt).
    void dflash_reset_ctx_kv();

    // Internal: drop the oldest `n_drop` columns of the per-layer K/V side
    // store, shifting the surviving columns left to offset 0. Bumps
    // `dflash.ctx_pos_base` by n_drop and decrements `dflash.ctx_filled`
    // by n_drop. Used by dflash_extend() when a new write would exceed
    // dflash.ctx_capacity. Returns true on success.
    bool dflash_slide_left(int64_t n_drop);

    // -----------------------------------------------------------------------
    // DDTree (DFlash Phase 2): tree-shaped attention mask
    // -----------------------------------------------------------------------
    // Stage A scope: enable callers to install a custom [n × n] visibility
    // matrix that overrides the seq-id-based attention mask for the next
    // verification decode. The override is performed in
    // `llm_graph_input_attn_kv::set_input` AFTER the standard mask has been
    // written, so prefix tokens already in the KV cache get the regular
    // causal/seq-id treatment and only the new tokens (the tree itself)
    // are patched. When no tree mask is installed (the default), the
    // override block is a no-op and behaviour is bit-identical to the
    // pre-Phase-2 code.
    //
    // The actual tree builder, the tree-aware accept walk, and the
    // per-branch KV rollback live in `common/speculative.cpp` and ship in
    // Stages B and C — see `logs/core_architecture/09_lucebox_reference.md`
    // item B Phase 2 notes.
    void set_tree_mask(const uint8_t * visibility, int n_tree_tokens);
    void clear_tree_mask();

    void attach_threadpool(
            ggml_threadpool_t threadpool,
            ggml_threadpool_t threadpool_batch);

    void detach_threadpool();

    void set_n_threads(int32_t n_threads, int32_t n_threads_batch);

    void set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data);

    void set_embeddings (bool value);
    void set_causal_attn(bool value);
    void set_warmup(bool value);

    void set_adapter_lora(
            llama_adapter_lora * adapter,
            float scale);

    bool rm_adapter_lora(
            llama_adapter_lora * adapter);

    void clear_adapter_lora();

    bool apply_adapter_cvec(
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

    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> memory_breakdown() const;

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
            ggml_opt_epoch_callback callback_eval,
            int64_t                 resume_from_batch = -1);

    // Optimizer state access for checkpointing (delegated to ggml_opt API)
    int64_t opt_get_iter();

    // Optimizer state persistence
    bool opt_save_state(const char* filename);
    bool opt_load_state(const char* filename);

    // Clean up optimizer context to free memory and allow reinitialization
    void opt_cleanup();

    // Request early exit from training epoch (thread-safe)
    void opt_request_stop();

    // Reset the stop flag to allow training to continue
    void opt_reset_stop();

    void opt_epoch_iter(
            ggml_opt_dataset_t               dataset,
            ggml_opt_result_t                result,
            const std::vector<llama_token> & tokens,
            const std::vector<llama_token> & labels_sparse,
            const std::vector<int32_t>     & masks_sparse,
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

    //
    // graph
    //

public:
    uint32_t graph_max_nodes() const;

    // can reuse the llm_graph_result instance of the context (for example to update a memory module)
    llm_graph_result * get_gf_res_reserve() const;

    // returns the result of ggml_backend_sched_graph_compute_async execution
    ggml_status graph_compute(ggml_cgraph * gf, bool batched);

    // overload that runs the graph on a caller-provided scheduler instead
    // of the default `sched` member. Used by `dflash_extend` so the
    // encoder graph runs on its dedicated `sched_dflash_encode`.
    // Threadpool / n_threads setup is identical to the default overload.
    ggml_status graph_compute(ggml_backend_sched_t sched_use, ggml_cgraph * gf, bool batched);

    // reserve a graph with a dummy ubatch of the specified size
    ggml_cgraph * graph_reserve(uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only = false);

private:
    llm_graph_params graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const;

    llm_graph_cb graph_get_cb() const;

    // TODO: read/write lora adapters and cvec
    size_t state_write_data(llama_io_write_i & io);
    size_t state_read_data (llama_io_read_i  & io);

    size_t state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags);
    size_t state_seq_read_data (llama_io_read_i  & io, llama_seq_id seq_id, llama_state_seq_flags flags);

    //
    // members
    //

    const llama_model & model;

    llama_cparams       cparams;
    llama_adapter_cvec  cvec;
    llama_adapter_loras loras;

    llama_cross cross; // TODO: tmp for handling cross-attention - need something better probably

    std::unique_ptr<llama_memory_i> memory;

    // decode output (2-dimensional array: [n_outputs][n_vocab])
    size_t  logits_size = 0; // capacity (of floats) for logits
    float * logits      = nullptr;

    // embeddings output (2-dimensional array: [n_outputs][n_embd])
    // populated only when pooling_type == LLAMA_POOLING_TYPE_NONE
    size_t  embd_size = 0; // capacity (of floats) for embeddings
    float * embd      = nullptr;

    // sequence embeddings output (map of [n_embd] vectors)
    // populated only when pooling_type != LLAMA_POOLING_TYPE_NONE
    std::map<llama_seq_id, std::vector<float>> embd_seq;

    // DFlash drafter cross-context state.
    // Populated by set_dflash_capture() (capture install) and
    // dflash_extend() (per-layer K/V side store population). Consumed
    // by llm_graph_input_dflash via the llama_dflash * forwarded
    // through llm_graph_params.
    llama_dflash dflash;

    // DDTree custom attention mask. Inactive (active==false) by default;
    // `set_tree_mask` flips active and copies the visibility matrix.
    // `graph_params()` forwards `tree_mask.active ? &tree_mask : nullptr`
    // into `llm_graph_params.tree_mask`, which the attention input class
    // picks up at construction. See `set_tree_mask` doc above.
    llama_tree_mask tree_mask;

    // ggml contexts + backend buffers backing the DFlash K/V side store.
    // One context per buffer-type (== one per device when the model is
    // split across GPUs); the same convention as llama_kv_cache_unified
    // uses for the regular KV cache. Only populated when the model arch
    // is LLM_ARCH_DFLASH; empty otherwise.
    std::vector<std::pair<ggml_context_ptr, ggml_backend_buffer_ptr>> dflash_kv_ctxs_bufs;

    // Encoder-graph result holder, kept across `llama_dflash_extend()`
    // calls so the same `llm_graph_result` instance is reused. Combined
    // with the dedicated `sched_dflash_encode` (above), this enables
    // per-`n_new` graph caching: when consecutive extends share the same
    // `n_new` value, `dflash_extend` skips `res->reset + rebuild +
    // sched_alloc_graph` and just re-runs `set_inputs + graph_compute` on
    // the already-built graph. The dedicated encoder scheduler is what
    // makes this safe — without it, the decoder's `sched_reset +
    // sched_alloc_graph` calls on the regular `sched` would clobber the
    // encoder's compute-buffer slot assignments and the cache-hit path
    // would crash on stale device pointers.
    //
    // `gf_res_dflash_encode_n_new` is the `n_new` value the cached graph
    // was built for. -1 means "not built yet"; on a `dflash_extend` call
    // with a different `n_new`, the cache misses and the field is
    // updated. Reset on `dflash_reset_ctx_kv()` is unnecessary — the
    // topology only depends on `n_new` (write_offset is a runtime input
    // via `pos_idx`), and the side-store tensor objects survive across
    // resets, so a cached graph stays valid across prompts.
    //
    // Cache hit rate observed on Qwen3-4B-DFlash speculative decoding:
    // 20-35% per generation (varies by prompt: math ~33%, code ~21%,
    // NL ~29% on 128-token generations). The wall-clock impact is
    // perf-neutral within run-to-run noise on Vulkan + 2x RTX 5090 —
    // the savings per cache hit (skipping a small graph build + a
    // ggml-alloc pass) are dominated by the GPU compute time of the
    // encoder graph itself. The architecture is in place for future
    // workloads where the saved overhead matters more (slower CPUs,
    // larger draft models, etc.).
    llm_graph_result_ptr gf_res_dflash_encode;
    int64_t              gf_res_dflash_encode_n_new = -1;

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

    // Dedicated scheduler for the DFlash encoder graph
    // (`llm_build_dflash_encode`). Lives in parallel with `sched` so that
    // each `dflash_extend` call's compute-buffer slot assignments survive
    // across intervening decoder runs on `sched` — without a separate
    // scheduler, every decoder `sched_reset + sched_alloc_graph` would
    // clobber the encoder's allocations and the per-`n_new` cache shortcut
    // in `dflash_extend` (which skips re-build + re-alloc on cache hit)
    // would crash on stale device pointers. Lazy-initialised on the first
    // dflash_extend call (only meaningful for LLM_ARCH_DFLASH contexts).
    ggml_backend_sched_ptr sched_dflash_encode;

    ggml_backend_t backend_cpu = nullptr;
    std::vector<ggml_backend_ptr> backends;

    // training
    ggml_opt_context_t opt_ctx = nullptr;
    uint32_t original_n_ctx_train = 0;

    // optimizer state loading (deferred until after ggml_opt_build)
    std::string pending_optimizer_checkpoint_path;
    bool should_load_optimizer_tensors = false;
    bool optimizer_tensors_loaded = false;
    ggml_opt_loss_type opt_loss_type = GGML_OPT_LOSS_TYPE_CROSS_ENTROPY;

    // early exit flag for training epochs (thread-safe)
    std::atomic<bool> training_should_stop{ false };

    ggml_threadpool_t threadpool       = nullptr;
    ggml_threadpool_t threadpool_batch = nullptr;

    ggml_abort_callback abort_callback      = nullptr;
    void *              abort_callback_data = nullptr;

    std::vector<std::pair<ggml_backend_t, ggml_backend_set_n_threads_t>> set_n_threads_fns;

    // buffer types used for the compute buffer of each backend
    std::vector<ggml_backend_t>             backend_ptrs;
    std::vector<ggml_backend_buffer_type_t> backend_buft;

    llm_graph_result_ptr gf_res_prev;
    llm_graph_result_ptr gf_res_reserve;

    // host buffer for the model output (logits and embeddings)
    ggml_backend_buffer_ptr buf_output;

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
