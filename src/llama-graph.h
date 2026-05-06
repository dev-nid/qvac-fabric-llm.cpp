#pragma once

#include "llama-arch.h"
#include "llama-batch.h"
#include "llama-hparams.h"
#include "llama-adapter.h"

#include <cstdint>
#include <vector>
#include <memory>
#include <set>
#include <functional>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct llama_cparams;

struct llama_memory_context_i;

class llama_kv_cache_context;
class llama_kv_cache_iswa_context;
class llama_memory_recurrent_context;
class llama_memory_hybrid_context;

// certain models (typically multi-modal) can produce different types of graphs
enum llm_graph_type {
    LLM_GRAPH_TYPE_DEFAULT,
    LLM_GRAPH_TYPE_ENCODER,
    LLM_GRAPH_TYPE_DECODER,
};

enum llm_ffn_op_type {
    LLM_FFN_SILU,
    LLM_FFN_GELU,
    LLM_FFN_RELU,
    LLM_FFN_RELU_SQR,
    LLM_FFN_SWIGLU,
    LLM_FFN_GEGLU,
    LLM_FFN_REGLU,
    LLM_FFN_SWIGLU_OAI_MOE,
};

enum llm_ffn_gate_type {
    LLM_FFN_SEQ,
    LLM_FFN_PAR, // ffn_gate is parallel to ffn_up
};

enum llm_norm_type {
    LLM_NORM,
    LLM_NORM_RMS,
    LLM_NORM_GROUP,
};

// TODO: tmp - need something better to pass the data from the encoder to the decoder
struct llama_cross {
    // the output embeddings from the encoder as a ggml tensor
    // TODO: this needs more work to be correct, for now copy the embeddings data to host memory
    //       ref: https://github.com/ggml-org/llama.cpp/pull/11213#discussion_r1969892524
    //ggml_tensor * t_embd = nullptr;

    int64_t n_embd = 0;
    int64_t n_enc  = 0;

    // embeddings data copied to host memory (tmp)
    std::vector<float> v_embd;

    // needed to construct the cross-attention mask in the decoder
    std::vector<std::set<llama_seq_id>> seq_ids_enc;
};

// DFlash cross-context state.
//
// Two roles, one struct:
//
//  (a) On the *draft* context — the speculative driver pushes per-block
//      target hidden states into the per-layer K/V side store via
//      llama_dflash_extend(). The drafter graph (llm_build_dflash) reads
//      `n_ctx` (the side store's currently-filled column count) and the
//      side store tensors `ctx_K[il]` / `ctx_V[il]` to build the
//      bidirectional cross-attention's K and V via zero-copy views.
//      The encoder graph (llm_build_dflash_encode) reads `n_features`
//      (= `n_target_layer_ids * n_embd_target`) when sizing its
//      target_hidden_new input.
//
//  (b) On the *target* context — the driver writes capture_layer_ids /
//      capture_n_embd via llama_set_dflash_capture() before any decode. The
//      target graph (e.g. llm_build_qwen3) tees out hidden states at those
//      layer indices, and after compute the context fills captured_features
//      and captured_n_outputs for the driver to read.
//
// The two roles operate on different contexts; the same struct is used so a
// single llama_dflash * forwarded through llm_graph_params lets the graph
// builder discover whichever role applies on its own context.
struct llama_dflash {
    // ---------- (a) drafter sizing ----------
    int64_t n_features = 0;       // per-token feature dim (n_target_layer_ids * n_embd_target)
    int64_t n_ctx      = 0;       // K_ctx column count seen by the drafter's
                                  // attention. Seeded from `cparams.n_ctx_seq`
                                  // at construction (worst-case for graph
                                  // reserve) and kept in sync with `ctx_filled`
                                  // by `dflash_extend()` at runtime, so the
                                  // kq_mask shape always matches the side
                                  // store view.

    // ---------- (b) target capture ----------
    // Layer indices (into the target model) whose post-block hidden states
    // should be teed out per decode. Empty = no capture.
    std::vector<int32_t> capture_layer_ids;

    // Per-layer hidden dimension of the *target* model.
    // (= hparams.n_embd of the target context.)
    int64_t capture_n_embd = 0;

    // After each decode, populated with [n_outputs, n_features] in row-major,
    //   captured_features[i_token * n_features + i_feat]
    // where n_features = capture_layer_ids.size() * capture_n_embd.
    std::vector<float> captured_features;
    int64_t            captured_n_outputs = 0;

    // Per-layer host staging buffers used by the post-decode read-back path
    // in llama_context::decode(). Each entry is sized
    //   capture_staging[il].size() == n_outputs_all * capture_n_embd
    // and holds the layer's contribution in column-per-token layout
    // (= the on-GPU tensor's native layout, contiguous after a single
    // tensor_get_async). After all ubatches have been read into staging
    // we sync the scheduler once and then host-transpose into
    // `captured_features` (row-per-token, all-layers-side-by-side
    // layout that the encoder graph in `dflash_extend()` expects).
    //
    // This batches the D2H reads from `n_layers * n_outputs` separate
    // calls (one per (layer, token) pair) down to `n_layers` per ubatch,
    // amortising the per-call backend submission overhead over more data
    // (~16x fewer transactions on Qwen3-4B-DFlash with bs=16). The
    // host-side transpose is a small set of fixed-stride memcpys and is
    // negligible vs even one async-get's submission overhead.
    std::vector<std::vector<float>> capture_staging;

    // ---------- (a') drafter inline top-K read-back ----------
    // After each decode on a draft context whose graph emitted
    // `res->t_dflash_topk` (i.e. LLM_ARCH_DFLASH; see
    // `src/models/dflash.cpp::llm_build_dflash`), this holds the K
    // candidate token ids per output position. Row-major
    // `[draft_topk_n_outputs, draft_topk_K]` int32: position i's
    // candidates live at indices [i*K .. i*K+K-1]. The slot `[i*K+0]` is
    // always the argmax — for K>=2 the graph emits both `ggml_top_k(K)`
    // (the K candidates, in unspecified order) and `ggml_top_k(1)` (the
    // argmax) and a tiny O(n_outputs * K) post-pass in
    // llama_context::decode swaps whichever top-K slot matches the
    // argmax into position 0. For K=1 (default; chain mode) the graph
    // emits a single `ggml_top_k(1)` whose output IS the argmax at slot
    // 0 — no swap needed. Sized by the post-decode loop in
    // llama_context::decode(). When `t_dflash_topk` was nullptr (e.g. on
    // the target context), this stays empty and `draft_topk_n_outputs`
    // stays 0. `draft_topk_K` mirrors `cparams.dflash_topk` at decode
    // time.
    //
    // History:
    //  * Pre-commit-33 the K>=2 graph used ggml_argsort_top_k (full
    //    O(V·log V) sort, 9.1% of total time on CPU at K=2).
    //  * Pre-commit-34 the K=1 path AND the K>=2 companion used
    //    ggml_argmax (single-threaded CPU kernel — see
    //    ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_argmax_f32:1567,
    //    `if (params->ith != 0) return;`). Both replaced with
    //    ggml_top_k, which parallelizes across the n_outputs rows.
    std::vector<int32_t> draft_topk;
    int64_t              draft_topk_n_outputs = 0;
    uint32_t             draft_topk_K         = 0;

    // Companion to draft_topk for the K>=2 path (commit 33; in commit 34
    // its source op changed from ggml_argmax to ggml_top_k(K=1)). Holds
    // the argmax of each position's logits so the post-decode swap pass
    // can restore the "draft_topk[i*K+0] is the argmax" invariant.
    // Layout: [draft_topk_n_outputs] int32. Empty (and unused) when K=1
    // because that fast path's draft_topk already IS the argmax —
    // no swap needed.
    std::vector<int32_t> draft_topk_argmax;

    // ---------- (c) per-layer K/V side store (paper §4.1 reuse) ----------
    // Persistent per-layer K_ctx and V_ctx tensors, allocated as backend
    // tensors at draft-context creation time and surviving across
    // llama_decode() calls. Layout per layer:
    //   ctx_K[il]: [n_embd_k_gqa, ctx_capacity]  (post wk + k_norm + RoPE)
    //   ctx_V[il]: [n_embd_v_gqa, ctx_capacity]  (post wv, no norm/no RoPE)
    // The encoder graph (llm_build_dflash_encode) computes K/V for newly
    // committed target features and writes them at offset `ctx_filled`,
    // bumping `ctx_filled` by `n_new` per call. The decoder graph
    // (llm_build_dflash) reads `n_ctx` columns via ggml_view_*().
    std::vector<ggml_tensor *> ctx_K;
    std::vector<ggml_tensor *> ctx_V;
    int64_t ctx_filled   = 0;
    int64_t ctx_capacity = 0;
    // Position of the first cached ctx token within the global sequence.
    // Always 0 for now; will become non-zero when sliding-window cap is
    // ported (per buun fork's GGML_DFLASH_MAX_CTX).
    int64_t ctx_pos_base = 0;
};

// DDTree (DFlash Phase 2): custom attention mask for tree-shaped verification
// batches. Stage A scope = the libllama plumbing only; the tree builder, the
// tree-aware accept walk, and the per-branch KV rollback all live in
// `common/speculative.cpp` and ship in later stages (B, C — see
// `logs/core_architecture/09_lucebox_reference.md` item B).
//
// When `active == false` (the default and the only state Stage A constructs),
// `llm_graph_input_attn_kv::set_input` runs the standard seq-id based mask
// path unchanged — i.e. a no-op for every existing graph build.
//
// When `active == true`, the standard mask runs first (so the prefix tokens
// already in the KV cache get the regular causal/seq-id treatment), then
// the top-left `n_tree_tokens × n_tree_tokens` block of the freshly written
// mask is overwritten with `visibility[i*n+j] ? 0.0f : -INFINITY`. This
// pattern (override the tree block AFTER the standard mask) was lifted as a
// design idea from `alternate/buun-llama-cpp/src/llama-graph.cpp:455-468`;
// no code is copied. Caller invariants:
//   * All `n_tree_tokens` tree nodes are submitted in a single ubatch with
//     `seq_id = {0}`, positions = `n_past - 1 + tree.depths[i]`.
//   * `visibility` is row-major `n_tree_tokens²`; `visibility[i*n+j]` means
//     "tree node i is allowed to attend to tree node j" (parent-pointer
//     reachability).
//   * `n_tree_tokens == ubatch->n_tokens` is asserted in `set_input`.
//
// The struct lives on `llama_context` (one per context); the C API
// (`llama_set_tree_mask` / `llama_clear_tree_mask`) is the only way to
// toggle `active`. Memory ownership: `visibility` owns its bytes; the
// caller's pointer is copied at `set_tree_mask` time.
struct llama_tree_mask {
    bool                 active        = false;
    int                  n_tree_tokens = 0;
    std::vector<uint8_t> visibility;
};

struct llm_graph_params;

//
// llm_graph_input
//

class llm_graph_input_i {
public:
    llm_graph_input_i() {
        const char * LLAMA_GRAPH_INPUT_DEBUG = getenv("LLAMA_GRAPH_INPUT_DEBUG");
        debug = LLAMA_GRAPH_INPUT_DEBUG ? atoi(LLAMA_GRAPH_INPUT_DEBUG) : 0;
    }

    virtual ~llm_graph_input_i() = default;

    virtual void set_input(const llama_ubatch * ubatch) = 0;

    // return true if the resulting input tensors using the provided graph parameters would be
    //   the same as the previous input tensors that we have currently stored in the object
    virtual bool can_reuse(const llm_graph_params & params) {
        // returning false here by default will prevent from reusing the graph if the check
        //   for the input type has not been implemented yet
        GGML_UNUSED(params);
        return false;
    }
protected:
    // env: LLAMA_GRAPH_INPUT_DEBUG
    int debug = 0;
};

using llm_graph_input_ptr = std::unique_ptr<llm_graph_input_i>;

class llm_graph_input_embd : public llm_graph_input_i {
public:
    llm_graph_input_embd()          = default;
    virtual ~llm_graph_input_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens = nullptr; // I32 [n_batch]
    ggml_tensor * embd   = nullptr; // F32 [n_embd, n_batch]
};

class llm_graph_input_pos : public llm_graph_input_i {
public:
    llm_graph_input_pos(uint32_t n_pos_per_embd) : n_pos_per_embd(n_pos_per_embd) {}
    virtual ~llm_graph_input_pos() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * pos = nullptr; // I32 [n_batch]

    const uint32_t n_pos_per_embd = 1;
};

// temperature tuning, used by llama4
class llm_graph_input_attn_temp : public llm_graph_input_i {
public:
    llm_graph_input_attn_temp(uint32_t n_attn_temp_floor_scale, float f_attn_temp_scale)
        : n_attn_temp_floor_scale(n_attn_temp_floor_scale), f_attn_temp_scale(f_attn_temp_scale) {}
    virtual ~llm_graph_input_attn_temp() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * attn_scale = nullptr; // F32 [n_batch]

    const uint32_t n_attn_temp_floor_scale;
    const float    f_attn_temp_scale;
};

class llm_graph_input_pos_bucket : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket(const llama_hparams & hparams) : hparams(hparams) {}
    virtual ~llm_graph_input_pos_bucket() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_batch, n_batch]

    const llama_hparams hparams;
};

class llm_graph_input_pos_bucket_kv : public llm_graph_input_i {
public:
    llm_graph_input_pos_bucket_kv(
            const llama_hparams & hparams,
            const llama_kv_cache_context * mctx) : hparams(hparams), mctx(mctx) {}
    virtual ~llm_graph_input_pos_bucket_kv() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * pos_bucket = nullptr; // I32 [n_kv, n_batch]

    const llama_hparams hparams;

    const llama_kv_cache_context * mctx;
};

class llm_graph_input_out_ids : public llm_graph_input_i {
public:
    llm_graph_input_out_ids(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            uint32_t n_outputs) : hparams(hparams), cparams(cparams), n_outputs(n_outputs) {}
    virtual ~llm_graph_input_out_ids() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * out_ids; // I32 [n_outputs]

    const llama_hparams hparams;
    const llama_cparams cparams;

    const uint32_t n_outputs;
};

class llm_graph_input_mean : public llm_graph_input_i {
public:
    llm_graph_input_mean(const llama_cparams & cparams) : cparams(cparams) {}
    virtual ~llm_graph_input_mean() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * mean; // F32 [n_batch, n_batch]

    const llama_cparams cparams;
};

class llm_graph_input_cls : public llm_graph_input_i {
public:
    llm_graph_input_cls(const llama_cparams & cparams, const llm_arch arch) : cparams(cparams), arch(arch) {}
    virtual ~llm_graph_input_cls() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cls; // I32 [n_batch]

    const llama_cparams cparams;
    const llm_arch arch;
};

class llm_graph_input_rs : public llm_graph_input_i {
public:
    llm_graph_input_rs(const llama_memory_recurrent_context * mctx) : mctx(mctx) {}
    virtual ~llm_graph_input_rs() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * s_copy;  // I32 [n_rs]

    // views of s_copy, computed once per graph
    // and shared across layers which use build_rs
    ggml_tensor * s_copy_main;   // I32 [n_seqs]
    ggml_tensor * s_copy_extra;  // I32 [n_rs - n_seqs]

    const llama_memory_recurrent_context * mctx;
};

class llm_graph_input_cross_embd : public llm_graph_input_i {
public:
    llm_graph_input_cross_embd(
            const llama_cross * cross) : cross(cross) {}
    virtual ~llm_graph_input_cross_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * cross_embd; // F32 [n_embd, n_outputs_enc]

    const llama_cross * cross;
};

// DFlash drafter context input.
// Builds the non-causal kq_mask for the drafter's cross-attention over
// the [side-store K/V | proposal] segments. Owned by the graph; reads
// committed-prefix sizing (`dflash->n_ctx`) from the llama_dflash struct
// on the context. The K_ctx / V_ctx tensors themselves are zero-copy
// views of the per-layer side store and are built directly in
// llm_build_dflash, not staged through this input class.
class llm_graph_input_dflash : public llm_graph_input_i {
public:
    llm_graph_input_dflash(const llama_dflash * dflash, int64_t n_ctx, int64_t n_block)
        : dflash(dflash), n_ctx(n_ctx), n_block(n_block) {}
    virtual ~llm_graph_input_dflash() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * kq_mask     = nullptr; // F32 [n_ctx + n_block, n_block_pad, 1, 1]
    ggml_tensor * kq_mask_cnv = nullptr; // f16 cast for flash_attn

    const llama_dflash * dflash;
    int64_t              n_ctx;
    int64_t              n_block;
};

class llm_graph_input_attn_no_cache : public llm_graph_input_i {
public:
    llm_graph_input_attn_no_cache(const llama_hparams & hparams, const llama_cparams & cparams) :
        hparams(hparams),
        cparams(cparams) {
    }
    ~llm_graph_input_attn_no_cache() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    // n_tokens == n_batch
    ggml_tensor * self_kq_mask         = nullptr; // F32 [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //     [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32 [n_tokens, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //     [n_tokens, n_batch/n_stream, 1, n_stream]

    const llama_hparams hparams;
    const llama_cparams cparams;
};

class llm_graph_input_attn_kv : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx,
            const llama_tree_mask * tree_mask = nullptr) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx),
        tree_mask(tree_mask) {
    }
    ~llm_graph_input_attn_kv() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }
    ggml_tensor * get_v_idxs() const { return self_v_idxs; }

    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    ggml_tensor * self_kq_mask     = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //     [n_kv, n_batch/n_stream, 1, n_stream]

    // note: these have to be copies because in order to be able to reuse a graph, its inputs
    //       need to carry these parameters with them. otherwise, they can point to freed
    //       llm_graph_params from a previous batch, causing stack-use-after-return
    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;

    // DDTree: when non-null AND `tree_mask->active`, set_input overwrites
    // the [n_tree_tokens × n_tree_tokens] block of the freshly written
    // attention mask with `visibility[i*n+j] ? 0.0f : -INFINITY`. Pointer
    // borrowed from llama_context::tree_mask (which owns the storage); the
    // pointer's address is stable across decode() calls. nullptr (the
    // default) means "no override — Stage A no-op".
    const llama_tree_mask * tree_mask = nullptr;
};

class llm_graph_input_attn_kv_iswa : public llm_graph_input_i {
public:
    llm_graph_input_attn_kv_iswa(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_iswa_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_kv_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs()     const { return self_k_idxs; }
    ggml_tensor * get_v_idxs()     const { return self_v_idxs; }
    ggml_tensor * get_k_idxs_swa() const { return self_k_idxs_swa; }
    ggml_tensor * get_v_idxs_swa() const { return self_v_idxs_swa; }

    ggml_tensor * get_kq_mask()     const { return self_kq_mask_cnv; }
    ggml_tensor * get_kq_mask_swa() const { return self_kq_mask_swa_cnv; }

    ggml_tensor * self_k_idxs     = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs     = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]
    ggml_tensor * self_k_idxs_swa = nullptr; // I64 [n_batch]
    ggml_tensor * self_v_idxs_swa = nullptr; // I64 [n_batch] or [n_batch*n_embd_v_gqa]

    ggml_tensor * self_kq_mask         = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv     = nullptr; //     [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa     = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_swa_cnv = nullptr; //     [n_kv, n_batch/n_stream, 1, n_stream]

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_iswa_context * mctx;
};

class llm_graph_input_attn_cross : public llm_graph_input_i {
public:
    llm_graph_input_attn_cross(const llama_cross * cross) : cross(cross) {}
    ~llm_graph_input_attn_cross() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * get_kq_mask_cross() const { return cross_kq_mask_cnv; }

    ggml_tensor * cross_kq_mask     = nullptr; // F32 [n_outputs_enc, n_batch, 1, 1]
    ggml_tensor * cross_kq_mask_cnv = nullptr; // F32 [n_outputs_enc, n_batch, 1, 1]

    const llama_cross * cross = nullptr;
};

class llm_graph_input_mem_hybrid : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid(
            std::unique_ptr<llm_graph_input_attn_kv> inp_attn,
            std::unique_ptr<llm_graph_input_rs>              inp_rs,
            const llama_memory_hybrid_context *              mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid() = default;

    void set_input(const llama_ubatch * ubatch) override;

    std::unique_ptr<llm_graph_input_attn_kv> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_kv * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const llama_memory_hybrid_context * mctx;
};

//
// llm_graph_result
//

// these objects deliver the result from the graph build process back to the llama_context
// note that the input tensors created for the graph are referenced here - the goal is to be able to populate their
//   specific data, by calling the set_inputs() method
// along with the input tensors, the object also provides commonly used outputs tensors, such as logits, embeddings, etc.
//   these are used by the llama_context to extact the relevant data, based on the compute parameters

// callback that allows us to apply custom logic to each tensor (e.g. ggml-alloc, offloading, etc.)
using llm_graph_cb = std::function<void(const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il)>;

class llm_graph_result;

struct llm_graph_params {
    llm_arch arch = LLM_ARCH_UNKNOWN;

    llama_hparams hparams;
    llama_cparams cparams;

    llama_ubatch ubatch; // note: intentionally make a copy

    llm_graph_type gtype;

    ggml_backend_sched_t sched;
    ggml_backend_t backend_cpu;

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;
    const llama_dflash           * dflash;    // DFlash drafter state (capture, K/V side store)
    const llama_tree_mask        * tree_mask; // DDTree custom attention mask (nullptr = no override)

    uint32_t n_outputs;

    llm_graph_cb cb;

    llm_graph_result * res;

    // return true if the "other" params would result in a graph with the same topology as with the current params
    //   having the same topology allows us to reuse the graph in some cases
    bool allow_reuse(const llm_graph_params & other) const {
        // first check the ubatch
        bool can_reuse_ubatch =
            ubatch.equal_seqs() == other.ubatch.equal_seqs() &&
            ubatch.n_tokens     == other.ubatch.n_tokens &&
            ubatch.n_seq_tokens == other.ubatch.n_seq_tokens &&
            ubatch.n_seqs       == other.ubatch.n_seqs &&
            ubatch.n_seqs_unq   == other.ubatch.n_seqs_unq &&
            (
                (!ubatch.token && !other.ubatch.token) ||
                (!ubatch.embd  && !other.ubatch.embd)
            );

        // when we split the batch using "equal_seqs" we have to verify that the participating sequences are the same
        //   the reason is because the set of attention streams would be different for different sequences
        if (can_reuse_ubatch && ubatch.equal_seqs()) {
            if (!ubatch.data) {
                // if the old ubatch does not own it's data, then we cannot guarantee that it is still alive, and
                //   therefore we cannot perform the sequence id check. normally should never happen
                can_reuse_ubatch = false;
            } else {
                for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                    can_reuse_ubatch &= ubatch.seq_id_unq[s] == other.ubatch.seq_id_unq[s];
                }
            }
        }

        if (!can_reuse_ubatch) {
            return false;
        }

        return
            cparams.embeddings  == other.cparams.embeddings  &&
            cparams.causal_attn == other.cparams.causal_attn &&
            arch      == other.arch  &&
            gtype     == other.gtype &&
            cvec      == other.cvec  &&
            loras     == other.loras &&
            cross     == other.cross &&
            dflash    == other.dflash &&
            tree_mask == other.tree_mask &&
            n_outputs == other.n_outputs;
    }
};

class llm_graph_result {
public:
    llm_graph_result(int64_t max_nodes);

    virtual ~llm_graph_result() = default;

    ggml_tensor * get_tokens()      const { return t_tokens; }
    ggml_tensor * get_logits()      const { return t_logits; }
    ggml_tensor * get_embd()        const { return t_embd; }
    ggml_tensor * get_embd_pooled() const { return t_embd_pooled; }

    ggml_cgraph  * get_gf()  const { return gf; }
    ggml_context * get_ctx() const { return ctx_compute.get(); }

    int64_t get_max_nodes() const;

    void reset();

    void set_inputs(const llama_ubatch * ubatch);

    // try to update the existing graph result using the new graph parameters in order to reuse it
    // this can only be done if we determine that the resulting graph using the new graph parameters
    //   would be identical to the existing graph. in that case, we simply have to update the memory
    //   contexts of the input tensors of the graph and we can reuse it for another computation
    // return true if the graph was updated and can be reused
    bool can_reuse(const llm_graph_params & params);

    llm_graph_input_i * add_input(llm_graph_input_ptr input);

    void set_params(const llm_graph_params & params);

    // important graph nodes
    ggml_tensor * t_tokens      = nullptr;
    ggml_tensor * t_logits      = nullptr;
    ggml_tensor * t_embd        = nullptr;
    ggml_tensor * t_embd_pooled = nullptr;

    // DFlash drafter: in-graph top-K candidate token IDs of the per-position
    // lm_head logits. Shape:
    //   K=1 (default): I32 [1, n_outputs]   (ggml_top_k(K=1) — argmax)
    //   K>=2:          I32 [K, n_outputs]   (ggml_top_k    — UNSORTED, see below)
    // Both shapes are n_outputs * K contiguous int32s in memory, drop-in
    // compatible for the flat byte-stride readback path.
    //
    // When this is non-null, llm_build_dflash sets t_logits = nullptr — the
    // float-logits read-back is intentionally skipped to eliminate the
    // bs * n_vocab * 4 byte PCIe transfer per block (replaced by a
    // K * bs * 4 byte int32 read-back). llama_context::decode() copies the
    // int32s into the dflash.draft_topk host buffer; the speculative-
    // decoding driver reads them back via llama_get_dflash_draft_topk().
    //
    // K>=2 ordering note: `ggml_top_k` returns the K largest indices in
    // **unspecified order** (the CPU kernel deliberately swaps positions
    // 0 and 1 to discourage callers from relying on order — see
    // `ggml/src/ggml-cpu/ops.cpp::ggml_compute_forward_top_k_f32`). Our
    // consumer (common/speculative.cpp) expects index [i*K+0] to be the
    // argmax. To preserve that invariant without paying for a full
    // O(V·log V) argsort, llm_build_dflash also emits `t_dflash_topk_argmax`
    // (= ggml_top_k(logits, 1)) for the K>=2 path; llama_context::decode()
    // performs an O(n_outputs * K) post-pass to swap whichever top-K slot
    // matches the argmax into position 0. K is small (typically 2-4) so
    // this pass is negligible.
    //
    // History: pre-commit-33 the K>=2 path used `ggml_argsort_top_k`
    // (full O(V·log V) sort, 9% of total time on CPU at K=2). Pre-
    // commit-34 the K=1 path AND the K>=2 companion used `ggml_argmax`
    // (single-threaded CPU kernel, ~17 ms wasted per 32-token gen across
    // 7 idle cores). Both now use `ggml_top_k`, which parallelizes
    // across rows.
    ggml_tensor * t_dflash_topk        = nullptr;
    ggml_tensor * t_dflash_topk_argmax = nullptr;

    // DFlash target hidden-state captures, one per entry in
    // dflash->capture_layer_ids (parallel order). Each tensor has shape
    // [n_embd_target, n_outputs] and is set with ggml_set_output() so the
    // backend keeps it allocated; llama_context::decode() copies the contents
    // into dflash.captured_features after compute.
    std::vector<ggml_tensor *> t_dflash_captures;

    std::vector<llm_graph_input_ptr> inputs;

    ggml_context_ptr ctx_compute;

    // memory buffers used to evaluate the model
    std::vector<uint8_t> buf_compute_meta;

    ggml_cgraph * gf;

    int64_t max_nodes;

private:
    // keep a copy of the previous graph parameters
    // we will use this to determine whether the graph can be reused by comparing them with the new parameters
    // note: these are updated after constructing the new graph
    llm_graph_params params;

    // env: LLAMA_GRAPH_RESULT_DEBUG
    int debug = 0;
};

using llm_graph_result_ptr = std::unique_ptr<llm_graph_result>;

//
// llm_graph_context
//

// used in build_rs to properly order writes and avoid unnecessary copies
using llm_graph_get_rows_fn = std::function<ggml_tensor * (ggml_context *, ggml_tensor * states, ggml_tensor * ids)>;

struct llm_graph_context {
    const llm_arch arch;

    const llama_hparams & hparams;
    const llama_cparams & cparams;
    const llama_ubatch  & ubatch;

    const int64_t n_embd;
    const int64_t n_layer;
    const int64_t n_rot;
    const int64_t n_ctx;       // user-specified context size (can be different from n_ctx_train)
    const int64_t n_head;
    const int64_t n_head_kv;
    const int64_t n_embd_head_k;
    const int64_t n_embd_k_gqa;
    const int64_t n_embd_head_v;
    const int64_t n_embd_v_gqa;
    const int64_t n_expert;
    const int64_t n_expert_used;

    const float freq_base;
    const float freq_scale;
    const float ext_factor;
    const float attn_factor;
    const float beta_fast;
    const float beta_slow;
    const float norm_eps;
    const float norm_rms_eps;

    const int64_t n_tokens;
    const int64_t n_outputs;
    const int32_t n_ctx_orig; // yarn

    const enum llama_pooling_type pooling_type;
    const enum llama_rope_type    rope_type;

    ggml_backend_sched_t sched;

    ggml_backend_t backend_cpu; // TODO: needed by build_attn_mha, figure out a way to remove?

    const llama_adapter_cvec     * cvec;
    const llama_adapter_loras    * loras;
    const llama_memory_context_i * mctx;
    const llama_cross            * cross;
    const llama_dflash           * dflash;    // DFlash drafter context
    const llama_tree_mask        * tree_mask; // DDTree custom attention mask (nullptr = no override)

    const llm_graph_cb & cb_func;

    llm_graph_result * res;

    ggml_context * ctx0 = nullptr;
    ggml_cgraph  * gf   = nullptr;

    llm_graph_context(const llm_graph_params & params);
    virtual ~llm_graph_context() = default;

    void cb(ggml_tensor * cur, const char * name, int il) const;

    //
    // common
    //

    ggml_tensor * build_cvec(
             ggml_tensor * cur,
                     int   il) const;

    // do mat_mul, while optionally apply lora
    ggml_tensor * build_lora_mm(
              ggml_tensor * w,
              ggml_tensor * cur) const;

    // do mat_mul_id, while optionally apply lora
    ggml_tensor * build_lora_mm_id(
              ggml_tensor * w,   // ggml_tensor * as
              ggml_tensor * cur, // ggml_tensor * b
              ggml_tensor * ids) const;

    ggml_tensor * build_norm(
             ggml_tensor * cur,
             ggml_tensor * mw,
             ggml_tensor * mb,
           llm_norm_type   type,
                     int   il) const;

    ggml_tensor * build_ffn(
             ggml_tensor * cur,
             ggml_tensor * up,
             ggml_tensor * up_b,
             ggml_tensor * up_s,
             ggml_tensor * gate,
             ggml_tensor * gate_b,
             ggml_tensor * gate_s,
             ggml_tensor * down,
             ggml_tensor * down_b,
             ggml_tensor * down_s,
             ggml_tensor * act_scales,
         llm_ffn_op_type   type_op,
       llm_ffn_gate_type   type_gate,
                     int   il) const;

    // build MoE FFN without bias tensors
    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * up_exps,
             ggml_tensor * gate_exps,
             ggml_tensor * down_exps,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                    bool   scale_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr) const;

    ggml_tensor * build_moe_ffn(
             ggml_tensor * cur,
             ggml_tensor * gate_inp,
             ggml_tensor * gate_inp_b,
             ggml_tensor * up_exps,
             ggml_tensor * up_exps_b,
             ggml_tensor * gate_exps,
             ggml_tensor * gate_exps_b,
             ggml_tensor * down_exps,
             ggml_tensor * down_exps_b,
             ggml_tensor * exp_probs_b,
                 int64_t   n_expert,
                 int64_t   n_expert_used,
         llm_ffn_op_type   type_op,
                    bool   norm_w,
                    bool   scale_w,
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr) const;

    //
    // inputs
    //

    ggml_tensor * build_inp_embd(ggml_tensor * tok_embd) const;
    ggml_tensor * build_inp_pos() const;
    ggml_tensor * build_inp_attn_scale() const;
    ggml_tensor * build_inp_out_ids() const;
    ggml_tensor * build_inp_mean() const;
    ggml_tensor * build_inp_cls() const;

    ggml_tensor * build_inp_cross_embd() const;
    ggml_tensor * build_inp_pos_bucket_enc() const;
    ggml_tensor * build_inp_pos_bucket_dec() const;
    ggml_tensor * build_pos_bias(ggml_tensor * pos_bucket, ggml_tensor * attn_rel_b) const;

    // DFlash drafter cross-context input.
    // Returns the kq_mask input for the drafter's bidirectional
    // cross-attention. K_ctx / V_ctx themselves are zero-copy views of
    // the per-layer side store (`dflash->ctx_K[il]` / `dflash->ctx_V[il]`)
    // and are built directly in llm_build_dflash; this function only
    // stages the kq_mask and registers it with the graph result.
    llm_graph_input_dflash * build_inp_dflash() const;

    // DFlash target hidden-state capture.
    // If `il` is in dflash->capture_layer_ids, register `cur` (after the
    // residual + post-FFN) as an output tensor and push it into
    // res->t_dflash_captures so llama_context::decode() can read it back.
    // No-op if dflash is null or capture is disabled.
    void build_dflash_capture(ggml_tensor * cur, int il) const;

    //
    // attention
    //

    ggml_tensor * build_attn_mha(
            ggml_tensor * q,       // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k,       // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v,       // [n_embd_head_v, n_head_v, n_tokens] (v_trans == false)
            ggml_tensor * kq_b,
            ggml_tensor * kq_mask,
            ggml_tensor * sinks,   // [n_head_q]
            ggml_tensor * v_mla,   // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_no_cache * build_attn_inp_no_cache() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_no_cache * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_kv * build_attn_inp_kv() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_kv * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_kv_iswa * build_attn_inp_kv_iswa() const;

    // note: if k_cur or v_cur are not provided, they will not be stored in the memory
    ggml_tensor * build_attn(
            llm_graph_input_attn_kv_iswa * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens] optional
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens] optional
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_cross * build_attn_inp_cross() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_cross * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v]
                  float   kq_scale,
                    int   il) const;

    //
    // recurrent
    //

    // TODO: move this implementation to llama_memory_recurrent.
    //       this is analogous to llama_kv_cache::cpy_k / cpy_v
    //       when moving, avoid passing `ggml_cgraph` - only pass `ggml_context`. would likely need to split the
    //         implementation in 2 separate methods. the goal is to avoid calling `ggml_build_forward_expand` in
    //         `llama_memory_recurrent`
    ggml_tensor * build_rs(
            ggml_tensor * s,
            ggml_tensor * state_copy_main,
            ggml_tensor * state_copy_extra,
                int32_t   state_size,
                int32_t   n_seqs,
               uint32_t   n_rs,
               uint32_t   rs_head,
               uint32_t   rs_size,
                int32_t   rs_zero,
            const llm_graph_get_rows_fn & get_state_rows = ggml_get_rows) const;

    llm_graph_input_rs * build_rs_inp() const;

    ggml_tensor * build_rs(
            llm_graph_input_rs * inp,
            ggml_tensor * s,
                int32_t   state_size,
                int32_t   n_seqs,
            const llm_graph_get_rows_fn & get_state_rows = ggml_get_rows) const;

    ggml_tensor * build_rwkv_token_shift_load(
        llm_graph_input_rs * inp,
        const llama_ubatch & ubatch,
                       int   il) const;

    ggml_tensor * build_rwkv_token_shift_store(
             ggml_tensor * token_shift,
      const llama_ubatch & ubatch,
                     int   il) const;
    //
    // hybrid
    //

    llm_graph_input_mem_hybrid * build_inp_mem_hybrid() const;

    //
    // pooling
    //

    void build_pooling(
            ggml_tensor * cls,
            ggml_tensor * cls_b,
            ggml_tensor * cls_out,
            ggml_tensor * cls_out_b) const;

    //
    // dense (out)
    //

    void build_dense_out(
            ggml_tensor * dense_2,
            ggml_tensor * dense_3) const;
};

// TODO: better name
int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional);
