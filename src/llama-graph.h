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
#include <map>

struct ggml_cgraph;
struct ggml_context;
struct ggml_tensor;

struct llama_cparams;
struct llama_layer;

struct llama_memory_context_i;

class llama_kv_cache_context;
class llama_kv_cache_iswa_context;
class llama_memory_recurrent_context;
class llama_memory_hybrid_context;
class llama_memory_hybrid_iswa_context;

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
//   (a) On the *draft* context — speculative driver pushes per-block target
//       hidden states into the per-layer K/V side store via
//       llama_dflash_extend(). The drafter graph reads `n_ctx` and the side
//       store tensors `ctx_K[il]` / `ctx_V[il]` to build cross-attention.
//   (b) On the *target* context — driver writes capture_layer_ids /
//       capture_n_embd via llama_dflash_set_capture() before any decode.
//       The target graph tees out hidden states at those layer indices.
struct llama_dflash {
    // ---------- (a) drafter sizing ----------
    int64_t n_features = 0; // per-token feature dim (n_target_layer_ids * n_embd_target)
    int64_t n_ctx      = 0; // K_ctx column count seen by drafter's attention

    // ---------- (b) target capture ----------
    std::vector<int32_t> capture_layer_ids;
    int64_t              capture_n_embd = 0;
    std::vector<float>   captured_features;
    int64_t              captured_n_outputs = 0;

    // Per-layer host staging buffers for batched D2H capture transfer.
    // The fast path leaves this empty (single packed D2H goes straight
    // into captured_features); only populated when the fallback path is
    // taken (graph result without t_dflash_captures_packed).
    std::vector<std::vector<float>> capture_staging;

    // when true, decode() skips the D2H of captured_features and the consumer
    // is expected to use llama_dflash_extend_from_ctx() to feed captures into
    // the draft via a device-to-device path. Set via
    // llama_dflash_set_skip_host_readback() on the TARGET context once,
    // typically at consumer init time.
    bool                 skip_host_readback = false;

    // cached pointer to the most recently finalized packed captures tensor
    // (from the last llama_decode() on this context). Valid until the next
    // decode on the same context. Consumers use this
    // for device-to-device extends.
    ggml_tensor *        last_packed_captures = nullptr;

    // ---------- (a') drafter inline top-K read-back ----------
    std::vector<int32_t> draft_topk;
    int64_t              draft_topk_n_outputs = 0;
    uint32_t             draft_topk_K         = 0;

    // Companion to draft_topk for the K>=2 path.
    std::vector<int32_t> draft_topk_argmax;

    // ---------- (c) per-layer K/V side store (paper §4.1 reuse) ----------
    std::vector<ggml_tensor *> ctx_K;
    std::vector<ggml_tensor *> ctx_V;
    int64_t ctx_filled   = 0;
    int64_t ctx_capacity = 0;
    int64_t ctx_pos_base = 0;

    // ---------- (d) inline encoder destinations + per-decode state ----
    //
    // When the TARGET context emits the encoder ops inline into its own
    // decode graph (cparams.dflash_inline_encoder), set_rows writes K_new /
    // V_new straight into the DRAFT context's side store. These pointers
    // are bound from the draft via llama_dflash_bind_inline_side_store()
    // and remain valid for the lifetime of the draft context.
    //
    // inline_write_offset and inline_pos_start are set per-decode by the
    // speculative driver before llama_decode(ctx_tgt, batch) so the inline
    // encoder input class can fill the pos_new / pos_idx tensors at
    // set_input() time.
    //
    // Empty / zero on contexts that don't participate in the inline path.
    std::vector<ggml_tensor *> inline_dst_K;
    std::vector<ggml_tensor *> inline_dst_V;
    int64_t                    inline_write_offset = 0;
    int64_t                    inline_pos_start    = 0;

    // ---------- (e) GDN history (target-side persistent buffers) ----
    //
    // When the TARGET context has cparams.dflash_gdn_history set and the
    // model has GatedDeltaNet (recurrent) layers, one persistent
    // per-token-state-history tensor is allocated per recurrent layer.
    // Shape per tensor: [S_v, S_v, H_v, gdn_history_max_tokens * n_seqs_max].
    // The qwen3.5 graph builder emits a ggml_cpy at the end of the GDN
    // layer that copies the packed state_history region of the
    // GATED_DELTA_NET_WITH_HISTORY op output into gdn_history[il]. After
    // the verify decode + sampler accept, the speculative driver builds
    // a small fixup graph that runs GATED_DELTA_NET_STATE_SELECT on each
    // gdn_history[il] and writes the chosen slab back into the
    // recurrent-state cache.
    //
    // Non-recurrent layers have gdn_history[il] == nullptr.
    // Empty on draft / non-GDN-target contexts.
    std::vector<ggml_tensor *> gdn_history;
    int64_t                    gdn_history_max_tokens = 0;  // shared cap across layers
    // per-seq dimension of the gdn_history persistent buffers (last dim
    // of each tensor view'd as 5-D [S_v, S_v, H, max_tokens, n_seqs_max]).
    // 1 in chain mode; bumped
    // to cparams.n_seq_max in tree mode. The kernel routes per-seq
    // intermediate-state spills using this stride.
    int64_t                    gdn_history_n_seqs_max = 1;

    // Companion persistent buffers for the conv (r_l) side of each GDN
    // layer. Holds the FULL conv_input tensor from each chain-verify
    // decode (shape [conv_kernel_size - 1 + n_seq_tokens, conv_channels,
    // n_seqs]). On partial acceptance, the next decode's conv-state
    // fixup reads rows [k_index+1 : k_index + conv_kernel_size] of this
    // buffer to recover the correct conv state ending at the last
    // accepted position.
    //
    // Allocated and shaped at sched_reserve() time with the same cap
    // as gdn_history (gdn_history_max_tokens). One tensor per GDN
    // layer; nullptr for full-attention layers.
    std::vector<ggml_tensor *> conv_history;

    // Per-decode fixup index, set by the speculative driver between
    // chain-verify decodes. >= 0 = select state_history[k_index] for
    // each GDN layer at the start of the next verify graph; < 0 = no
    // fixup (state_select falls back to current ssm slot — first decode
    // after prefill or full-acceptance iter). Default -1.
    //
    // Drives both the GDN state_select op AND the conv-state fixup
    // (they share the same k_index because they're rolling back to the
    // same chain position).
    int32_t                    gdn_history_k_index = -1;

    // tree-mode per-seq k_index. Populated by
    // llama_dflash_set_gdn_history_k_index_per_seq before each
    // tree-verify decode. Empty in chain mode (scalar
    // gdn_history_k_index applies). When non-empty, the gdn_fixup
    // input class writes this whole vector into the k_index tensor
    // (sized [n_seqs]) consumed by state_select; the kernel picks
    // per-seq state_history[k_indices[s], s].
    std::vector<int32_t>       gdn_history_k_indices;

    // tree-mode parent_ids buffer for the next decode.
    // Shape [n_tokens, n_seqs] in row-major order — entry
    // [t + s * n_tokens] is t's parent in seq s, or
    // GGML_GDN_TREE_ROOT_PARENT (= -1) if t's parent is the pre-block
    // state. Populated by llama_dflash_set_gdn_history_parent_ids;
    // consumed by the per-decode parent_ids input class that mirrors
    // it to a device-side I32 tensor at set_input time.
    //
    // Empty in chain mode. The qwen35-family graph builder only emits
    // parent_ids inputs when cparams.n_seq_max > 1 &&
    // cparams.dflash_gdn_history.
    std::vector<int32_t>       gdn_history_parent_ids;
    int32_t                    gdn_history_parent_ids_n_tokens = 0;
    int32_t                    gdn_history_parent_ids_n_seqs   = 0;
};

// DFlash custom attention mask for tree-shaped verification.
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
    llm_graph_input_embd(int64_t n_embd) : n_embd(n_embd) {}
    virtual ~llm_graph_input_embd() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * tokens = nullptr; // I32 [n_batch]
    ggml_tensor * embd   = nullptr; // F32 [n_embd, n_batch]

    const int64_t n_embd = 0;
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
    llm_graph_input_attn_temp(uint32_t n_attn_temp_floor_scale, float f_attn_temp_scale, float f_attn_temp_offset)
        : n_attn_temp_floor_scale(n_attn_temp_floor_scale), f_attn_temp_scale(f_attn_temp_scale), f_attn_temp_offset(f_attn_temp_offset) {}
    virtual ~llm_graph_input_attn_temp() = default;

    void set_input(const llama_ubatch * ubatch) override;

    ggml_tensor * attn_scale = nullptr; // F32 [n_batch]

    const uint32_t n_attn_temp_floor_scale;
    const float    f_attn_temp_scale;
    const float    f_attn_temp_offset;
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

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * s_copy;  // I32 [n_rs]

    // views of s_copy, computed once per graph
    // and shared across layers which use build_rs
    ggml_tensor * s_copy_main;   // I32 [n_seqs]
    ggml_tensor * s_copy_extra;  // I32 [n_rs - n_seqs]

    const llama_memory_recurrent_context * mctx;

    // used in view offsets, need to match for valid graph reuse
    uint32_t head;
    int32_t rs_z;
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
// the [side-store K/V | proposal] segments. When the draft model has
// SWA layers (Gemma-family targets), a second mask kq_mask_swa is also
// built; the draft graph selects per-layer between the two masks.
class llm_graph_input_dflash : public llm_graph_input_i {
public:
    llm_graph_input_dflash(const llama_dflash * dflash, int64_t n_ctx, int64_t n_block,
                           uint32_t n_swa = 0, llama_swa_type swa_type = LLAMA_SWA_TYPE_NONE)
        : dflash(dflash), n_ctx(n_ctx), n_block(n_block),
          n_swa(n_swa), swa_type(swa_type) {}
    virtual ~llm_graph_input_dflash() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * kq_mask         = nullptr; // F32 [n_ctx + n_block, n_block_pad, 1, 1]
    ggml_tensor * kq_mask_cnv     = nullptr; // f16 cast for flash_attn

    // Optional SWA-restricted variant; nullptr on models with no SWA
    // layers. Same layout as kq_mask, but additionally masks ctx slots
    // whose absolute position is further than n_swa behind the query
    // (using llama_hparams::is_masked_swa for the cutoff rule).
    ggml_tensor * kq_mask_swa     = nullptr;
    ggml_tensor * kq_mask_swa_cnv = nullptr;

    const llama_dflash * dflash;
    int64_t              n_ctx;
    int64_t              n_block;

    uint32_t             n_swa;
    llama_swa_type       swa_type;
};

// inline-encoder input class. Holds the pos_new (RoPE positions) and pos_idx
// (write slots into the side store) tensors created in the target's main
// decode graph. set_input() fills both from a live pointer
// to the target context's llama_dflash struct (inline_pos_start /
// inline_write_offset are bumped by the speculative driver before each
// llama_decode call).
//
// `n_outputs` is the number of capture positions in this graph build —
// equals n_outputs of the target decode and matches the row count of the
// packed captures tensor that feeds the encoder.
class llm_graph_input_dflash_inline_encode : public llm_graph_input_i {
public:
    llm_graph_input_dflash_inline_encode(llama_dflash * dflash, int64_t n_outputs)
        : dflash(dflash), n_outputs(n_outputs) {}
    virtual ~llm_graph_input_dflash_inline_encode() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * pos_new = nullptr; // I32 [n_outputs]
    ggml_tensor * pos_idx = nullptr; // I64 [n_outputs]

    llama_dflash * dflash;     // non-owning, lives on the target context
    int64_t        n_outputs;
};

// DFlash GDN history fixup input. Holds an I32 `k_index` tensor that the
// in-graph state_select op consumes. set_input() reads the host-set value(s)
// off the llama_dflash struct (set between decodes
// by the speculative driver).
//
// `k_index` is sized [1] for chain mode (n_seqs == 1) and [n_seqs] for
// tree mode (n_seqs > 1). The constructor argument `k_index_count` records
// which one was allocated so set_input() picks the right host source
// (scalar gdn_history_k_index vs per-seq
// gdn_history_k_indices).
class llm_graph_input_dflash_gdn_fixup : public llm_graph_input_i {
public:
    llm_graph_input_dflash_gdn_fixup(llama_dflash * dflash, int32_t k_index_count)
        : dflash(dflash), k_index_count(k_index_count) {}
    virtual ~llm_graph_input_dflash_gdn_fixup() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * k_index = nullptr; // I32 [k_index_count]

    llama_dflash * dflash; // non-owning
    int32_t        k_index_count = 1;
};

// tree-mode parent_ids input class. Mirrors the host-side
// dflash->gdn_history_parent_ids vector to a device-side I32 tensor at
// set_input time. The qwen35-family graph builder emits one of these per
// recurrent layer; all of them set_input from the same vector.
class llm_graph_input_dflash_gdn_parent_ids : public llm_graph_input_i {
public:
    explicit llm_graph_input_dflash_gdn_parent_ids(llama_dflash * dflash)
        : dflash(dflash) {}
    virtual ~llm_graph_input_dflash_gdn_parent_ids() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * parent_ids = nullptr; // I32 [n_tokens, n_seqs]

    llama_dflash * dflash; // non-owning
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

    // note: assumes v_rot^2 == I
    ggml_tensor * self_k_rot = nullptr;
    ggml_tensor * self_v_rot = nullptr;

    // note: these have to be copies because in order to be able to reuse a graph, its inputs
    //       need to carry these parameters with them. otherwise, they can point to freed
    //       llm_graph_params from a previous batch, causing stack-use-after-return
    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;

    // when non-null and `tree_mask->active`, set_input overwrites the
    // [n_tree_tokens × n_tree_tokens] block of the freshly written attention
    // mask with `visibility[i*n+j] ? 0.0f : -INFINITY`. Pointer borrowed from
    // llama_context::tree_mask. nullptr (the default) is a no-op.
    const llama_tree_mask * tree_mask = nullptr;
};

// V-less input for the KV cache
// ref: https://github.com/ggml-org/llama.cpp/pull/19067
class llm_graph_input_attn_k : public llm_graph_input_i {
public:
    llm_graph_input_attn_k(
            const llama_hparams & hparams,
            const llama_cparams & cparams,
            const llama_kv_cache_context * mctx) :
        hparams(hparams),
        cparams(cparams),
        mctx(mctx) {
    }
    ~llm_graph_input_attn_k() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    ggml_tensor * get_k_idxs() const { return self_k_idxs; }

    ggml_tensor * get_kq_mask() const { return self_kq_mask_cnv; }

    ggml_tensor * self_k_idxs = nullptr; // I64 [n_batch]

    ggml_tensor * self_kq_mask     = nullptr; // F32 [n_kv, n_batch/n_stream, 1, n_stream]
    ggml_tensor * self_kq_mask_cnv = nullptr; //     [n_kv, n_batch/n_stream, 1, n_stream]

    const llama_hparams hparams;
    const llama_cparams cparams;

    const llama_kv_cache_context * mctx;
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

    ggml_tensor * self_k_rot = nullptr;
    ggml_tensor * self_v_rot = nullptr;

    ggml_tensor * self_k_rot_swa = nullptr;
    ggml_tensor * self_v_rot_swa = nullptr;

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
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv> inp_attn,
            std::unique_ptr<llm_graph_input_rs>      inp_rs,
            const llama_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_kv * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_context * mctx;
};

class llm_graph_input_mem_hybrid_k : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_k(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_k> inp_attn,
            std::unique_ptr<llm_graph_input_rs>      inp_rs,
            const llama_memory_hybrid_context *      mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_k() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_k> inp_attn;
    std::unique_ptr<llm_graph_input_rs>      inp_rs;

    llm_graph_input_attn_k * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs      * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_context * mctx;
};

class llm_graph_input_mem_hybrid_iswa : public llm_graph_input_i {
public:
    llm_graph_input_mem_hybrid_iswa(
            const llama_cparams & cparams,
            std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn,
            std::unique_ptr<llm_graph_input_rs>          inp_rs,
            const llama_memory_hybrid_iswa_context *     mctx) :
        inp_attn(std::move(inp_attn)),
        inp_rs(std::move(inp_rs)),
        cparams(cparams),
        mctx(mctx) { }
    virtual ~llm_graph_input_mem_hybrid_iswa() = default;

    void set_input(const llama_ubatch * ubatch) override;

    bool can_reuse(const llm_graph_params & params) override;

    std::unique_ptr<llm_graph_input_attn_kv_iswa> inp_attn;
    std::unique_ptr<llm_graph_input_rs>          inp_rs;

    llm_graph_input_attn_kv_iswa * get_attn() const { return inp_attn.get(); }
    llm_graph_input_rs           * get_recr() const { return inp_rs.get(); }

    const llama_cparams cparams;

    const llama_memory_hybrid_iswa_context * mctx;
};

class llm_graph_input_sampling : public llm_graph_input_i {
public:
    llm_graph_input_sampling(std::map<llama_seq_id, llama_sampler *> samplers) :
        samplers(std::move(samplers)) { }
    virtual ~llm_graph_input_sampling() = default;

    void set_input(const llama_ubatch * ubatch) override;
    bool can_reuse(const llm_graph_params & params) override;

    std::map<llama_seq_id, llama_sampler *> samplers;
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
    const llama_dflash           * dflash    = nullptr; // DFlash drafter state (capture, K/V side store)
    const llama_tree_mask        * tree_mask = nullptr; // DFlash tree-mode custom attention mask (nullptr = no override)

    std::map<llama_seq_id, llama_sampler *> samplers;

    static bool samplers_equal(
          const std::map<llama_seq_id, llama_sampler *> & lhs,
          const std::map<llama_seq_id, llama_sampler *> & rhs) {
        if (lhs.size() != rhs.size()) {
            return false;
        }
        for (const auto & [seq_id, sampler] : lhs) {
            auto it = rhs.find(seq_id);
            if (it == rhs.end() || it->second != sampler) {
                return false;
            }
        }
        return true;
    }

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

        if (n_outputs != other.n_outputs) {
            return false;
        }

        if (!samplers_equal(samplers, other.samplers)) {
            return false;
        }

        if (samplers.size() > 0) {
            if (!ubatch.data || !other.ubatch.data) {
                return false;
            }

            // check that the outputs are the same for all samplers
            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                if (ubatch.output[i]    != other.ubatch.output[i] ||
                    ubatch.seq_id[i][0] != other.ubatch.seq_id[i][0]) {
                    return false;
                }
            }
        }

        return
            cparams.embeddings  == other.cparams.embeddings  &&
            cparams.causal_attn == other.cparams.causal_attn &&
            arch  == other.arch  &&
            gtype == other.gtype &&
            cvec  == other.cvec  &&
            loras == other.loras &&
            cross == other.cross &&
            dflash    == other.dflash &&
            tree_mask == other.tree_mask;
    }
};

class llm_graph_result {
public:
    llm_graph_result(int64_t max_nodes);

    virtual ~llm_graph_result() = default;

    ggml_tensor * get_inp_tokens()  const { return t_inp_tokens; }
    ggml_tensor * get_logits()      const { return t_logits; }
    ggml_tensor * get_embd()        const { return t_embd; }
    ggml_tensor * get_embd_pooled() const { return t_embd_pooled; }

    ggml_cgraph  * get_gf()  const { return gf; }
    ggml_context * get_ctx() const { return ctx_compute.get(); }

    int64_t get_max_nodes() const;

    void reset();

    void set_inputs(const llama_ubatch * ubatch);
    void set_outputs();

    // try to update the existing graph result using the new graph parameters in order to reuse it
    // this can only be done if we determine that the resulting graph using the new graph parameters
    //   would be identical to the existing graph. in that case, we simply have to update the memory
    //   contexts of the input tensors of the graph and we can reuse it for another computation
    // return true if the graph was updated and can be reused
    bool can_reuse(const llm_graph_params & params);

    llm_graph_input_i * add_input(llm_graph_input_ptr input);

    void set_params(const llm_graph_params & params);

    // important graph nodes
    ggml_tensor * t_inp_tokens  = nullptr;
    ggml_tensor * t_inp_embd    = nullptr; // [n_embd_inp, n_tokens]
    ggml_tensor * t_logits      = nullptr;
    ggml_tensor * t_embd        = nullptr;
    ggml_tensor * t_embd_pooled = nullptr;

    // DFlash drafter: in-graph top-K candidate token IDs of the per-position
    // lm_head logits. When non-null, llm_build_dflash sets t_logits = nullptr
    // — the float-logits read-back is skipped to eliminate the
    // bs * n_vocab * 4 byte PCIe transfer per block (replaced by a
    // K * bs * 4 byte int32 read-back).
    ggml_tensor * t_dflash_topk        = nullptr;
    ggml_tensor * t_dflash_topk_argmax = nullptr;

    // DFlash target hidden-state captures, one per entry in
    // dflash->capture_layer_ids (parallel order). Each tensor has shape
    // [n_embd_target, n_outputs].
    std::vector<ggml_tensor *> t_dflash_captures;

    // DFlash target captures packed on-graph: same data as
    // t_dflash_captures but as a single tensor of shape
    // [n_layers * n_embd_target, n_outputs] in the same row-per-token,
    // all-layers-side-by-side layout the encoder graph in dflash_extend()
    // consumes. Built once the last expected capture lands (see
    // build_dflash_capture in llama-graph.cpp). Optional: nullptr if the
    // model didn't request DFlash capture, or if only a subset of expected
    // captures arrived. Consumers should fall back to t_dflash_captures
    // when this is null.
    ggml_tensor * t_dflash_captures_packed = nullptr;

    // graph-input dedup: cached shared inputs for the per-layer GDN history
    // fixup (k_index) and parent_ids tensors. Without dedup, the qwen35-family
    // builder emits one fresh INPUT-flagged
    // tensor per recurrent layer per fixup type — 48 GDN layers × 2
    // fixups = 96 input tensors per build, which trips the historic
    // GGML_SCHED_MAX_SPLIT_INPUTS = 30 cap on multi-GPU pipeline-
    // parallel paths (Vulkan default, CUDA when ngpus > 1). All those
    // tensors carry the same scalar/vector value within a single build,
    // so a single shared input is correct. Raw pointers; the inputs
    // themselves are owned by `inputs` and freed via that vector.
    // Cleared in reset() per build.
    llm_graph_input_dflash_gdn_fixup     * t_dflash_gdn_fixup_shared      = nullptr;
    llm_graph_input_dflash_gdn_parent_ids * t_dflash_gdn_parent_ids_shared = nullptr;

    std::map<llama_seq_id, ggml_tensor*> t_sampled_logits;
    std::map<llama_seq_id, ggml_tensor*> t_candidates;
    std::map<llama_seq_id, ggml_tensor*> t_sampled;
    std::map<llama_seq_id, ggml_tensor*> t_sampled_probs;

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

struct llm_graph_qkv {
    ggml_tensor * q; // [n_embd_head, n_head,    n_tokens]
    ggml_tensor * k; // [n_embd_head, n_head_kv, n_tokens]
    ggml_tensor * v; // [n_embd_head, n_head_kv, n_tokens]
};

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
    const llama_dflash           * dflash    = nullptr; // DFlash drafter context
    const llama_tree_mask        * tree_mask = nullptr; // DFlash tree-mode custom attention mask (nullptr = no override)

    std::map<llama_seq_id, llama_sampler *> samplers;

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

    // do mat_mul, while optionally apply lora and per-tensor scale
    ggml_tensor * build_lora_mm(
              ggml_tensor * w,
              ggml_tensor * cur,
              ggml_tensor * w_s = nullptr) const;

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


    // compute Q, K, V projections with optional bias and reshape
    // supports both fused wqkv and separate wq/wk/wv paths
    llm_graph_qkv build_qkv(
        const llama_layer & layer,
              ggml_tensor * cur,
                  int64_t   n_embd_head,
                  int64_t   n_head,
                  int64_t   n_head_kv,
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
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr) const;

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
                   float   w_scale,
            llama_expert_gating_func_type gating_op,
                     int   il,
             ggml_tensor * probs_in = nullptr,
             ggml_tensor * gate_up_exps = nullptr,
             ggml_tensor * gate_up_exps_b = nullptr,
             ggml_tensor * up_exps_s = nullptr,
             ggml_tensor * gate_exps_s = nullptr,
             ggml_tensor * down_exps_s = nullptr) const;

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
    // the per-layer side store and are built directly in the dflash graph.
    llm_graph_input_dflash * build_inp_dflash() const;

    // DFlash target hidden-state capture.
    // If `il` is in dflash->capture_layer_ids, register `cur` as an output
    // tensor and push it into res->t_dflash_captures so llama_context::decode()
    // can read it back. No-op if dflash is null or capture is disabled.
    void build_dflash_capture(ggml_tensor * cur, int il) const;

    // inline encoder: called from the per-arch graph builder AFTER the main
    // layer loop with the packed captures tensor (built
    // earlier by build_dflash_capture at the final captured layer).
    // Emits the encoder graph:
    //   dflash_fc -> RMSNorm(dflash_hidden_norm)
    //   -> for each draft layer:
    //        wk * h_proj -> reshape -> attn_k_norm -> RoPE -> set_rows(ctx_K)
    //        wv * h_proj -> reshape                       -> set_rows(ctx_V)
    // directly into the target's decode graph. set_rows destinations come
    // from dflash->inline_dst_{K,V} (bound from the draft context via
    // llama_dflash_bind_inline_side_store). The encoder weights come from
    // the model's target_dflash_* pointers (bound from the draft model
    // via llama_dflash_bind_encoder). Uses *target's* hparams for the
    // per-head dims and RoPE params — correct only when target & draft
    // share them (Qwen3 family does; Qwen3.5 does not and needs explicit
    // sizing through a later patch). No-op when inline mode is off or
    // any required state is missing.
    void build_dflash_inline_encoder(const llama_model & model,
                                     ggml_tensor *       packed_captures) const;

    // DFlash GDN history fixup. Returns a `selected_state` tensor of shape
    // [S_v, S_v, H_v, n_seqs] that the caller (qwen35.cpp) uses directly
    // as the GDN op's `state` input, bypassing the
    // ssm_states_all read entirely on this layer in this decode.
    //
    // Semantics (encoded in the state_select kernel):
    //   k_index >= 0  : selected = gdn_history[il][..., k_index, :].
    //   k_index <  0  : selected = current ssm_states_all_slot value
    //                   (fallback path; net effect == legacy read).
    //
    // Returns nullptr (no ops emitted) when cparams.dflash_gdn_history is
    // false, dflash is null, or the gdn_history buffer for this layer
    // wasn't allocated — caller falls back to the legacy build_rs path.
    //
    // In tree mode (cparams.n_seq_max > 1 && cparams.dflash_gdn_history)
    // the k_index tensor is sized [n_seqs] and set_input pulls from
    // dflash->gdn_history_k_indices; in chain mode it's [1] and pulls
    // from the scalar gdn_history_k_index. `n_seqs` here matches the
    // ubatch's n_seqs at graph build time.
    ggml_tensor * build_dflash_gdn_history_fixup_or_null(
            int           il,
            ggml_tensor * ssm_states_all_slot_view,
            int64_t       n_seqs) const;

    // build the parent_ids graph input for one GDN layer in tree mode.
    // Returns the input's I32 tensor of shape [n_tokens, n_seqs], or nullptr
    // when tree mode is not active (chain mode or no
    // dflash on the context). The set_input call on every emitted
    // instance pulls from the same dflash->gdn_history_parent_ids
    // vector, so the host source stays in lock-step across layers.
    // Per-layer cost: one host->device I32 mirror of size
    // n_tokens * n_seqs * 4 bytes (negligible at typical tree budgets).
    ggml_tensor * build_dflash_gdn_parent_ids_or_null(
            int64_t n_tokens,
            int64_t n_seqs) const;

    // graph-input dedup helper. Returns the shared k_index graph input tensor
    // (lazily creating + caching on `res` on first call within a build, then
    // reusing across layers / call sites).
    // Returns nullptr when dflash + GDN-history isn't active. Used by
    // both build_dflash_gdn_history_fixup_or_null (GDN state_select)
    // and the conv_state_history_select site in the qwen35-family
    // graph builders. All call sites within one build pass the same
    // k_index_count (chain: 1; tree: n_seqs); shape mismatch falls
    // back to a fresh input (defensive — shouldn't happen).
    ggml_tensor * build_dflash_gdn_fixup_k_index_or_null(
            int32_t k_index_count) const;

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
            ggml_tensor * wo_s,
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
            ggml_tensor * wo_s,
            ggml_tensor * q_cur, // [n_embd_head_q, n_head_q, n_tokens]
            ggml_tensor * k_cur, // [n_embd_head_k, n_head_k, n_tokens]
            ggml_tensor * v_cur, // [n_embd_head_v, n_head_v, n_tokens]
            ggml_tensor * kq_b,
            ggml_tensor * sinks, // [n_head_q]
            ggml_tensor * v_mla, // [n_embd_head_v_mla, n_embd_head_v, n_head_v] // TODO: remove
                  float   kq_scale,
                    int   il) const;

    llm_graph_input_attn_k  * build_attn_inp_k() const;

    ggml_tensor * build_attn(
            llm_graph_input_attn_k * inp,
            ggml_tensor * wo,
            ggml_tensor * wo_b,
            ggml_tensor * wo_s,
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
            ggml_tensor * wo_s,
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
            ggml_tensor * wo_s,
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
    llm_graph_input_mem_hybrid_k * build_inp_mem_hybrid_k() const;

    llm_graph_input_mem_hybrid_iswa * build_inp_mem_hybrid_iswa() const;

    //
    // pooling
    //

    void build_pooling(
            ggml_tensor * cls,
            ggml_tensor * cls_b,
            ggml_tensor * cls_out,
            ggml_tensor * cls_out_b,
            ggml_tensor * cls_norm) const;

    //
    // sampling (backend sampling)
    //

    void build_sampling() const;

    //
    // dense (out)
    //

    void build_dense_out(
            ggml_tensor * dense_2,
            ggml_tensor * dense_2_b,
            ggml_tensor * dense_3) const;
};

// TODO: better name
int32_t llama_relative_position_bucket(llama_pos x, llama_pos y, uint64_t n_buckets, bool bidirectional);
