#include "gated_delta_net.cuh"
#include <cuda_fp16.h>
#include <type_traits>

// Tree-mode parent-index sentinel. A parent index < 0 means "the parent is the
// pre-block state" (root of the DFS-flattened sub-tree); the warp reloads
// s_shard from curr_state instead of an intermediate-state slot.
#ifndef GGML_GDN_TREE_ROOT_PARENT
#define GGML_GDN_TREE_ROOT_PARENT (-1)
#endif

// Inter-state load/store helpers. The persistent intermediate-state buffer
// (used by tree mode for branch-point reloads) can live in either f32 or
// f16; f16 halves the per-layer footprint.
//
// In chain mode (TREE_MODE=false, InterT=float) the kernel only ever writes
// here; the helpers compile down to a plain store.
static __device__ __forceinline__ float load_inter_state(const float * p, int idx) {
    return p[idx];
}
static __device__ __forceinline__ float load_inter_state(const __half * p, int idx) {
    return __half2float(p[idx]);
}
static __device__ __forceinline__ void store_inter_state(float * p, int idx, float v) {
    p[idx] = v;
}
static __device__ __forceinline__ void store_inter_state(__half * p, int idx, float v) {
    p[idx] = __float2half(v);
}

template <int S_v, bool KDA, bool TREE_MODE, typename InterT = float>
__global__ void __launch_bounds__((ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v) * 4, 2)
gated_delta_net_cuda(const float * q,
                                     const float * k,
                                     const float * v,
                                     const float * g,
                                     const float * beta,
                                     const float * curr_state,
                                     float *       dst,
                                     // per-token intermediate-state region. nullable: when null
                                     // (chain-mode no-history path) the kernel doesn't write per-token state.
                                     // When non-null and TREE_MODE=true, also acts as the reload source at branch
                                     // points. Memory is owned by the caller (either the appended dst region for
                                     // chain-mode history, or an external persistent buffer for tree mode).
                                     InterT *      inter_states,
                                     // parent_ids[n_seqs * n_tokens] int32, read only when
                                     // TREE_MODE=true. parent_t < 0 signals "reload from curr_state".
                                     const int *   parent_ids,
                                     int64_t       H,
                                     int64_t       n_tokens,
                                     int64_t       n_seqs,
                                     int64_t       sq1,
                                     int64_t       sq2,
                                     int64_t       sq3,
                                     int64_t       sv1,
                                     int64_t       sv2,
                                     int64_t       sv3,
                                     int64_t       sb1,
                                     int64_t       sb2,
                                     int64_t       sb3,
                                     const uint3   neqk1_magic,
                                     const uint3   rq3_magic,
                                     float         scale) {
    const uint32_t h_idx    = blockIdx.x;
    const uint32_t sequence = blockIdx.y;
    // each warp owns one column, using warp-level primitives to reduce across rows
    const int      lane     = threadIdx.x;
    const int      col      = blockIdx.z * blockDim.y + threadIdx.y;

    const uint32_t iq1 = fastmodulo(h_idx, neqk1_magic);
    const uint32_t iq3 = fastdiv(sequence, rq3_magic);

    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    float *       attn_data        = dst;
    float *       state            = dst + attn_score_elems;

    const int64_t state_offset = (sequence * H + h_idx) * S_v * S_v;
    state += state_offset;
    curr_state += state_offset + col * S_v;
    attn_data += (sequence * n_tokens * H + h_idx) * S_v;

    // Per-(seq, head) base for this block's per-token intermediates at t=0.
    // Advance by S_v * S_v * H each `t` so the layout stays
    // [S_v, S_v, H, n_tokens, n_seqs] (matches the chain-mode embedded region).
    InterT * inter_base = (inter_states != nullptr)
        ? inter_states + (sequence * n_tokens * H + h_idx) * S_v * S_v
        : nullptr;

    // Tree-mode parent slice for this sequence. Read-only.
    const int * parent_ids_seq = nullptr;
    if constexpr (TREE_MODE) {
        parent_ids_seq = parent_ids + sequence * n_tokens;
    }

    constexpr int warp_size = ggml_cuda_get_physical_warp_size() < S_v ? ggml_cuda_get_physical_warp_size() : S_v;
    static_assert(S_v % warp_size == 0, "S_v must be a multiple of warp_size");
    constexpr int rows_per_lane = (S_v + warp_size - 1) / warp_size;
    float         s_shard[rows_per_lane];
    // state is stored transposed: M[col][i] = S[i][col], row col is contiguous

#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i = r * warp_size + lane;
        s_shard[r]  = curr_state[i];
    }

    for (int t = 0; t < n_tokens; t++) {
        // Tree branch-point reload: if this token's parent in the DFS-flattened
        // tree isn't the previous token in processing order, pull its state
        // back from the intermediate-state region. Same-thread read-after-write
        // on global memory - no __syncthreads() because each lane writes and
        // reads its own (col, row) slots.
        if constexpr (TREE_MODE) {
            if (t > 0) {
                const int parent_t = parent_ids_seq[t];
                if (parent_t < 0) {
                    // Root-of-branch: reset to the pre-block state.
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        s_shard[r] = curr_state[i];
                    }
                } else if (parent_t != t - 1) {
                    // Cross-branch: pull state from the intermediate region at
                    // parent_t. inter_base is per-seq, per-head; parent_t picks
                    // the slot, col/i picks the element. load_inter_state
                    // converts InterT (f32 or f16) -> float.
                    const InterT * parent_base = inter_states
                        + ((sequence * n_tokens + parent_t) * H + h_idx) * S_v * S_v;
#pragma unroll
                    for (int r = 0; r < rows_per_lane; r++) {
                        const int i = r * warp_size + lane;
                        s_shard[r] = load_inter_state(parent_base, col * S_v + i);
                    }
                }
                // parent_t == t - 1: sequential, keep s_shard in registers.
            }
        }

        const float * q_t = q + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * k_t = k + iq3 * sq3 + t * sq2 + iq1 * sq1;
        const float * v_t = v + sequence * sv3 + t * sv2 + h_idx * sv1;

        const int64_t gb_offset = sequence * sb3 + t * sb2 + h_idx * sb1;
        const float * beta_t = beta + gb_offset;
        const float * g_t    = g    + gb_offset * (KDA ? S_v : 1);

        const float beta_val = *beta_t;

        // Cache k and q in registers
        float k_reg[rows_per_lane];
        float q_reg[rows_per_lane];
#pragma unroll
        for (int r = 0; r < rows_per_lane; r++) {
            const int i = r * warp_size + lane;
            k_reg[r] = k_t[i];
            q_reg[r] = q_t[i];
        }

        if constexpr (!KDA) {
            const float g_val = expf(*g_t);

            // kv[col] = (S^T @ k)[col] = sum_i S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                kv_shard += s_shard[r] * k_reg[r];
            }
            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - g * kv[col]) * beta
            float delta_col = (v_t[col] - g_val * kv_col) * beta_val;

            // fused: S[i][col] = g * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                s_shard[r]  = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        } else {
            // kv[col] = sum_i g[i] * S[i][col] * k[i]
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                kv_shard += expf(g_t[i]) * s_shard[r] * k_reg[r];
            }

            float kv_col = warp_reduce_sum<warp_size>(kv_shard);

            // delta[col] = (v[col] - kv[col]) * beta
            float delta_col = (v_t[col] - kv_col) * beta_val;

            // fused: S[i][col] = g[i] * S[i][col] + k[i] * delta[col]
            // attn[col] = (S^T @ q)[col] = sum_i S[i][col] * q[i]
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                s_shard[r]  = expf(g_t[i]) * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }

            float attn_col = warp_reduce_sum<warp_size>(attn_partial);

            if (lane == 0) {
                attn_data[col] = attn_col * scale;
            }
        }

        // per-token intermediate-state spill (chain-mode history, or
        // tree-mode reload source). Same transposed layout as the final-state
        // write below. store_inter_state() compiles to a plain store for
        // InterT=float (chain) and __float2half for InterT=__half.
        if (inter_base != nullptr) {
#pragma unroll
            for (int r = 0; r < rows_per_lane; r++) {
                const int i = r * warp_size + lane;
                store_inter_state(inter_base, col * S_v + i, s_shard[r]);
            }
            inter_base += S_v * S_v * H;
        }

        attn_data += S_v * H;
    }

    // Write state back to global memory (transposed layout)
#pragma unroll
    for (int r = 0; r < rows_per_lane; r++) {
        const int i          = r * warp_size + lane;
        state[col * S_v + i] = s_shard[r];
    }
}

template <bool KDA, bool TREE_MODE, typename InterT = float>
static void launch_gated_delta_net(
        const float * q_d, const float * k_d, const float * v_d,
        const float * g_d, const float * b_d, const float * s_d,
        float * dst_d,
        InterT * inter_states_d,
        const int * parent_ids_d,
        int64_t S_v,   int64_t H, int64_t n_tokens, int64_t n_seqs,
        int64_t sq1,   int64_t sq2, int64_t sq3,
        int64_t sv1,   int64_t sv2, int64_t sv3,
        int64_t sb1,   int64_t sb2, int64_t sb3,
        int64_t neqk1, int64_t rq3,
        float scale, cudaStream_t stream) {
    //TODO: Add chunked kernel for even faster pre-fill
    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;
    dim3      grid_dims(H, n_seqs, (S_v + num_warps - 1) / num_warps);
    dim3      block_dims(warp_size <= S_v ? warp_size : S_v, num_warps, 1);

    const uint3 neqk1_magic = init_fastdiv_values(neqk1);
    const uint3 rq3_magic   = init_fastdiv_values(rq3);

    int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    switch (S_v) {
        case 16:
            gated_delta_net_cuda<16, KDA, TREE_MODE, InterT><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, inter_states_d, parent_ids_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 32:
            gated_delta_net_cuda<32, KDA, TREE_MODE, InterT><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, inter_states_d, parent_ids_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        case 64: {
            gated_delta_net_cuda<64, KDA, TREE_MODE, InterT><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, inter_states_d, parent_ids_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        case 128: {
            gated_delta_net_cuda<128, KDA, TREE_MODE, InterT><<<grid_dims, block_dims, 0, stream>>>(
                q_d, k_d, v_d, g_d, b_d, s_d, dst_d, inter_states_d, parent_ids_d, H,
                n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
                sb1, sb2, sb3, neqk1_magic, rq3_magic, scale);
            break;
        }
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q     = dst->src[0];
    ggml_tensor * src_k     = dst->src[1];
    ggml_tensor * src_v     = dst->src[2];
    ggml_tensor * src_g     = dst->src[3];
    ggml_tensor * src_beta  = dst->src[4];
    ggml_tensor * src_state = dst->src[5];

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    // strides in floats (beta strides used for both g and beta offset computation)
    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    cudaStream_t stream = ctx.stream();

    if (kda) {
        launch_gated_delta_net<true, false, float>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            /*inter_states_d=*/ nullptr,
            /*parent_ids_d=*/   nullptr,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    } else {
        launch_gated_delta_net<false, false, float>(q_d, k_d, v_d, g_d, b_d, s_d, dst_d,
            /*inter_states_d=*/ nullptr,
            /*parent_ids_d=*/   nullptr,
            S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,
            sb1, sb2, sb3, neqk1, rq3, scale, stream);
    }
}

// DFlash GDN-with-history entry point. The dst tensor packs a state-history
// region after attn-output and final-state for chain-mode rollback. Tree mode
// optionally adds:
//   - src[6] = parent_ids (i32, [n_tokens, n_seqs]) -> tree-mode recurrence
//   - src[7] = persist_inter (f32 or f16, contiguous external buffer for the
//     intermediate states; replaces the embedded dst region as the spill
//     target). Required to be non-null when src[6] is non-null because the
//     tree reload reads from this buffer at branch points.
//
// Chain mode (src[6] == nullptr, src[7] == nullptr): the intermediate region
// inside dst is the spill target.
void ggml_cuda_op_gated_delta_net_with_history(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_q             = dst->src[0];
    ggml_tensor * src_k             = dst->src[1];
    ggml_tensor * src_v             = dst->src[2];
    ggml_tensor * src_g             = dst->src[3];
    ggml_tensor * src_beta          = dst->src[4];
    ggml_tensor * src_state         = dst->src[5];
    ggml_tensor * src_parent        = dst->src[6]; // optional: tree mode
    ggml_tensor * src_persist_inter = dst->src[7]; // optional: external f32/f16 buffer

    GGML_TENSOR_LOCALS(int64_t, neq, src_q, ne);
    GGML_TENSOR_LOCALS(size_t , nbq, src_q, nb);
    GGML_TENSOR_LOCALS(int64_t, nek, src_k, ne);
    GGML_TENSOR_LOCALS(size_t , nbk, src_k, nb);
    GGML_TENSOR_LOCALS(int64_t, nev, src_v, ne);
    GGML_TENSOR_LOCALS(size_t,  nbv, src_v, nb);
    GGML_TENSOR_LOCALS(size_t,  nbb, src_beta, nb);

    const int64_t S_v      = nev0;
    const int64_t H        = nev1;
    const int64_t n_tokens = nev2;
    const int64_t n_seqs   = nev3;

    const bool kda = (src_g->ne[0] == S_v);

    GGML_ASSERT(neq1 == nek1);
    const int64_t neqk1 = neq1;

    const int64_t rq3 = nev3 / neq3;

    const float * q_d = (const float *) src_q->data;
    const float * k_d = (const float *) src_k->data;
    const float * v_d = (const float *) src_v->data;
    const float * g_d = (const float *) src_g->data;
    const float * b_d = (const float *) src_beta->data;

    const float * s_d   = (const float *) src_state->data;
    float *       dst_d = (float *) dst->data;

    GGML_ASSERT(ggml_is_contiguous_rows(src_q));
    GGML_ASSERT(ggml_is_contiguous_rows(src_k));
    GGML_ASSERT(ggml_is_contiguous_rows(src_v));
    GGML_ASSERT(ggml_are_same_stride(src_q, src_k));
    GGML_ASSERT(src_g->ne[0] == 1 || kda);
    GGML_ASSERT(ggml_is_contiguous(src_g));
    GGML_ASSERT(ggml_is_contiguous(src_beta));
    GGML_ASSERT(ggml_is_contiguous(src_state));

    if (src_parent != nullptr) {
        GGML_ASSERT(src_parent->type == GGML_TYPE_I32);
        GGML_ASSERT(ggml_is_contiguous(src_parent));
        GGML_ASSERT(ggml_nelements(src_parent) == n_tokens * n_seqs);
        // Tree mode requires an external persist buffer: the embedded dst
        // region's element type is fixed to f32 by the op factory, but
        // tree-mode reloads need to address a buffer that may also be f16.
        GGML_ASSERT(src_persist_inter != nullptr &&
                    "GATED_DELTA_NET_WITH_HISTORY: tree mode (parent_ids != nullptr) "
                    "requires an external persist_inter buffer (src[7])");
    }

    if (src_persist_inter != nullptr) {
        GGML_ASSERT(src_persist_inter->type == GGML_TYPE_F32 ||
                    src_persist_inter->type == GGML_TYPE_F16);
        GGML_ASSERT(ggml_is_contiguous(src_persist_inter));
        // External buffer must be sized to hold the per-token intermediate
        // states for this verify call: [S_v, S_v, H, n_tokens, n_seqs].
        // Asserted on element count to allow the caller's allocator to
        // round up storage for max_tokens / max_seqs.
        GGML_ASSERT(ggml_nelements(src_persist_inter) >= S_v * S_v * H * n_tokens * n_seqs);
    }

    const int64_t sq1 = nbq1 / sizeof(float);
    const int64_t sq2 = nbq2 / sizeof(float);
    const int64_t sq3 = nbq3 / sizeof(float);
    const int64_t sv1 = nbv1 / sizeof(float);
    const int64_t sv2 = nbv2 / sizeof(float);
    const int64_t sv3 = nbv3 / sizeof(float);
    const int64_t sb1 = nbb1 / sizeof(float);
    const int64_t sb2 = nbb2 / sizeof(float);
    const int64_t sb3 = nbb3 / sizeof(float);

    const float scale = 1.0f / sqrtf((float) S_v);

    // dst layout (floats):
    //   attn:           S_v * H * n_tokens * n_seqs
    //   final_state:    S_v * S_v * H * n_seqs
    //   state_history:  S_v * S_v * H * n_tokens * n_seqs  (this region)
    //
    // When src_persist_inter is non-null, the kernel spills to that external
    // buffer instead of the embedded region. The embedded region inside dst
    // is then unused (caller still allocates it via the op factory's shape
    // for layout compatibility; we accept the small VRAM cost in chain mode
    // to keep the dst shape stable).
    float * embedded_history_d = dst_d
        + S_v * H * n_tokens * n_seqs
        + S_v * S_v * H * n_seqs;

    cudaStream_t stream = ctx.stream();

    const bool tree_mode = (src_parent != nullptr);
    const int * parent_ids_d = tree_mode ? (const int *) src_parent->data : nullptr;

    void * persist_inter_d = (src_persist_inter != nullptr) ? src_persist_inter->data : nullptr;
    const bool persist_is_f16 = (src_persist_inter != nullptr) && (src_persist_inter->type == GGML_TYPE_F16);

    // Distinguish the two tree-mode contracts via dst->op:
    //   - WITH_HISTORY (tree mode) + cpy after: write into dst's f32 embedded
    //     region; the graph builder appends a ggml_cpy(state_history_view,
    //     persist_inter) after the op. Unified path with Vulkan to dodge a
    //     same-thread read-after-write hazard on a separate f16 storage buffer
    //     on some Vulkan implementations.
    //   - WITH_HISTORY_TREE_PERSIST: write per-token states straight into
    //     persist_inter (f16 or f32, in-kernel type-converted). No follow-up
    //     cpy. Saves ~9 ms/iter on tree-budget-22 Qwen3.5-27B; lucebox
    //     reference: `ggml_gated_delta_net_tree_persist`. CUDA-only — see
    //     the `case GGML_OP_GATED_DELTA_NET_WITH_HISTORY_TREE_PERSIST` arm
    //     in `ggml_backend_cuda_device_supports_op`.
    //   - chain mode + external persist_inter: keep writing straight
    //     into persist_inter (chain-compatibility path).
    //   - chain mode without persist_inter: embedded f32 region only.
    const bool tree_persist_op =
        (dst->op == GGML_OP_GATED_DELTA_NET_WITH_HISTORY_TREE_PERSIST);
    GGML_ASSERT(!tree_persist_op || tree_mode);
    GGML_ASSERT(!tree_persist_op || persist_inter_d != nullptr);

    #define GDN_LAUNCH(KDA_VAL)                                                                 \
        do {                                                                                    \
            if (tree_persist_op) {                                                              \
                /* CUDA-only fast tree path: kernel writes per-token states */                  \
                /* directly into persist_inter (no follow-up cpy needed). */                    \
                if (persist_is_f16) {                                                           \
                    __half * persist_typed = (__half *) persist_inter_d;                        \
                    launch_gated_delta_net<KDA_VAL, true, __half>(                              \
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d,                                    \
                        persist_typed, parent_ids_d,                                            \
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                 \
                        sb1, sb2, sb3, neqk1, rq3, scale, stream);                              \
                } else {                                                                        \
                    float * persist_typed = (float *) persist_inter_d;                          \
                    launch_gated_delta_net<KDA_VAL, true, float>(                               \
                        q_d, k_d, v_d, g_d, b_d, s_d, dst_d,                                    \
                        persist_typed, parent_ids_d,                                            \
                        S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                 \
                        sb1, sb2, sb3, neqk1, rq3, scale, stream);                              \
                }                                                                               \
            } else if (tree_mode) {                                                             \
                launch_gated_delta_net<KDA_VAL, true, float>(                                   \
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d,                                        \
                    embedded_history_d, parent_ids_d,                                           \
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                     \
                    sb1, sb2, sb3, neqk1, rq3, scale, stream);                                  \
            } else if (persist_inter_d == nullptr) {                                            \
                launch_gated_delta_net<KDA_VAL, false, float>(                                  \
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d,                                        \
                    embedded_history_d, /*parent_ids=*/nullptr,                                 \
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                     \
                    sb1, sb2, sb3, neqk1, rq3, scale, stream);                                  \
            } else if (persist_is_f16) {                                                        \
                __half * persist_typed = (__half *) persist_inter_d;                            \
                launch_gated_delta_net<KDA_VAL, false, __half>(                                 \
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d,                                        \
                    persist_typed, /*parent_ids=*/nullptr,                                      \
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                     \
                    sb1, sb2, sb3, neqk1, rq3, scale, stream);                                  \
            } else {                                                                            \
                float * persist_typed = (float *) persist_inter_d;                              \
                launch_gated_delta_net<KDA_VAL, false, float>(                                  \
                    q_d, k_d, v_d, g_d, b_d, s_d, dst_d,                                        \
                    persist_typed, /*parent_ids=*/nullptr,                                      \
                    S_v, H, n_tokens, n_seqs, sq1, sq2, sq3, sv1, sv2, sv3,                     \
                    sb1, sb2, sb3, neqk1, rq3, scale, stream);                                  \
            }                                                                                   \
        } while (0)

    if (kda) {
        GDN_LAUNCH(true);
    } else {
        GDN_LAUNCH(false);
    }
    #undef GDN_LAUNCH
}

// DFlash state-history fixup: select state_history[..., k_index, :] and copy
// it into a [S_v, S_v, H_v, n_seqs] result tensor.
//
// state_history layout (contiguous floats):
//   [S_v, S_v, H_v, n_tokens, n_seqs]  - viewed as 4-D
//   [S_v, S_v, H_v, n_tokens * n_seqs] by ggml.
//
// k_index is an I32 tensor read on-device.
//   - 1 element  -> chain mode: same scalar k for every seq.
//   - n_seqs     -> tree mode: per-seq k_index; seq `s` reads
//                   `k_index_ptr[s]`. -1 in any slot triggers the fallback
//                   copy for that seq.
//
// When `k_index < 0` (host signal "no fixup needed"), the kernel
// copies `fallback` -> dst for that seq instead of selecting from
// state_history. `fallback` is expected to be contiguous and have the same
// element count as dst. May be nullptr if the caller guarantees k >= 0.
// Templated on the persistent state-history element type so the kernel can
// read either F32 (chain mode) or F16 (tree mode). The dst tensor stays F32
// (the GDN op's `state` input element type). SrcT = float compiles to a
// plain load; SrcT = __half emits a __half2float conversion on read.
template <typename SrcT>
__global__ void gated_delta_net_state_select_cuda(
        const SrcT  * state_history,
        const int   * k_index_ptr,
        const float * fallback,         // may be nullptr
        float       * dst,
        int           S_v,
        int           H_v,
        int           n_tokens,
        int           n_seqs,
        int           k_index_count) {  // 1 (chain) or n_seqs (tree)
    const int seq    = blockIdx.z;
    const int h      = blockIdx.y;
    const int col    = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane   = threadIdx.x;

    if (col >= S_v) return;

    // Per-seq k_index when count > 1, else the single scalar.
    const int k_idx = (k_index_count == 1) ? k_index_ptr[0] : k_index_ptr[seq];

    const int64_t slab_elems = (int64_t) S_v * S_v;
    float       * dst_slab = dst + ((int64_t) seq * H_v + h) * slab_elems;

    if (k_idx < 0) {
        // No-fixup path: copy fallback -> dst. When fallback is nullptr
        // the caller guaranteed this branch wouldn't be taken; treat as
        // bug.
        if (fallback == nullptr) {
            return;
        }
        const float * fb_slab = fallback + ((int64_t) seq * H_v + h) * slab_elems;
        for (int r = lane; r < S_v; r += blockDim.x) {
            dst_slab[col * S_v + r] = fb_slab[col * S_v + r];
        }
        return;
    }

    // Defensive clamp.
    const int t = (k_idx >= n_tokens ? (n_tokens - 1) : k_idx);

    const SrcT * src_slab = state_history
        + ((int64_t) seq * n_tokens * H_v + (int64_t) t * H_v + h) * slab_elems;

    for (int r = lane; r < S_v; r += blockDim.x) {
        // Conversion is a no-op for SrcT = float; F16->F32 promotion for
        // SrcT = __half. The kernel doesn't fuse anything else here so
        // the extra widening cost is buried under global-memory latency.
        dst_slab[col * S_v + r] = (float) src_slab[col * S_v + r];
    }
}

void ggml_cuda_op_gated_delta_net_state_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_sh     = dst->src[0];
    ggml_tensor * src_kindex = dst->src[1];
    ggml_tensor * src_fb     = dst->src[2]; // optional fallback

    // src_sh may be F32 (chain) or F16 (tree). Same shape; the kernel picks
    // the SrcT template at runtime via the dtype below.
    GGML_ASSERT(src_sh->type == GGML_TYPE_F32 || src_sh->type == GGML_TYPE_F16);
    GGML_ASSERT(src_kindex->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src_sh));

    const int64_t S_v      = src_sh->ne[0];
    const int64_t S_v2     = src_sh->ne[1];
    const int64_t H_v      = src_sh->ne[2];
    const int64_t n_tok_x_seqs = src_sh->ne[3];

    GGML_ASSERT(S_v2 == S_v);

    // The op factory shapes dst as [S_v, S_v, H_v, n_seqs]. n_tokens is
    // derived as src_sh->ne[3] / n_seqs.
    const int64_t n_seqs   = dst->ne[3];
    GGML_ASSERT(dst->ne[0] == S_v && dst->ne[1] == S_v && dst->ne[2] == H_v);
    GGML_ASSERT(n_tok_x_seqs % n_seqs == 0);
    const int64_t n_tokens = n_tok_x_seqs / n_seqs;

    // k_index is either a scalar (chain mode) or an [n_seqs] vector (tree mode).
    const int64_t k_index_count = ggml_nelements(src_kindex);
    GGML_ASSERT(k_index_count == 1 || k_index_count == n_seqs);

    const int   * ki_d  = (const int   *) src_kindex->data;
    const float * fb_d  = (src_fb != nullptr) ? (const float *) src_fb->data : nullptr;
    float       * dst_d = (float       *) dst->data;

    if (src_fb != nullptr) {
        GGML_ASSERT(src_fb->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_nelements(src_fb) == ggml_nelements(dst));
    }

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int num_warps = 4;

    dim3 grid((S_v + num_warps - 1) / num_warps, H_v, n_seqs);
    dim3 block(warp_size, num_warps, 1);

    cudaStream_t stream = ctx.stream();
    if (src_sh->type == GGML_TYPE_F16) {
        const __half * sh_d = (const __half *) src_sh->data;
        gated_delta_net_state_select_cuda<__half><<<grid, block, 0, stream>>>(
            sh_d, ki_d, fb_d, dst_d,
            (int) S_v, (int) H_v, (int) n_tokens, (int) n_seqs,
            (int) k_index_count);
    } else {
        const float * sh_d = (const float *) src_sh->data;
        gated_delta_net_state_select_cuda<float><<<grid, block, 0, stream>>>(
            sh_d, ki_d, fb_d, dst_d,
            (int) S_v, (int) H_v, (int) n_tokens, (int) n_seqs,
            (int) k_index_count);
    }
}

// DFlash conv-state fixup kernel.
//
// Layout (contiguous floats, row-major):
//   conv_history: [conv_history_rows, conv_channels, n_seqs]
//   fallback:     [conv_state_rows,    conv_channels, n_seqs]
//   dst:          [conv_state_rows,    conv_channels, n_seqs]
//
// k_index >= 0 -> dst[r, c, s] = conv_history[r + (k_index + 1), c, s]
//                 for r in [0, conv_state_rows).
// k_index <  0 -> dst[r, c, s] = fallback[r, c, s].
//
// Thread layout: blockIdx.x = channel-chunk, blockIdx.y = seq, threadIdx.x =
// channel-lane, threadIdx.y = row (covers all conv_state_rows in one block).
__global__ void dflash_conv_state_history_select_cuda(
        const float * conv_history,
        const int   * k_index_ptr,
        const float * fallback,
        float       * dst,
        int           conv_state_rows,
        int           conv_history_rows,
        int           conv_channels,
        int           n_seqs) {
    const int seq      = blockIdx.y;
    const int c_chunk  = blockIdx.x;
    const int c_lane   = threadIdx.x;
    const int r        = threadIdx.y;
    const int c        = c_chunk * blockDim.x + c_lane;

    if (c >= conv_channels) return;
    if (r >= conv_state_rows) return;

    const int k_idx = *k_index_ptr;

    // dst is contiguous [conv_state_rows, conv_channels, n_seqs].
    // Row-major: dst[r, c, s] sits at offset (s * channels + c) * rows + r.
    const int64_t dst_idx =
        ((int64_t) seq * conv_channels + c) * conv_state_rows + r;

    if (k_idx < 0) {
        // Fallback path: same layout as dst.
        const int64_t fb_idx =
            ((int64_t) seq * conv_channels + c) * conv_state_rows + r;
        dst[dst_idx] = fallback[fb_idx];
        return;
    }

    // History path: pick rows [k_idx + 1, k_idx + 1 + conv_state_rows)
    // of conv_history along dim 0. Defensive clamp.
    int row_off = k_idx + 1;
    const int row_max = conv_history_rows - conv_state_rows;
    if (row_off < 0) row_off = 0;
    if (row_off > row_max) row_off = row_max;

    const int64_t src_idx =
        ((int64_t) seq * conv_channels + c) * conv_history_rows + (row_off + r);

    dst[dst_idx] = conv_history[src_idx];
}

void ggml_cuda_op_dflash_conv_state_history_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_hist   = dst->src[0];
    ggml_tensor * src_kindex = dst->src[1];
    ggml_tensor * src_fb     = dst->src[2];

    GGML_ASSERT(src_hist->type    == GGML_TYPE_F32);
    GGML_ASSERT(src_kindex->type  == GGML_TYPE_I32);
    GGML_ASSERT(src_fb            != nullptr);
    GGML_ASSERT(src_fb->type      == GGML_TYPE_F32);
    GGML_ASSERT(ggml_nelements(src_kindex) == 1);
    GGML_ASSERT(ggml_is_contiguous(src_hist));

    const int64_t conv_state_rows   = dst->ne[0];
    const int64_t conv_channels     = dst->ne[1];
    const int64_t n_seqs            = dst->ne[2];
    const int64_t conv_history_rows = src_hist->ne[0];

    GGML_ASSERT(src_hist->ne[1] == conv_channels);
    GGML_ASSERT(src_hist->ne[2] == n_seqs);
    GGML_ASSERT(src_fb->ne[0]   == conv_state_rows);
    GGML_ASSERT(src_fb->ne[1]   == conv_channels);
    GGML_ASSERT(src_fb->ne[2]   == n_seqs);
    GGML_ASSERT(conv_history_rows >= conv_state_rows);

    const float * hist_d = (const float *) src_hist->data;
    const int   * ki_d   = (const int   *) src_kindex->data;
    const float * fb_d   = (const float *) src_fb->data;
    float       * dst_d  = (float       *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = (int) conv_state_rows;  // 3 for conv_kernel_size = 4
    const int channels_per_block = warp_size;

    dim3 grid((conv_channels + channels_per_block - 1) / channels_per_block, n_seqs, 1);
    dim3 block(channels_per_block, rows_per_block, 1);

    cudaStream_t stream = ctx.stream();
    dflash_conv_state_history_select_cuda<<<grid, block, 0, stream>>>(
        hist_d, ki_d, fb_d, dst_d,
        (int) conv_state_rows, (int) conv_history_rows,
        (int) conv_channels, (int) n_seqs);
}

// tree-aware variant of dflash_conv_state_history_select.
//
// Difference from the chain kernel: when k_idx >= 0, instead of picking
// conv_history rows [k_idx + 1, k_idx + conv_state_rows), walk parent_ids
// starting from k_idx to compute virt[K-2..0] (K = conv_state_rows + 1):
//
//   virt[K-2] = k_idx;
//   virt[K-3] = parent_ids[virt[K-2]];     // first ancestor (may be -1 = root)
//   virt[K-4] = parent_ids[virt[K-3]];     // grandparent (or -1 - 1 = -2)
//   ...
//
// Convention: parent_ids[0] = -1 (root sentinel). Once a virt[k+1] is
// negative, we keep counting down (virt[k] = virt[k+1] - 1) so the row
// offset (K-1 + virt[k]) lands in the prev_state region (conv_history
// rows [0, K-2]). The result row r maps to virt index r (oldest first),
// so dst[r, c, s] = conv_history[K-1+virt[r], c, s].
//
// For chain-shape verifies (parent_ids[i] == i - 1), virt[r] = k_idx -
// (K-2-r), giving the same K-1 contiguous rows as the non-tree kernel.
__global__ void dflash_conv_state_history_select_tree_cuda(
        const float * conv_history,
        const int   * k_index_ptr,
        const int   * parent_ids,
        const float * fallback,
        float       * dst,
        int           conv_state_rows,
        int           conv_history_rows,
        int           conv_channels,
        int           n_seqs,
        int           parent_ids_n_tokens) {
    const int seq      = blockIdx.y;
    const int c_chunk  = blockIdx.x;
    const int c_lane   = threadIdx.x;
    const int r        = threadIdx.y;
    const int c        = c_chunk * blockDim.x + c_lane;

    if (c >= conv_channels) return;
    if (r >= conv_state_rows) return;

    const int k_idx = *k_index_ptr;

    const int64_t dst_idx =
        ((int64_t) seq * conv_channels + c) * conv_state_rows + r;

    if (k_idx < 0) {
        const int64_t fb_idx =
            ((int64_t) seq * conv_channels + c) * conv_state_rows + r;
        dst[dst_idx] = fallback[fb_idx];
        return;
    }

    // Walk parent chain to compute virt[r]. Each thread independently walks
    // (cheap: at most conv_state_rows hops, conv_state_rows = 3 for the
    // standard Qwen3.5 conv_kernel_size = 4). All threads in a block walk
    // the same path so warp-level execution is uniform.
    int virt_r = k_idx;
    for (int hop = conv_state_rows - 1; hop > r; --hop) {
        if (virt_r >= 0 && virt_r < parent_ids_n_tokens) {
            virt_r = parent_ids[virt_r];
        } else {
            // Already past the root sentinel: keep counting down so the
            // resulting row offset lands deeper into the prev_state region.
            virt_r = virt_r - 1;
        }
    }

    int row_off = (conv_state_rows /* = K-1 */) + virt_r;
    if (row_off < 0) row_off = 0;
    const int row_max_excl = conv_history_rows;
    if (row_off >= row_max_excl) row_off = row_max_excl - 1;

    const int64_t src_idx =
        ((int64_t) seq * conv_channels + c) * conv_history_rows + row_off;

    dst[dst_idx] = conv_history[src_idx];
}

void ggml_cuda_op_dflash_conv_state_history_select_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    ggml_tensor * src_hist    = dst->src[0];
    ggml_tensor * src_kindex  = dst->src[1];
    ggml_tensor * src_fb      = dst->src[2];
    ggml_tensor * src_parents = dst->src[3];

    GGML_ASSERT(src_hist->type    == GGML_TYPE_F32);
    GGML_ASSERT(src_kindex->type  == GGML_TYPE_I32);
    GGML_ASSERT(src_fb            != nullptr);
    GGML_ASSERT(src_fb->type      == GGML_TYPE_F32);
    GGML_ASSERT(src_parents       != nullptr);
    GGML_ASSERT(src_parents->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_nelements(src_kindex) == 1);
    GGML_ASSERT(ggml_is_contiguous(src_hist));
    GGML_ASSERT(ggml_is_contiguous(src_parents));

    const int64_t conv_state_rows   = dst->ne[0];
    const int64_t conv_channels     = dst->ne[1];
    const int64_t n_seqs            = dst->ne[2];
    const int64_t conv_history_rows = src_hist->ne[0];

    GGML_ASSERT(src_hist->ne[1] == conv_channels);
    GGML_ASSERT(src_hist->ne[2] == n_seqs);
    GGML_ASSERT(src_fb->ne[0]   == conv_state_rows);
    GGML_ASSERT(src_fb->ne[1]   == conv_channels);
    GGML_ASSERT(src_fb->ne[2]   == n_seqs);
    GGML_ASSERT(conv_history_rows >= conv_state_rows);

    // parent_ids is shaped [n_tokens, n_seqs] in the spec driver; tree-mode
    // verify is single-seq so n_seqs == 1, total elements = n_tokens.
    const int64_t parent_ids_total = ggml_nelements(src_parents);
    GGML_ASSERT(parent_ids_total > 0);

    const float * hist_d    = (const float *) src_hist->data;
    const int   * ki_d      = (const int   *) src_kindex->data;
    const float * fb_d      = (const float *) src_fb->data;
    const int   * parents_d = (const int   *) src_parents->data;
    float       * dst_d     = (float       *) dst->data;

    const int warp_size = ggml_cuda_info().devices[ggml_cuda_get_device()].warp_size;
    const int rows_per_block = (int) conv_state_rows;
    const int channels_per_block = warp_size;

    dim3 grid((conv_channels + channels_per_block - 1) / channels_per_block, n_seqs, 1);
    dim3 block(channels_per_block, rows_per_block, 1);

    cudaStream_t stream = ctx.stream();
    dflash_conv_state_history_select_tree_cuda<<<grid, block, 0, stream>>>(
        hist_d, ki_d, parents_d, fb_d, dst_d,
        (int) conv_state_rows, (int) conv_history_rows,
        (int) conv_channels, (int) n_seqs,
        (int) parent_ids_total);
}
