#include "ssm-conv.cuh"
#include "unary.cuh"

template <bool apply_silu, size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                    const float * __restrict__ bias,
                                    const int src0_nb0, const int src0_nb1, const int src0_nb2, const int src1_nb1,
                                    float * __restrict__ dst, const int dst_nb0, const int dst_nb1, const int dst_nb2,
                                    const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    float x[d_conv] = { 0.0f };
    float w[d_conv] = { 0.0f };

#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    for (int64_t i = 0; i < n_t; i++) {
        float sumf = 0.0f;

        if (i == 0) {
            for (size_t j = 0; j < d_conv; j++) {
                x[j] = x_block[tid * stride_x + j];
            }
        } else {
            x[(i - 1) % d_conv] = x_block[tid * stride_x + i + d_conv - 1];
        }

#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += x[(i + j) % d_conv] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu, size_t split_d_inner, size_t d_conv, int64_t split_n_t>
static __global__ void ssm_conv_long_token_f32(const float * __restrict__ src0, const float * __restrict__ src1,
                                               const float * __restrict__ bias,
                                               const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                               const int src1_nb1, float * __restrict__ dst, const int dst_nb0,
                                               const int dst_nb1, const int dst_nb2, const int64_t n_t) {
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;
    const int bidy = blockIdx.y;
    const int bidz = blockIdx.z;

    const float * x_block = (const float *) ((const char *) src0 + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1 +
                                             bidz * split_n_t * src0_nb0);
    const float * w_block = (const float *) ((const char *) src1 + bidy * split_d_inner * src1_nb1);
    float *       y_block =
        (float *) ((char *) dst + bidx * dst_nb2 + bidz * split_n_t * dst_nb1 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    const int64_t local_n_t = min(split_n_t, n_t - bidz * split_n_t);
    const int     n_cols    = d_conv - 1 + split_n_t;

    extern __shared__ float smem[];

    constexpr int load_cols   = d_conv - 1 + split_n_t;
    constexpr int total_elems = split_d_inner * load_cols;
    int row = tid / load_cols;
    int col = tid % load_cols;
#pragma unroll
    for (int idx = 0; idx < total_elems; idx += split_d_inner) {
        if (row < (int)split_d_inner) {
            smem[row * n_cols + col] = x_block[row * stride_x + col];
        }

        col += split_d_inner;
        row += col / load_cols;
        col  = col % load_cols;
        if (idx >= total_elems - tid - split_d_inner) {
            break;
        }
    }
    __syncthreads();

    // Load weights into registers (done once, small)
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    float b = bias != nullptr ? bias[bidy * split_d_inner + tid] : 0.0f;

    // Compute from shared memory
    for (int64_t i = 0; i < local_n_t; i++) {
        float sumf = 0.0f;
#pragma unroll
        for (size_t j = 0; j < d_conv; j++) {
            sumf += smem[tid * n_cols + i + j] * w[j];
        }
        sumf += b;
        y_block[i * stride_y + tid] = apply_silu ? ggml_cuda_op_silu_single(sumf) : sumf;
    }
}

template <bool apply_silu>
static void ssm_conv_f32_cuda(const float * src0, const float * src1, const float * bias, const int src0_nb0, const int src0_nb1,
                              const int src0_nb2, const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                              const int dst_nb2, const int64_t nc, const int64_t nr, const int64_t n_t,
                              const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        if (n_t <= 32) {
            const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
            ssm_conv_f32<apply_silu, threads, kNC><<<blocks, threads, 0, stream>>>(src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
                                                                       dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        } else {
            const int64_t split_n_t = 32;
            dim3          blocks(n_s, (nr + threads - 1) / threads, (n_t + split_n_t - 1) / split_n_t);
            const size_t  smem_size = threads * (kNC - 1 + split_n_t) * sizeof(float);
            ssm_conv_long_token_f32<apply_silu, threads, kNC, split_n_t><<<blocks, threads, smem_size, stream>>>(
                src0, src1, bias, src0_nb0, src0_nb1, src0_nb2, src1_nb1, dst, dst_nb0, dst_nb1, dst_nb2, n_t);
        }
    };

    switch (nc) {
        case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
        case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
        case 5: launch_kernel(std::integral_constant<int, 5>{}); break;
        case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("Only support kernel sizes 3, 4, 5, 9 right now.");
    }
}

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node, ggml_tensor * silu_dst) {
    const struct ggml_tensor * src0 = dst->src[0];  // conv_x
    const struct ggml_tensor * src1 = dst->src[1];  // conv1d.weight
    const bool fuse_bias = bias_add_node != nullptr;
    const bool fuse_silu = silu_dst != nullptr;

    // bias always comes with silu.
    GGML_ASSERT(!fuse_bias || fuse_silu);

    // The bias (when fused) is the non-conv operand of the ADD node.
    const struct ggml_tensor * bias = fuse_bias ? (bias_add_node->src[0] == dst ? bias_add_node->src[1] : bias_add_node->src[0]) : nullptr;

    // When fusing, write to silu_dst (the node downstream references).
    const struct ggml_tensor * out = fuse_silu ? silu_dst : dst;

    const int64_t nc  = src1->ne[0];                // d_conv
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = out->ne[1];                 // tokens per sequence
    const int64_t n_s = out->ne[2];                 // number of sequences in the batch

    GGML_ASSERT(out->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));

    const float * src0_d = (const float *) src0->data;
    const float * src1_d = (const float *) src1->data;
    const float * bias_d = fuse_bias ? (const float *) bias->data : nullptr;
    float *       dst_d  = (float *) out->data;
    cudaStream_t  stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(out->type == GGML_TYPE_F32);
    if (fuse_bias) {
        GGML_ASSERT(bias->type == GGML_TYPE_F32);
        GGML_ASSERT(ggml_is_contiguous(bias));
        GGML_ASSERT(ggml_nelements(bias) == nr);
    }

    if (fuse_silu) {
        ssm_conv_f32_cuda<true>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    } else {
        ssm_conv_f32_cuda<false>(src0_d, src1_d, bias_d, src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1], dst_d, out->nb[0], out->nb[1],
                          out->nb[2], nc, nr, n_t, n_s, stream);
    }
}

// ===========================================================================
// DFlash Phase 5: tree-mode ssm_conv kernel.
//
// For each new-token i in a tree-shaped batch, walks the parent chain K-1
// times via parent_ids[] to assemble the K-tap conv window from this token's
// own ancestor slots, instead of the sequential i..i+K-1 slide that the
// chain kernel uses. This is the conv-side equivalent of the gated_delta_net
// branch-point reload added in the same session: without it, sibling
// branches share ancestors that don't belong to them and the conv state
// cross-contaminates, killing acceptance at non-trivial tree depths.
//
// Virtual-slot encoding: a non-negative parent index `p` (= a token earlier
// in this same verify call) maps to sx slot (K-1 + p). A parent index of
// -1 means "before the block" — i.e., the K-1 slots of carry-in conv state.
// Each successive walk beyond -1 decrements by 1, so virtual slot -k maps to
// sx slot (K-1 - k), which indexes into the carry region [0, K-1).
//
// Mirrors lucebox/sglang's causal_conv1d HAS_EAGLE_TREE_CUSTOM_ATTN_MASK
// path. No bias / silu fusion: callers tag tree mode via a separate op so
// the chain-mode fusion pipeline stays untouched.
// ===========================================================================
template <size_t split_d_inner, size_t d_conv>
static __global__ void ssm_conv_tree_f32(
        const float * __restrict__ src0,         // sx: [K-1+n_t, d_inner, n_s]
        const float * __restrict__ src1,         // c:  [K, d_inner]
        const int   * __restrict__ parent_ids,   // [n_t, n_s]
        const int src0_nb0, const int src0_nb1, const int src0_nb2,
        const int src1_nb1,
        float * __restrict__ dst,                // [d_inner, n_t, n_s]
        const int dst_nb0, const int dst_nb1, const int dst_nb2,
        const int64_t n_t) {
    GGML_UNUSED(src0_nb0);
    const int tid  = threadIdx.x;
    const int bidx = blockIdx.x;  // sequence
    const int bidy = blockIdx.y;  // d_inner / split_d_inner

    const float * x_block = (const float *) ((const char *) src0
        + bidx * src0_nb2 + bidy * split_d_inner * src0_nb1);
    const float * w_block = (const float *) ((const char *) src1
        + bidy * split_d_inner * src1_nb1);
    float *       y_block = (float *) ((char *) dst
        + bidx * dst_nb2 + bidy * split_d_inner * dst_nb0);

    const int stride_x = src0_nb1 / sizeof(float);
    const int stride_w = src1_nb1 / sizeof(float);
    const int stride_y = dst_nb1 / sizeof(float);

    // Load kernel weights into registers (small, K-tap).
    float w[d_conv] = { 0.0f };
#pragma unroll
    for (size_t j = 0; j < d_conv; j++) {
        w[j] = w_block[tid * stride_w + j];
    }

    const int * parent_ids_seq = parent_ids + bidx * n_t;

    for (int64_t i = 0; i < n_t; i++) {
        // Walk the parent chain K-1 times to build the conv window.
        // ancestors[K-1] is token i itself; ancestors[k] for k < K-1 is the
        // virtual slot index of the (K-1-k)-th-most-recent ancestor.
        // Negative entries decay through the carry-in conv state region.
        int ancestors[d_conv];
        ancestors[d_conv - 1] = (int) i;
#pragma unroll
        for (int k = (int) d_conv - 2; k >= 0; k--) {
            int prev = ancestors[k + 1];
            int next;
            if (prev >= 0) {
                next = parent_ids_seq[prev]; // -1 if parent is before the block
            } else {
                next = prev - 1;             // continue decaying through old state
            }
            ancestors[k] = next;
        }

        float sumf = 0.0f;
#pragma unroll
        for (size_t k = 0; k < d_conv; k++) {
            // Map virtual slot -> sx slot: sx_slot = (K-1) + ancestors[k].
            const int sx_slot = (int) (d_conv - 1) + ancestors[k];
            const float x_val = x_block[tid * stride_x + sx_slot];
            sumf += x_val * w[k];
        }
        y_block[i * stride_y + tid] = sumf;
    }
}

static void ssm_conv_tree_f32_cuda(const float * src0, const float * src1, const int * parent_ids,
                                   const int src0_nb0, const int src0_nb1, const int src0_nb2,
                                   const int src1_nb1, float * dst, const int dst_nb0, const int dst_nb1,
                                   const int dst_nb2, const int64_t nc, const int64_t nr,
                                   const int64_t n_t, const int64_t n_s, cudaStream_t stream) {
    const int threads = 128;
    GGML_ASSERT(nr % threads == 0);

    const dim3 blocks(n_s, (nr + threads - 1) / threads, 1);
    auto launch_kernel = [&](auto NC) {
        constexpr int kNC = decltype(NC)::value;
        ssm_conv_tree_f32<threads, kNC><<<blocks, threads, 0, stream>>>(
            src0, src1, parent_ids, src0_nb0, src0_nb1, src0_nb2, src1_nb1,
            dst, dst_nb0, dst_nb1, dst_nb2, n_t);
    };

    switch (nc) {
        case 3: launch_kernel(std::integral_constant<int, 3>{}); break;
        case 4: launch_kernel(std::integral_constant<int, 4>{}); break;
        case 5: launch_kernel(std::integral_constant<int, 5>{}); break;
        case 9: launch_kernel(std::integral_constant<int, 9>{}); break;
        default: GGML_ABORT("ssm_conv_tree only supports kernel sizes 3, 4, 5, 9.");
    }
}

void ggml_cuda_op_ssm_conv_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const struct ggml_tensor * src0       = dst->src[0]; // sx: conv input window
    const struct ggml_tensor * src1       = dst->src[1]; // c:  conv weights
    const struct ggml_tensor * parent_ids = dst->src[2]; // [n_t, n_s] int32

    GGML_ASSERT(parent_ids != nullptr);
    GGML_ASSERT(parent_ids->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(parent_ids));

    const int64_t nc  = src1->ne[0];                // d_conv (K)
    const int64_t nr  = src0->ne[1];                // d_inner
    const int64_t n_t = dst->ne[1];                 // tokens per sequence
    const int64_t n_s = dst->ne[2];                 // number of sequences

    GGML_ASSERT(dst->ne[0] == nr);
    GGML_ASSERT(src0->nb[0] == sizeof(float));
    GGML_ASSERT(src1->nb[0] == sizeof(float));
    GGML_ASSERT(src0->nb[1] == src0->ne[0] * sizeof(float));
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);
    GGML_ASSERT(parent_ids->ne[0] == n_t);
    GGML_ASSERT(parent_ids->ne[1] == n_s);

    const float * src0_d       = (const float *) src0->data;
    const float * src1_d       = (const float *) src1->data;
    const int   * parent_ids_d = (const int   *) parent_ids->data;
    float       * dst_d        = (float       *) dst->data;
    cudaStream_t  stream       = ctx.stream();

    ssm_conv_tree_f32_cuda(src0_d, src1_d, parent_ids_d,
        src0->nb[0], src0->nb[1], src0->nb[2], src1->nb[1],
        dst_d, dst->nb[0], dst->nb[1], dst->nb[2],
        nc, nr, n_t, n_s, stream);
}
