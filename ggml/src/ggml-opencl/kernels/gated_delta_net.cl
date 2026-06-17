#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

// Gated DeltaNet autoregressive recurrence for GGML_OP_GATED_DELTA_NET.
//
// This is the n_tokens == 1 Qwen3.5 decode specialization. The output tensor is
// packed as [o (S*H*N) | s_out (S*S*H*N)], matching ggml_gated_delta_net().
// State rows follow the upstream CPU layout: row j is stored at state[j*S + i].
#define GATED_DELTA_NET_S 128

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#elif defined(INTEL_GPU)
REQD_SUBGROUP_SIZE_32
#endif
kernel void kernel_gated_delta_net_f32_s128(
    global const char * q_g,    ulong off_q,
    global const char * k_g,    ulong off_k,
    global const char * v_g,    ulong off_v,
    ulong nbv1, ulong nbv2, ulong nbv3,
    global const char * g_g,    ulong off_g,
    global const char * b_g,    ulong off_b,
    global const char * s_in,   ulong off_s_in,
    global       char * o_g,    ulong off_o,
    global       char * s_out,  ulong off_s_out,
    int H,
    int g_ne0
) {
    const int S = GATED_DELTA_NET_S;
    const int h = get_group_id(1);
    const int n = get_group_id(2);
    const int tid = get_local_id(0);
    const int hn = n * H + h;

    const float q_scale = rsqrt((float) S);

    global const float * q_p     = (global const float *)(q_g + off_q)       + (size_t)hn * S;
    global const float * k_p     = (global const float *)(k_g + off_k)       + (size_t)hn * S;
    global const float * v_p     = (global const float *)(v_g + off_v + (ulong)h * nbv1 + (ulong)n * nbv3);
    global const float * g_p     = (global const float *)(g_g + off_g)       + (size_t)hn * g_ne0;
    global const float * b_p     = (global const float *)(b_g + off_b)       + (size_t)hn;
    global const float * s_in_p  = (global const float *)(s_in + off_s_in)   + (size_t)hn * S * S;
    global       float * o_p     = (global       float *)(o_g + off_o)       + (size_t)hn * S;
    global       float * s_out_p = (global       float *)(s_out + off_s_out) + (size_t)hn * S * S;

    __local float k_l[GATED_DELTA_NET_S];
    __local float q_l[GATED_DELTA_NET_S];
    __local float v_l[GATED_DELTA_NET_S];
    __local float d_l[GATED_DELTA_NET_S];
    __local float g_l[GATED_DELTA_NET_S];
    __local float beta_v;
    __local float partial_kq[4];

    k_l[tid] = k_p[tid];
    q_l[tid] = q_p[tid] * q_scale;
    v_l[tid] = v_p[tid];
    if (tid < 4) {
        partial_kq[tid] = 0.0f;
    }
    if (g_ne0 == 1) {
        if (tid == 0) {
            g_l[0] = g_p[0];
        }
    } else {
        g_l[tid] = g_p[tid];
    }
    if (tid == 0) {
        beta_v = b_p[0];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    float sk_j = 0.0f;
    float sq_j = 0.0f;
    for (int i = 0; i < S; ++i) {
        const float g_v  = (g_ne0 == 1) ? g_l[0] : g_l[i];
        const float gate = exp(g_v);
        const float s_ji = s_in_p[tid * S + i];
        sk_j += s_ji * gate * k_l[i];
        sq_j += s_ji * gate * q_l[i];
    }

    const float dj = beta_v * (v_l[tid] - sk_j);
    d_l[tid] = dj;

    const float kq_partial = sub_group_reduce_add(k_l[tid] * q_l[tid]);
    if (get_sub_group_local_id() == 0) {
        partial_kq[get_sub_group_id()] = kq_partial;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const float kq = partial_kq[0] + partial_kq[1] + partial_kq[2] + partial_kq[3];

    o_p[tid] = sq_j + dj * kq;

    for (int i = 0; i < S; ++i) {
        const float g_v  = (g_ne0 == 1) ? g_l[0] : g_l[i];
        const float gate = exp(g_v);
        const float s_ji = s_in_p[tid * S + i];
        s_out_p[tid * S + i] = s_ji * gate + d_l[tid] * k_l[i];
    }
}
