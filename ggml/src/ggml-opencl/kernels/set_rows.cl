#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// v = { mp, L, d }
inline uint fastdiv(uint n, uint4 v) {
    uint msbs;
    msbs = mul_hi(n, v.s0);
    return (msbs + n) >> v.s1;
}
inline uint fastmod(uint n, uint4 v) {
    uint q = fastdiv(n, v);
    return n - q * v.s2;
}

kernel void kernel_set_rows_f32_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global float * dst_row = (global float *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = (float)src_row[ind];
    }
}

kernel void kernel_set_rows_f16_i64(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    long i1 = ((global long *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global half  * dst_row = (global half  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

kernel void kernel_set_rows_f32_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global float * dst_row = (global float *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = (float)src_row[ind];
    }
}

kernel void kernel_set_rows_f16_i32(
        global char * src0,
        ulong         offset0,
        global char * src1,
        ulong         offset1,
        global char * dst,
        ulong         offsetd,
        int           ne01,
        ulong         nb01,
        ulong         nb02,
        ulong         nb03,
        uint4         ne11,
        uint4         ne12,
        ulong         nb10,
        ulong         nb11,
        ulong         nb12,
        int           nblk0,
        ulong         nb1,
        ulong         nb2,
        ulong         nb3
) {
    src0 = src0 + offset0;
    src1 = src1 + offset1;
    dst  = dst  + offsetd;

    int i03 = get_group_id(2);
    int i02 = get_group_id(1);
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);

    if (i01 >= ne01) {
        return;
    }

    //int i12 = i03%ne12;
    //int i11 = i02%ne11;
    int i12 = fastmod(i03, ne12);
    int i11 = fastmod(i02, ne11);

    int i10 = i01;
    int i1  = ((global int *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0];

    global half  * dst_row = (global half  *) (dst  +  i1*nb1  + i02*nb2  + i03*nb3);
    global float * src_row = (global float *) (src0 + i01*nb01 + i02*nb02 + i03*nb03);

    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {
        dst_row[ind] = src_row[ind];
    }
}

// =========================================================================
// Quantized and BF16 set_rows kernels
// =========================================================================

typedef char  int8_t;
typedef uchar uint8_t;
typedef uint  uint32_t;

#define QK4_0  32
#define QK4_1  32
#define QK5_0  32
#define QK5_1  32
#define QK8_0  32
#define QK4_NL 32

typedef struct { half    d;          uint8_t qs[QK4_0/2]; } block_q4_0;
typedef struct { half    d; half m;  uint8_t qs[QK4_1/2]; } block_q4_1;
typedef struct { half    d;          uint8_t qh[4]; uint8_t qs[QK5_0/2]; } block_q5_0;
typedef struct { half    d; half m;  uint8_t qh[4]; uint8_t qs[QK5_1/2]; } block_q5_1;
typedef struct { half    d;          int8_t  qs[QK8_0]; } block_q8_0;
typedef struct { half    d;          uint8_t qs[QK4_NL/2]; } block_iq4_nl;

constant int8_t kvalues_iq4nl[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113
};

inline ushort float_to_bf16(float f) {
    uint u = as_uint(f);
    u += 0x7FFF + ((u >> 16) & 1);
    return (ushort)(u >> 16);
}

inline int best_index_iq4nl(float x) {
    if (x <= (float)kvalues_iq4nl[0])  return 0;
    if (x >= (float)kvalues_iq4nl[15]) return 15;
    int lo = 0, hi = 15;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (x < (float)kvalues_iq4nl[mid]) hi = mid; else lo = mid;
    }
    return x - (float)kvalues_iq4nl[lo] < (float)kvalues_iq4nl[hi] - x ? lo : hi;
}

// --- Quantization helpers (f32 source → quantized block destination) ---

inline void quantize_f32_q4_0(global const float * x, global block_q4_0 * y) {
    float amax = 0.0f, vmax = 0.0f;
    for (int j = 0; j < QK4_0; ++j) {
        float v = x[j];
        if (amax < fabs(v)) { amax = fabs(v); vmax = v; }
    }
    float d  = vmax / -8.0f;
    float id = d != 0.0f ? 1.0f/d : 0.0f;
    y->d = (half)d;
    for (int j = 0; j < QK4_0/2; ++j) {
        uint8_t xi0 = (uint8_t)clamp((int)(x[j]         * id + 8.5f), 0, 15);
        uint8_t xi1 = (uint8_t)clamp((int)(x[QK4_0/2+j] * id + 8.5f), 0, 15);
        y->qs[j] = xi0 | (xi1 << 4);
    }
}

inline void quantize_f32_q4_1(global const float * x, global block_q4_1 * y) {
    float vmin = FLT_MAX, vmax = -FLT_MAX;
    for (int j = 0; j < QK4_1; ++j) {
        vmin = fmin(vmin, x[j]);
        vmax = fmax(vmax, x[j]);
    }
    float d  = (vmax - vmin) / 15.0f;
    float id = d != 0.0f ? 1.0f/d : 0.0f;
    y->d = (half)d;
    y->m = (half)vmin;
    for (int j = 0; j < QK4_1/2; ++j) {
        uint8_t xi0 = (uint8_t)clamp((int)((x[j]         - vmin)*id + 0.5f), 0, 15);
        uint8_t xi1 = (uint8_t)clamp((int)((x[QK4_1/2+j] - vmin)*id + 0.5f), 0, 15);
        y->qs[j] = xi0 | (xi1 << 4);
    }
}

inline void quantize_f32_q5_0(global const float * x, global block_q5_0 * y) {
    float amax = 0.0f, vmax = 0.0f;
    for (int j = 0; j < QK5_0; ++j) {
        float v = x[j];
        if (amax < fabs(v)) { amax = fabs(v); vmax = v; }
    }
    float d  = vmax / -16.0f;
    float id = d != 0.0f ? 1.0f/d : 0.0f;
    y->d = (half)d;
    uint32_t qh = 0;
    for (int j = 0; j < QK5_0/2; ++j) {
        uint8_t xi0 = (uint8_t)clamp((int)(x[j]         * id + 16.5f), 0, 31);
        uint8_t xi1 = (uint8_t)clamp((int)(x[QK5_0/2+j] * id + 16.5f), 0, 31);
        y->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((uint32_t)(xi0 & 0x10) >> 4) << (j);
        qh |= ((uint32_t)(xi1 & 0x10) >> 4) << (j + QK5_0/2);
    }
    y->qh[0] = (uint8_t)(qh);
    y->qh[1] = (uint8_t)(qh >> 8);
    y->qh[2] = (uint8_t)(qh >> 16);
    y->qh[3] = (uint8_t)(qh >> 24);
}

inline void quantize_f32_q5_1(global const float * x, global block_q5_1 * y) {
    float vmin = x[0], vmax = x[0];
    for (int j = 1; j < QK5_1; ++j) {
        vmin = fmin(vmin, x[j]);
        vmax = fmax(vmax, x[j]);
    }
    float d  = (vmax - vmin) / 31.0f;
    float id = d != 0.0f ? 1.0f/d : 0.0f;
    y->d = (half)d;
    y->m = (half)vmin;
    uint32_t qh = 0;
    for (int j = 0; j < QK5_1/2; ++j) {
        uint8_t xi0 = (uint8_t)clamp((int)((x[j]         - vmin)*id + 0.5f), 0, 31);
        uint8_t xi1 = (uint8_t)clamp((int)((x[QK5_1/2+j] - vmin)*id + 0.5f), 0, 31);
        y->qs[j] = (xi0 & 0xf) | ((xi1 & 0xf) << 4);
        qh |= ((uint32_t)(xi0 & 0x10) >> 4) << (j);
        qh |= ((uint32_t)(xi1 & 0x10) >> 4) << (j + QK5_1/2);
    }
    y->qh[0] = (uint8_t)(qh);
    y->qh[1] = (uint8_t)(qh >> 8);
    y->qh[2] = (uint8_t)(qh >> 16);
    y->qh[3] = (uint8_t)(qh >> 24);
}

inline void quantize_f32_q8_0(global const float * x, global block_q8_0 * y) {
    float amax = 0.0f;
    for (int j = 0; j < QK8_0; ++j) {
        amax = fmax(amax, fabs(x[j]));
    }
    float d  = amax / 127.0f;
    float id = d != 0.0f ? 1.0f/d : 0.0f;
    y->d = (half)d;
    for (int j = 0; j < QK8_0; ++j) {
        y->qs[j] = (int8_t)round(x[j] * id);
    }
}

inline void quantize_f32_iq4_nl(global const float * x, global block_iq4_nl * y) {
    float amax = 0.0f, vmax = 0.0f;
    for (int j = 0; j < QK4_NL; ++j) {
        float v = x[j];
        if (amax < fabs(v)) { amax = fabs(v); vmax = v; }
    }
    float d  = vmax / (float)kvalues_iq4nl[0];
    float id = d != 0.0f ? 1.0f/d : 0.0f;
    float sumqx = 0.0f, sumq2 = 0.0f;
    for (int j = 0; j < QK4_NL/2; ++j) {
        uint8_t xi0 = (uint8_t)best_index_iq4nl(x[j]          * id);
        uint8_t xi1 = (uint8_t)best_index_iq4nl(x[QK4_NL/2+j] * id);
        y->qs[j] = xi0 | (xi1 << 4);
        float v0 = (float)kvalues_iq4nl[xi0];
        float v1 = (float)kvalues_iq4nl[xi1];
        float w0 = x[j]          * x[j];
        float w1 = x[QK4_NL/2+j] * x[QK4_NL/2+j];
        sumqx += w0*v0*x[j] + w1*v1*x[QK4_NL/2+j];
        sumq2 += w0*v0*v0   + w1*v1*v1;
    }
    y->d = (half)(sumq2 > 0.0f ? sumqx/sumq2 : d);
}

// --- Kernel generation macros ---

#define KERNEL_SET_ROWS_QUANT_IMPL(KERNEL_NAME, IDX_TYPE, BLOCK_TYPE, QK, QUANTIZE_FUNC) \
kernel void KERNEL_NAME(                                                       \
        global char * src0,  ulong offset0,                                    \
        global char * src1,  ulong offset1,                                    \
        global char * dst,   ulong offsetd,                                    \
        int ne01, ulong nb01, ulong nb02, ulong nb03,                         \
        uint4 ne11, uint4 ne12,                                                \
        ulong nb10, ulong nb11, ulong nb12,                                   \
        int nblk0, ulong nb1, ulong nb2, ulong nb3                            \
) {                                                                            \
    src0 += offset0; src1 += offset1; dst += offsetd;                          \
    int i03 = get_group_id(2);                                                 \
    int i02 = get_group_id(1);                                                 \
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);            \
    if (i01 >= ne01) return;                                                   \
    int i12 = fastmod(i03, ne12);                                              \
    int i11 = fastmod(i02, ne11);                                              \
    int i10 = i01;                                                             \
    IDX_TYPE i1 = ((global IDX_TYPE *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0]; \
    global float * src_row = (global float *)(src0 + i01*nb01 + i02*nb02 + i03*nb03); \
    global char  * dst_row = (global char  *)(dst  +  i1*nb1  + i02*nb2  + i03*nb3);  \
    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {   \
        QUANTIZE_FUNC(src_row + ind*(QK), (global BLOCK_TYPE *)(dst_row) + ind); \
    }                                                                          \
}

#define KERNEL_SET_ROWS_QUANT(NAME, BLOCK_TYPE, QK, QUANTIZE_FUNC)             \
    KERNEL_SET_ROWS_QUANT_IMPL(kernel_set_rows_##NAME##_i64, long, BLOCK_TYPE, QK, QUANTIZE_FUNC) \
    KERNEL_SET_ROWS_QUANT_IMPL(kernel_set_rows_##NAME##_i32, int,  BLOCK_TYPE, QK, QUANTIZE_FUNC)

KERNEL_SET_ROWS_QUANT(q4_0,   block_q4_0,   QK4_0,  quantize_f32_q4_0)
KERNEL_SET_ROWS_QUANT(q4_1,   block_q4_1,   QK4_1,  quantize_f32_q4_1)
KERNEL_SET_ROWS_QUANT(q5_0,   block_q5_0,   QK5_0,  quantize_f32_q5_0)
KERNEL_SET_ROWS_QUANT(q5_1,   block_q5_1,   QK5_1,  quantize_f32_q5_1)
KERNEL_SET_ROWS_QUANT(q8_0,   block_q8_0,   QK8_0,  quantize_f32_q8_0)
KERNEL_SET_ROWS_QUANT(iq4_nl, block_iq4_nl, QK4_NL, quantize_f32_iq4_nl)

// --- BF16 (element-wise, blck_size = 1) ---

#define KERNEL_SET_ROWS_BF16_IMPL(KERNEL_NAME, IDX_TYPE)                       \
kernel void KERNEL_NAME(                                                       \
        global char * src0,  ulong offset0,                                    \
        global char * src1,  ulong offset1,                                    \
        global char * dst,   ulong offsetd,                                    \
        int ne01, ulong nb01, ulong nb02, ulong nb03,                         \
        uint4 ne11, uint4 ne12,                                                \
        ulong nb10, ulong nb11, ulong nb12,                                   \
        int nblk0, ulong nb1, ulong nb2, ulong nb3                            \
) {                                                                            \
    src0 += offset0; src1 += offset1; dst += offsetd;                          \
    int i03 = get_group_id(2);                                                 \
    int i02 = get_group_id(1);                                                 \
    int i01 = get_group_id(0)*get_local_size(1) + get_local_id(1);            \
    if (i01 >= ne01) return;                                                   \
    int i12 = fastmod(i03, ne12);                                              \
    int i11 = fastmod(i02, ne11);                                              \
    int i10 = i01;                                                             \
    IDX_TYPE i1 = ((global IDX_TYPE *)(src1 + i10*nb10 + i11*nb11 + i12*nb12))[0]; \
    global float  * src_row = (global float  *)(src0 + i01*nb01 + i02*nb02 + i03*nb03); \
    global ushort * dst_row = (global ushort *)(dst  +  i1*nb1  + i02*nb2  + i03*nb3);  \
    for (int ind = get_local_id(0); ind < nblk0; ind += get_local_size(0)) {   \
        dst_row[ind] = float_to_bf16(src_row[ind]);                            \
    }                                                                          \
}

KERNEL_SET_ROWS_BF16_IMPL(kernel_set_rows_bf16_i64, long)
KERNEL_SET_ROWS_BF16_IMPL(kernel_set_rows_bf16_i32, int)
