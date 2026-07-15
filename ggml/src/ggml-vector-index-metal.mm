#include "ggml-vector-index-metal.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>

namespace {

constexpr char kMetalSource[] =
R"metal(
#include <metal_stdlib>
using namespace metal;

kernel void ggml_vec_index_dot_f32(
        device const float * vectors [[buffer(0)]],
        device const uchar * active  [[buffer(1)]],
        device const float * queries [[buffer(2)]],
        device       float * scores  [[buffer(3)]],
        constant uint      & dim     [[buffer(4)]],
        constant uint      & n_slots [[buffer(5)]],
        uint gid [[thread_position_in_grid]]) {
    const uint slot = gid % n_slots;
    const uint q    = gid / n_slots;

    if (active[slot] == 0) {
        scores[gid] = -3.4028234663852886e38f;
        return;
    }

    float acc = 0.0f;
    const device float * v = vectors + (uint64_t) slot * dim;
    const device float * x = queries + (uint64_t) q * dim;
    for (uint i = 0; i < dim; ++i) {
        acc += x[i] * v[i];
    }
    scores[gid] = acc;
}

kernel void ggml_vec_index_topk_f32(
        device const float * vectors    [[buffer(0)]],
        device const uchar * active     [[buffer(1)]],
        device const float * queries    [[buffer(2)]],
        device       float * scores_out [[buffer(3)]],
        device       uint  * slots_out  [[buffer(4)]],
        constant uint      & dim        [[buffer(5)]],
        constant uint      & n_slots    [[buffer(6)]],
        constant uint      & k          [[buffer(7)]],
        constant uint      & block_size [[buffer(8)]],
        constant uint      & n_blocks   [[buffer(9)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]]) {
    constexpr uint max_k = 64;
    constexpr uint max_block = 256;
    threadgroup float block_scores[max_block];
    threadgroup uint block_slots[max_block];

    const uint gid = tgpig.x;
    const uint block = gid % n_blocks;
    const uint q = gid / n_blocks;
    const uint begin = block * block_size;
    const uint slot = begin + tid;

    if (tid < max_block) {
        block_scores[tid] = -3.4028234663852886e38f;
        block_slots[tid] = 0xffffffffu;
    }

    if (tid < block_size && slot < n_slots && active[slot] != 0) {
        const device float * x = queries + (uint64_t) q * dim;
        const device float * v = vectors + (uint64_t) slot * dim;
        float acc = 0.0f;
        for (uint i = 0; i < dim; ++i) {
            acc += x[i] * v[i];
        }
        block_scores[tid] = acc;
        block_slots[tid] = slot;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) {
        return;
    }

    float best_scores[max_k];
    uint best_slots[max_k];

    for (uint i = 0; i < k; ++i) {
        best_scores[i] = -3.4028234663852886e38f;
        best_slots[i] = 0xffffffffu;
    }

    for (uint i = 0; i < block_size; ++i) {
        const float acc = block_scores[i];
        const uint candidate_slot = block_slots[i];
        if (candidate_slot == 0xffffffffu || acc <= best_scores[k - 1]) {
            continue;
        }

        uint pos = k - 1;
        while (pos > 0 && acc > best_scores[pos - 1]) {
            best_scores[pos] = best_scores[pos - 1];
            best_slots[pos] = best_slots[pos - 1];
            --pos;
        }
        best_scores[pos] = acc;
        best_slots[pos] = candidate_slot;
    }

    const uint out_base = gid * k;
    for (uint i = 0; i < k; ++i) {
        scores_out[out_base + i] = best_scores[i];
        slots_out[out_base + i] = best_slots[i];
    }
}

kernel void ggml_vec_index_topk_q8(
        device const char  * codes      [[buffer(0)]],
        device const float * scales     [[buffer(1)]],
        device const uchar * active     [[buffer(2)]],
        device const float * queries    [[buffer(3)]],
        device       float * scores_out [[buffer(4)]],
        device       uint  * slots_out  [[buffer(5)]],
        constant uint      & dim        [[buffer(6)]],
        constant uint      & n_slots    [[buffer(7)]],
        constant uint      & k          [[buffer(8)]],
        constant uint      & block_size [[buffer(9)]],
        constant uint      & n_blocks   [[buffer(10)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]]) {
    constexpr uint max_k = 64;
    constexpr uint max_block = 256;
    threadgroup float block_scores[max_block];
    threadgroup uint block_slots[max_block];

    const uint gid = tgpig.x;
    const uint block = gid % n_blocks;
    const uint q = gid / n_blocks;
    const uint begin = block * block_size;
    const uint slot = begin + tid;

    if (tid < max_block) {
        block_scores[tid] = -3.4028234663852886e38f;
        block_slots[tid] = 0xffffffffu;
    }

    if (tid < block_size && slot < n_slots && active[slot] != 0) {
        const device float * x = queries + (uint64_t) q * dim;
        const device char * c = codes + (uint64_t) slot * dim;
        const float scale = scales[slot];
        float acc = 0.0f;
        for (uint i = 0; i < dim; ++i) {
            acc += x[i] * (static_cast<float>(c[i]) * scale);
        }
        block_scores[tid] = acc;
        block_slots[tid] = slot;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) {
        return;
    }

    float best_scores[max_k];
    uint best_slots[max_k];

    for (uint i = 0; i < k; ++i) {
        best_scores[i] = -3.4028234663852886e38f;
        best_slots[i] = 0xffffffffu;
    }

    for (uint i = 0; i < block_size; ++i) {
        const float acc = block_scores[i];
        const uint candidate_slot = block_slots[i];
        if (candidate_slot == 0xffffffffu || acc <= best_scores[k - 1]) {
            continue;
        }

        uint pos = k - 1;
        while (pos > 0 && acc > best_scores[pos - 1]) {
            best_scores[pos] = best_scores[pos - 1];
            best_slots[pos] = best_slots[pos - 1];
            --pos;
        }
        best_scores[pos] = acc;
        best_slots[pos] = candidate_slot;
    }

    const uint out_base = gid * k;
    for (uint i = 0; i < k; ++i) {
        scores_out[out_base + i] = best_scores[i];
        slots_out[out_base + i] = best_slots[i];
    }
}

kernel void ggml_vec_index_topk_q4(
        device const uchar * codes      [[buffer(0)]],
        device const float * scales     [[buffer(1)]],
        device const uchar * active     [[buffer(2)]],
        device const float * queries    [[buffer(3)]],
        device       float * scores_out [[buffer(4)]],
        device       uint  * slots_out  [[buffer(5)]],
        constant uint      & dim        [[buffer(6)]],
        constant uint      & row_bytes  [[buffer(7)]],
        constant uint      & n_slots    [[buffer(8)]],
        constant uint      & k          [[buffer(9)]],
        constant uint      & block_size [[buffer(10)]],
        constant uint      & n_blocks   [[buffer(11)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]]) {
    constexpr uint max_k = 64;
    constexpr uint max_block = 256;
    threadgroup float block_scores[max_block];
    threadgroup uint block_slots[max_block];

    const uint gid = tgpig.x;
    const uint block = gid % n_blocks;
    const uint q = gid / n_blocks;
    const uint begin = block * block_size;
    const uint slot = begin + tid;

    if (tid < max_block) {
        block_scores[tid] = -3.4028234663852886e38f;
        block_slots[tid] = 0xffffffffu;
    }

    if (tid < block_size && slot < n_slots && active[slot] != 0) {
        const device float * x = queries + (uint64_t) q * dim;
        const device uchar * c = codes + (uint64_t) slot * row_bytes;
        const float scale = scales[slot];
        float acc = 0.0f;
        for (uint i = 0; i < dim; ++i) {
            const uchar byte = c[i / 2];
            const uchar nibble = (i & 1u) == 0u ? (byte & 0x0fu) : (byte >> 4);
            acc += x[i] * (static_cast<float>(static_cast<int>(nibble) - 8) * scale);
        }
        block_scores[tid] = acc;
        block_slots[tid] = slot;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) {
        return;
    }

    float best_scores[max_k];
    uint best_slots[max_k];

    for (uint i = 0; i < k; ++i) {
        best_scores[i] = -3.4028234663852886e38f;
        best_slots[i] = 0xffffffffu;
    }

    for (uint i = 0; i < block_size; ++i) {
        const float acc = block_scores[i];
        const uint candidate_slot = block_slots[i];
        if (candidate_slot == 0xffffffffu || acc <= best_scores[k - 1]) {
            continue;
        }

        uint pos = k - 1;
        while (pos > 0 && acc > best_scores[pos - 1]) {
            best_scores[pos] = best_scores[pos - 1];
            best_slots[pos] = best_slots[pos - 1];
            --pos;
        }
        best_scores[pos] = acc;
        best_slots[pos] = candidate_slot;
    }

    const uint out_base = gid * k;
    for (uint i = 0; i < k; ++i) {
        scores_out[out_base + i] = best_scores[i];
        slots_out[out_base + i] = best_slots[i];
    }
}

kernel void ggml_vec_index_topk_filter_f32(
        device const float * vectors      [[buffer(0)]],
        device const uchar * active       [[buffer(1)]],
        device const float * queries      [[buffer(2)]],
        device const uint  * filter_slots [[buffer(3)]],
        device       float * scores_out   [[buffer(4)]],
        device       uint  * slots_out    [[buffer(5)]],
        constant uint      & dim          [[buffer(6)]],
        constant uint      & n_filter     [[buffer(7)]],
        constant uint      & k            [[buffer(8)]],
        constant uint      & block_size   [[buffer(9)]],
        constant uint      & n_blocks     [[buffer(10)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]]) {
    constexpr uint max_k = 64;
    constexpr uint max_block = 256;
    threadgroup float block_scores[max_block];
    threadgroup uint block_slots[max_block];

    const uint gid = tgpig.x;
    const uint block = gid % n_blocks;
    const uint q = gid / n_blocks;
    const uint filter_pos = block * block_size + tid;

    if (tid < max_block) {
        block_scores[tid] = -3.4028234663852886e38f;
        block_slots[tid] = 0xffffffffu;
    }

    if (tid < block_size && filter_pos < n_filter) {
        const uint slot = filter_slots[filter_pos];
        if (active[slot] != 0) {
            const device float * x = queries + (uint64_t) q * dim;
            const device float * v = vectors + (uint64_t) slot * dim;
            float acc = 0.0f;
            for (uint i = 0; i < dim; ++i) {
                acc += x[i] * v[i];
            }
            block_scores[tid] = acc;
            block_slots[tid] = slot;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) {
        return;
    }

    float best_scores[max_k];
    uint best_slots[max_k];

    for (uint i = 0; i < k; ++i) {
        best_scores[i] = -3.4028234663852886e38f;
        best_slots[i] = 0xffffffffu;
    }

    for (uint i = 0; i < block_size; ++i) {
        const float acc = block_scores[i];
        const uint candidate_slot = block_slots[i];
        if (candidate_slot == 0xffffffffu || acc <= best_scores[k - 1]) {
            continue;
        }

        uint pos = k - 1;
        while (pos > 0 && acc > best_scores[pos - 1]) {
            best_scores[pos] = best_scores[pos - 1];
            best_slots[pos] = best_slots[pos - 1];
            --pos;
        }
        best_scores[pos] = acc;
        best_slots[pos] = candidate_slot;
    }

    const uint out_base = gid * k;
    for (uint i = 0; i < k; ++i) {
        scores_out[out_base + i] = best_scores[i];
        slots_out[out_base + i] = best_slots[i];
    }
}

kernel void ggml_vec_index_topk_filter_q8(
        device const char  * codes        [[buffer(0)]],
        device const float * scales       [[buffer(1)]],
        device const uchar * active       [[buffer(2)]],
        device const float * queries      [[buffer(3)]],
        device const uint  * filter_slots [[buffer(4)]],
        device       float * scores_out   [[buffer(5)]],
        device       uint  * slots_out    [[buffer(6)]],
        constant uint      & dim          [[buffer(7)]],
        constant uint      & n_filter     [[buffer(8)]],
        constant uint      & k            [[buffer(9)]],
        constant uint      & block_size   [[buffer(10)]],
        constant uint      & n_blocks     [[buffer(11)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]]) {
    constexpr uint max_k = 64;
    constexpr uint max_block = 256;
    threadgroup float block_scores[max_block];
    threadgroup uint block_slots[max_block];

    const uint gid = tgpig.x;
    const uint block = gid % n_blocks;
    const uint q = gid / n_blocks;
    const uint filter_pos = block * block_size + tid;

    if (tid < max_block) {
        block_scores[tid] = -3.4028234663852886e38f;
        block_slots[tid] = 0xffffffffu;
    }

    if (tid < block_size && filter_pos < n_filter) {
        const uint slot = filter_slots[filter_pos];
        if (active[slot] != 0) {
            const device float * x = queries + (uint64_t) q * dim;
            const device char * c = codes + (uint64_t) slot * dim;
            const float scale = scales[slot];
            float acc = 0.0f;
            for (uint i = 0; i < dim; ++i) {
                acc += x[i] * (static_cast<float>(c[i]) * scale);
            }
            block_scores[tid] = acc;
            block_slots[tid] = slot;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) {
        return;
    }

    float best_scores[max_k];
    uint best_slots[max_k];

    for (uint i = 0; i < k; ++i) {
        best_scores[i] = -3.4028234663852886e38f;
        best_slots[i] = 0xffffffffu;
    }

    for (uint i = 0; i < block_size; ++i) {
        const float acc = block_scores[i];
        const uint candidate_slot = block_slots[i];
        if (candidate_slot == 0xffffffffu || acc <= best_scores[k - 1]) {
            continue;
        }

        uint pos = k - 1;
        while (pos > 0 && acc > best_scores[pos - 1]) {
            best_scores[pos] = best_scores[pos - 1];
            best_slots[pos] = best_slots[pos - 1];
            --pos;
        }
        best_scores[pos] = acc;
        best_slots[pos] = candidate_slot;
    }

    const uint out_base = gid * k;
    for (uint i = 0; i < k; ++i) {
        scores_out[out_base + i] = best_scores[i];
        slots_out[out_base + i] = best_slots[i];
    }
}

kernel void ggml_vec_index_topk_filter_q4(
        device const uchar * codes        [[buffer(0)]],
        device const float * scales       [[buffer(1)]],
        device const uchar * active       [[buffer(2)]],
        device const float * queries      [[buffer(3)]],
        device const uint  * filter_slots [[buffer(4)]],
        device       float * scores_out   [[buffer(5)]],
        device       uint  * slots_out    [[buffer(6)]],
        constant uint      & dim          [[buffer(7)]],
        constant uint      & row_bytes    [[buffer(8)]],
        constant uint      & n_filter     [[buffer(9)]],
        constant uint      & k            [[buffer(10)]],
        constant uint      & block_size   [[buffer(11)]],
        constant uint      & n_blocks     [[buffer(12)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]]) {
    constexpr uint max_k = 64;
    constexpr uint max_block = 256;
    threadgroup float block_scores[max_block];
    threadgroup uint block_slots[max_block];

    const uint gid = tgpig.x;
    const uint block = gid % n_blocks;
    const uint q = gid / n_blocks;
    const uint filter_pos = block * block_size + tid;

    if (tid < max_block) {
        block_scores[tid] = -3.4028234663852886e38f;
        block_slots[tid] = 0xffffffffu;
    }

    if (tid < block_size && filter_pos < n_filter) {
        const uint slot = filter_slots[filter_pos];
        if (active[slot] != 0) {
            const device float * x = queries + (uint64_t) q * dim;
            const device uchar * c = codes + (uint64_t) slot * row_bytes;
            const float scale = scales[slot];
            float acc = 0.0f;
            for (uint i = 0; i < dim; ++i) {
                const uchar byte = c[i / 2];
                const uchar nibble = (i & 1u) == 0u ? (byte & 0x0fu) : (byte >> 4);
                acc += x[i] * (static_cast<float>(static_cast<int>(nibble) - 8) * scale);
            }
            block_scores[tid] = acc;
            block_slots[tid] = slot;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) {
        return;
    }

    float best_scores[max_k];
    uint best_slots[max_k];

    for (uint i = 0; i < k; ++i) {
        best_scores[i] = -3.4028234663852886e38f;
        best_slots[i] = 0xffffffffu;
    }

    for (uint i = 0; i < block_size; ++i) {
        const float acc = block_scores[i];
        const uint candidate_slot = block_slots[i];
        if (candidate_slot == 0xffffffffu || acc <= best_scores[k - 1]) {
            continue;
        }

        uint pos = k - 1;
        while (pos > 0 && acc > best_scores[pos - 1]) {
            best_scores[pos] = best_scores[pos - 1];
            best_slots[pos] = best_slots[pos - 1];
            --pos;
        }
        best_scores[pos] = acc;
        best_slots[pos] = candidate_slot;
    }

    const uint out_base = gid * k;
    for (uint i = 0; i < k; ++i) {
        scores_out[out_base + i] = best_scores[i];
        slots_out[out_base + i] = best_slots[i];
    }
}

kernel void ggml_vec_index_topk_filter_blocks_f32(
        device const float * vectors       [[buffer(0)]],
        device const uchar * active        [[buffer(1)]],
        device const float * queries       [[buffer(2)]],
        device const uint  * filter_slots  [[buffer(3)]],
        device const uint  * block_queries [[buffer(4)]],
        device const uint  * block_offsets [[buffer(5)]],
        device const uint  * block_counts  [[buffer(6)]],
        device       float * scores_out    [[buffer(7)]],
        device       uint  * slots_out     [[buffer(8)]],
        constant uint      & dim           [[buffer(9)]],
        constant uint      & k             [[buffer(10)]],
        constant uint      & block_size    [[buffer(11)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]]) {
    constexpr uint max_k = 64;
    constexpr uint max_block = 256;
    threadgroup float block_scores[max_block];
    threadgroup uint block_slots[max_block];

    const uint gid = tgpig.x;
    const uint q = block_queries[gid];
    const uint offset = block_offsets[gid];
    const uint count = block_counts[gid];
    const uint filter_pos = offset + tid;

    if (tid < max_block) {
        block_scores[tid] = -3.4028234663852886e38f;
        block_slots[tid] = 0xffffffffu;
    }

    if (tid < block_size && tid < count) {
        const uint slot = filter_slots[filter_pos];
        if (active[slot] != 0) {
            const device float * x = queries + (uint64_t) q * dim;
            const device float * v = vectors + (uint64_t) slot * dim;
            float acc = 0.0f;
            for (uint i = 0; i < dim; ++i) {
                acc += x[i] * v[i];
            }
            block_scores[tid] = acc;
            block_slots[tid] = slot;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) {
        return;
    }

    float best_scores[max_k];
    uint best_slots[max_k];

    for (uint i = 0; i < k; ++i) {
        best_scores[i] = -3.4028234663852886e38f;
        best_slots[i] = 0xffffffffu;
    }

    for (uint i = 0; i < block_size; ++i) {
        const float acc = block_scores[i];
        const uint candidate_slot = block_slots[i];
        if (candidate_slot == 0xffffffffu || acc <= best_scores[k - 1]) {
            continue;
        }

        uint pos = k - 1;
        while (pos > 0 && acc > best_scores[pos - 1]) {
            best_scores[pos] = best_scores[pos - 1];
            best_slots[pos] = best_slots[pos - 1];
            --pos;
        }
        best_scores[pos] = acc;
        best_slots[pos] = candidate_slot;
    }

    const uint out_base = gid * k;
    for (uint i = 0; i < k; ++i) {
        scores_out[out_base + i] = best_scores[i];
        slots_out[out_base + i] = best_slots[i];
    }
}

kernel void ggml_vec_index_topk_filter_blocks_q8(
        device const char  * codes         [[buffer(0)]],
        device const float * scales        [[buffer(1)]],
        device const uchar * active        [[buffer(2)]],
        device const float * queries       [[buffer(3)]],
        device const uint  * filter_slots  [[buffer(4)]],
        device const uint  * block_queries [[buffer(5)]],
        device const uint  * block_offsets [[buffer(6)]],
        device const uint  * block_counts  [[buffer(7)]],
        device       float * scores_out    [[buffer(8)]],
        device       uint  * slots_out     [[buffer(9)]],
        constant uint      & dim           [[buffer(10)]],
        constant uint      & k             [[buffer(11)]],
        constant uint      & block_size    [[buffer(12)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]]) {
    constexpr uint max_k = 64;
    constexpr uint max_block = 256;
    threadgroup float block_scores[max_block];
    threadgroup uint block_slots[max_block];

    const uint gid = tgpig.x;
    const uint q = block_queries[gid];
    const uint offset = block_offsets[gid];
    const uint count = block_counts[gid];
    const uint filter_pos = offset + tid;

    if (tid < max_block) {
        block_scores[tid] = -3.4028234663852886e38f;
        block_slots[tid] = 0xffffffffu;
    }

    if (tid < block_size && tid < count) {
        const uint slot = filter_slots[filter_pos];
        if (active[slot] != 0) {
            const device float * x = queries + (uint64_t) q * dim;
            const device char * c = codes + (uint64_t) slot * dim;
            const float scale = scales[slot];
            float acc = 0.0f;
            for (uint i = 0; i < dim; ++i) {
                acc += x[i] * (static_cast<float>(c[i]) * scale);
            }
            block_scores[tid] = acc;
            block_slots[tid] = slot;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) {
        return;
    }

    float best_scores[max_k];
    uint best_slots[max_k];

    for (uint i = 0; i < k; ++i) {
        best_scores[i] = -3.4028234663852886e38f;
        best_slots[i] = 0xffffffffu;
    }

    for (uint i = 0; i < block_size; ++i) {
        const float acc = block_scores[i];
        const uint candidate_slot = block_slots[i];
        if (candidate_slot == 0xffffffffu || acc <= best_scores[k - 1]) {
            continue;
        }

        uint pos = k - 1;
        while (pos > 0 && acc > best_scores[pos - 1]) {
            best_scores[pos] = best_scores[pos - 1];
            best_slots[pos] = best_slots[pos - 1];
            --pos;
        }
        best_scores[pos] = acc;
        best_slots[pos] = candidate_slot;
    }

    const uint out_base = gid * k;
    for (uint i = 0; i < k; ++i) {
        scores_out[out_base + i] = best_scores[i];
        slots_out[out_base + i] = best_slots[i];
    }
}

kernel void ggml_vec_index_topk_filter_blocks_q4(
        device const uchar * codes         [[buffer(0)]],
        device const float * scales        [[buffer(1)]],
        device const uchar * active        [[buffer(2)]],
        device const float * queries       [[buffer(3)]],
        device const uint  * filter_slots  [[buffer(4)]],
        device const uint  * block_queries [[buffer(5)]],
        device const uint  * block_offsets [[buffer(6)]],
        device const uint  * block_counts  [[buffer(7)]],
        device       float * scores_out    [[buffer(8)]],
        device       uint  * slots_out     [[buffer(9)]],
        constant uint      & dim           [[buffer(10)]],
        constant uint      & row_bytes     [[buffer(11)]],
        constant uint      & k             [[buffer(12)]],
        constant uint      & block_size    [[buffer(13)]],
        uint3 tgpig [[threadgroup_position_in_grid]],
        uint  tid   [[thread_index_in_threadgroup]]) {
    constexpr uint max_k = 64;
    constexpr uint max_block = 256;
    threadgroup float block_scores[max_block];
    threadgroup uint block_slots[max_block];

    const uint gid = tgpig.x;
    const uint q = block_queries[gid];
    const uint offset = block_offsets[gid];
    const uint count = block_counts[gid];
    const uint filter_pos = offset + tid;

    if (tid < max_block) {
        block_scores[tid] = -3.4028234663852886e38f;
        block_slots[tid] = 0xffffffffu;
    }

    if (tid < block_size && tid < count) {
        const uint slot = filter_slots[filter_pos];
        if (active[slot] != 0) {
            const device float * x = queries + (uint64_t) q * dim;
            const device uchar * c = codes + (uint64_t) slot * row_bytes;
            const float scale = scales[slot];
            float acc = 0.0f;
            for (uint i = 0; i < dim; ++i) {
                const uchar byte = c[i / 2];
                const uchar nibble = (i & 1u) == 0u ? (byte & 0x0fu) : (byte >> 4);
                acc += x[i] * (static_cast<float>(static_cast<int>(nibble) - 8) * scale);
            }
            block_scores[tid] = acc;
            block_slots[tid] = slot;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tid != 0) {
        return;
    }

    float best_scores[max_k];
    uint best_slots[max_k];

    for (uint i = 0; i < k; ++i) {
        best_scores[i] = -3.4028234663852886e38f;
        best_slots[i] = 0xffffffffu;
    }

    for (uint i = 0; i < block_size; ++i) {
        const float acc = block_scores[i];
        const uint candidate_slot = block_slots[i];
        if (candidate_slot == 0xffffffffu || acc <= best_scores[k - 1]) {
            continue;
        }

        uint pos = k - 1;
        while (pos > 0 && acc > best_scores[pos - 1]) {
            best_scores[pos] = best_scores[pos - 1];
            best_slots[pos] = best_slots[pos - 1];
            --pos;
        }
        best_scores[pos] = acc;
        best_slots[pos] = candidate_slot;
    }

    const uint out_base = gid * k;
    for (uint i = 0; i < k; ++i) {
        scores_out[out_base + i] = best_scores[i];
        slots_out[out_base + i] = best_slots[i];
    }
}
)metal";

struct MetalState {
    id<MTLDevice> device = nil;
    id<MTLCommandQueue> queue = nil;
    id<MTLComputePipelineState> pipeline = nil;
    id<MTLComputePipelineState> topk_pipeline = nil;
    id<MTLComputePipelineState> topk_q8_pipeline = nil;
    id<MTLComputePipelineState> topk_q4_pipeline = nil;
    id<MTLComputePipelineState> topk_filter_pipeline = nil;
    id<MTLComputePipelineState> topk_filter_q8_pipeline = nil;
    id<MTLComputePipelineState> topk_filter_q4_pipeline = nil;
    id<MTLComputePipelineState> topk_filter_blocks_pipeline = nil;
    id<MTLComputePipelineState> topk_filter_blocks_q8_pipeline = nil;
    id<MTLComputePipelineState> topk_filter_blocks_q4_pipeline = nil;
    bool initialized = false;
};

std::mutex & metal_mutex() {
    static std::mutex mutex;
    return mutex;
}

MetalState & metal_state() {
    static MetalState state;
    return state;
}

bool init_metal_unlocked(MetalState & state) {
    if (state.initialized) {
        return state.pipeline != nil;
    }
    state.initialized = true;

    @autoreleasepool {
        state.device = MTLCreateSystemDefaultDevice();
        if (state.device == nil) {
            return false;
        }

        NSError * error = nil;
        NSString * source =
            [[NSString alloc] initWithBytes:kMetalSource
                                     length:sizeof(kMetalSource) - 1
                                   encoding:NSUTF8StringEncoding];
        id<MTLLibrary> library = [state.device newLibraryWithSource:source options:nil error:&error];
        [source release];
        if (library == nil || error != nil) {
            [library release];
            return false;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"ggml_vec_index_dot_f32"];
        if (function == nil) {
            [library release];
            return false;
        }

        state.pipeline = [state.device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        if (state.pipeline == nil || error != nil) {
            [library release];
            return false;
        }

        [library release];

        state.queue = [state.device newCommandQueue];
        if (state.queue == nil) {
            [state.pipeline release];
            state.pipeline = nil;
            [state.topk_pipeline release];
            state.topk_pipeline = nil;
            [state.topk_q8_pipeline release];
            state.topk_q8_pipeline = nil;
            [state.topk_q4_pipeline release];
            state.topk_q4_pipeline = nil;
            [state.topk_filter_pipeline release];
            state.topk_filter_pipeline = nil;
            [state.topk_filter_q8_pipeline release];
            state.topk_filter_q8_pipeline = nil;
            [state.topk_filter_q4_pipeline release];
            state.topk_filter_q4_pipeline = nil;
            [state.topk_filter_blocks_pipeline release];
            state.topk_filter_blocks_pipeline = nil;
            [state.topk_filter_blocks_q8_pipeline release];
            state.topk_filter_blocks_q8_pipeline = nil;
            [state.topk_filter_blocks_q4_pipeline release];
            state.topk_filter_blocks_q4_pipeline = nil;
            return false;
        }
    }

    return true;
}

bool init_topk_unlocked(MetalState & state) {
    if (!init_metal_unlocked(state)) {
        return false;
    }
    if (state.topk_pipeline != nil) {
        return true;
    }

    @autoreleasepool {
        NSError * error = nil;
        NSString * source =
            [[NSString alloc] initWithBytes:kMetalSource
                                     length:sizeof(kMetalSource) - 1
                                   encoding:NSUTF8StringEncoding];
        id<MTLLibrary> library = [state.device newLibraryWithSource:source options:nil error:&error];
        [source release];
        if (library == nil || error != nil) {
            [library release];
            return false;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"ggml_vec_index_topk_f32"];
        if (function == nil) {
            [library release];
            return false;
        }

        state.topk_pipeline = [state.device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        [library release];
        return state.topk_pipeline != nil && error == nil;
    }
}

bool init_topk_q8_unlocked(MetalState & state) {
    if (!init_metal_unlocked(state)) {
        return false;
    }
    if (state.topk_q8_pipeline != nil) {
        return true;
    }

    @autoreleasepool {
        NSError * error = nil;
        NSString * source =
            [[NSString alloc] initWithBytes:kMetalSource
                                     length:sizeof(kMetalSource) - 1
                                   encoding:NSUTF8StringEncoding];
        id<MTLLibrary> library = [state.device newLibraryWithSource:source options:nil error:&error];
        [source release];
        if (library == nil || error != nil) {
            [library release];
            return false;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"ggml_vec_index_topk_q8"];
        if (function == nil) {
            [library release];
            return false;
        }

        state.topk_q8_pipeline = [state.device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        [library release];
        return state.topk_q8_pipeline != nil && error == nil;
    }
}

bool init_topk_q4_unlocked(MetalState & state) {
    if (!init_metal_unlocked(state)) {
        return false;
    }
    if (state.topk_q4_pipeline != nil) {
        return true;
    }

    @autoreleasepool {
        NSError * error = nil;
        NSString * source =
            [[NSString alloc] initWithBytes:kMetalSource
                                     length:sizeof(kMetalSource) - 1
                                   encoding:NSUTF8StringEncoding];
        id<MTLLibrary> library = [state.device newLibraryWithSource:source options:nil error:&error];
        [source release];
        if (library == nil || error != nil) {
            [library release];
            return false;
        }

        id<MTLFunction> function = [library newFunctionWithName:@"ggml_vec_index_topk_q4"];
        if (function == nil) {
            [library release];
            return false;
        }

        state.topk_q4_pipeline = [state.device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        [library release];
        return state.topk_q4_pipeline != nil && error == nil;
    }
}

bool init_named_pipeline_unlocked(
        MetalState & state,
        NSString * function_name,
        id<MTLComputePipelineState> & pipeline) {
    if (!init_metal_unlocked(state)) {
        return false;
    }
    if (pipeline != nil) {
        return true;
    }

    @autoreleasepool {
        NSError * error = nil;
        NSString * source =
            [[NSString alloc] initWithBytes:kMetalSource
                                     length:sizeof(kMetalSource) - 1
                                   encoding:NSUTF8StringEncoding];
        id<MTLLibrary> library = [state.device newLibraryWithSource:source options:nil error:&error];
        [source release];
        if (library == nil || error != nil) {
            [library release];
            return false;
        }

        id<MTLFunction> function = [library newFunctionWithName:function_name];
        if (function == nil) {
            [library release];
            return false;
        }

        pipeline = [state.device newComputePipelineStateWithFunction:function error:&error];
        [function release];
        [library release];
        return pipeline != nil && error == nil;
    }
}

bool init_topk_filter_unlocked(MetalState & state) {
    return init_named_pipeline_unlocked(
        state, @"ggml_vec_index_topk_filter_f32", state.topk_filter_pipeline);
}

bool init_topk_filter_q8_unlocked(MetalState & state) {
    return init_named_pipeline_unlocked(
        state, @"ggml_vec_index_topk_filter_q8", state.topk_filter_q8_pipeline);
}

bool init_topk_filter_q4_unlocked(MetalState & state) {
    return init_named_pipeline_unlocked(
        state, @"ggml_vec_index_topk_filter_q4", state.topk_filter_q4_pipeline);
}

bool init_topk_filter_blocks_unlocked(MetalState & state) {
    return init_named_pipeline_unlocked(
        state, @"ggml_vec_index_topk_filter_blocks_f32", state.topk_filter_blocks_pipeline);
}

bool init_topk_filter_blocks_q8_unlocked(MetalState & state) {
    return init_named_pipeline_unlocked(
        state, @"ggml_vec_index_topk_filter_blocks_q8", state.topk_filter_blocks_q8_pipeline);
}

bool init_topk_filter_blocks_q4_unlocked(MetalState & state) {
    return init_named_pipeline_unlocked(
        state, @"ggml_vec_index_topk_filter_blocks_q4", state.topk_filter_blocks_q4_pipeline);
}

} // namespace

struct ggml_vec_index_metal_index {
    id<MTLBuffer> vectors = nil;
    id<MTLBuffer> q8_codes = nil;
    id<MTLBuffer> q8_scales = nil;
    id<MTLBuffer> q4_codes = nil;
    id<MTLBuffer> q4_scales = nil;
    id<MTLBuffer> active = nil;
    size_t n_slots = 0;
    int dim = 0;
};

bool ggml_vec_index_metal_available(void) {
    std::lock_guard<std::mutex> lock(metal_mutex());
    return init_metal_unlocked(metal_state());
}

void ggml_vec_index_metal_free(ggml_vec_index_metal_index * cache) {
    if (cache == nullptr) {
        return;
    }
    [cache->vectors release];
    [cache->q8_codes release];
    [cache->q8_scales release];
    [cache->q4_codes release];
    [cache->q4_scales release];
    [cache->active release];
    delete cache;
}

int ggml_vec_index_metal_prepare_f32(
        ggml_vec_index_metal_index ** cache,
        const float   * vectors,
        const uint8_t * active,
        size_t          n_slots,
        int             dim) {
    if (cache == nullptr || vectors == nullptr || active == nullptr ||
        n_slots == 0 || dim <= 0 ||
        n_slots > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_metal_unlocked(state)) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(dim);
        if (dim_sz != 0 && n_slots > SIZE_MAX / dim_sz / sizeof(float)) {
            return -2;
        }

        ggml_vec_index_metal_index * next = new ggml_vec_index_metal_index;

        id<MTLBuffer> vectors_buffer =
            [state.device newBufferWithBytes:vectors
                                      length:n_slots * dim_sz * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> active_buffer =
            [state.device newBufferWithBytes:active
                                      length:n_slots * sizeof(uint8_t)
                                     options:MTLResourceStorageModeShared];
        if (vectors_buffer == nil || active_buffer == nil) {
            [vectors_buffer release];
            [active_buffer release];
            delete next;
            return -7;
        }

        next->vectors = vectors_buffer;
        next->active = active_buffer;
        next->n_slots = n_slots;
        next->dim = dim;

        ggml_vec_index_metal_free(*cache);
        *cache = next;
    }

    return 0;
}

int ggml_vec_index_metal_prepare_q8(
        ggml_vec_index_metal_index ** cache,
        const int8_t  * codes,
        const float   * scales,
        const uint8_t * active,
        size_t          n_slots,
        int             dim) {
    if (cache == nullptr || codes == nullptr || scales == nullptr || active == nullptr ||
        n_slots == 0 || dim <= 0 ||
        n_slots > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_metal_unlocked(state)) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(dim);
        if (dim_sz != 0 && n_slots > SIZE_MAX / dim_sz / sizeof(int8_t)) {
            return -2;
        }

        ggml_vec_index_metal_index * next = new ggml_vec_index_metal_index;

        id<MTLBuffer> codes_buffer =
            [state.device newBufferWithBytes:codes
                                      length:n_slots * dim_sz * sizeof(int8_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scales_buffer =
            [state.device newBufferWithBytes:scales
                                      length:n_slots * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> active_buffer =
            [state.device newBufferWithBytes:active
                                      length:n_slots * sizeof(uint8_t)
                                     options:MTLResourceStorageModeShared];
        if (codes_buffer == nil || scales_buffer == nil || active_buffer == nil) {
            [codes_buffer release];
            [scales_buffer release];
            [active_buffer release];
            delete next;
            return -7;
        }

        next->q8_codes = codes_buffer;
        next->q8_scales = scales_buffer;
        next->active = active_buffer;
        next->n_slots = n_slots;
        next->dim = dim;

        ggml_vec_index_metal_free(*cache);
        *cache = next;
    }

    return 0;
}

int ggml_vec_index_metal_prepare_q4(
        ggml_vec_index_metal_index ** cache,
        const uint8_t * codes,
        const float   * scales,
        const uint8_t * active,
        size_t          n_slots,
        int             dim) {
    if (cache == nullptr || codes == nullptr || scales == nullptr || active == nullptr ||
        n_slots == 0 || dim <= 0 ||
        n_slots > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_metal_unlocked(state)) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(dim);
        const size_t row_bytes = (dim_sz + 1) / 2;
        if (row_bytes != 0 && n_slots > SIZE_MAX / row_bytes) {
            return -2;
        }

        ggml_vec_index_metal_index * next = new ggml_vec_index_metal_index;

        id<MTLBuffer> codes_buffer =
            [state.device newBufferWithBytes:codes
                                      length:n_slots * row_bytes
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scales_buffer =
            [state.device newBufferWithBytes:scales
                                      length:n_slots * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> active_buffer =
            [state.device newBufferWithBytes:active
                                      length:n_slots * sizeof(uint8_t)
                                     options:MTLResourceStorageModeShared];
        if (codes_buffer == nil || scales_buffer == nil || active_buffer == nil) {
            [codes_buffer release];
            [scales_buffer release];
            [active_buffer release];
            delete next;
            return -7;
        }

        next->q4_codes = codes_buffer;
        next->q4_scales = scales_buffer;
        next->active = active_buffer;
        next->n_slots = n_slots;
        next->dim = dim;

        ggml_vec_index_metal_free(*cache);
        *cache = next;
    }

    return 0;
}

int ggml_vec_index_metal_score_f32(
        ggml_vec_index_metal_index * cache,
        const float   * queries,
        int             n_q,
        float         * scores) {
    if (cache == nullptr || queries == nullptr || scores == nullptr || n_q <= 0 ||
        cache->vectors == nil || cache->active == nil || cache->n_slots == 0 || cache->dim <= 0) {
        return -2;
    }

    const size_t n_slots = cache->n_slots;
    const size_t n_q_sz = static_cast<size_t>(n_q);
    if (n_slots > SIZE_MAX / n_q_sz) {
        return -2;
    }
    const size_t total = n_slots * n_q_sz;
    if (total > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_metal_unlocked(state)) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(cache->dim);
        if (dim_sz != 0 && n_q_sz > SIZE_MAX / dim_sz / sizeof(float)) {
            return -2;
        }
        if (total > SIZE_MAX / sizeof(float)) {
            return -2;
        }

        id<MTLBuffer> queries_buffer =
            [state.device newBufferWithBytes:queries
                                      length:n_q_sz * dim_sz * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buffer =
            [state.device newBufferWithLength:total * sizeof(float)
                                      options:MTLResourceStorageModeShared];

        if (queries_buffer == nil || scores_buffer == nil) {
            [queries_buffer release];
            [scores_buffer release];
            return -7;
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            [queries_buffer release];
            [scores_buffer release];
            return -99;
        }

        uint32_t dim_u32 = static_cast<uint32_t>(cache->dim);
        uint32_t n_slots_u32 = static_cast<uint32_t>(n_slots);

        [encoder setComputePipelineState:state.pipeline];
        [encoder setBuffer:cache->vectors offset:0 atIndex:0];
        [encoder setBuffer:cache->active  offset:0 atIndex:1];
        [encoder setBuffer:queries_buffer offset:0 atIndex:2];
        [encoder setBuffer:scores_buffer  offset:0 atIndex:3];
        [encoder setBytes:&dim_u32 length:sizeof(dim_u32) atIndex:4];
        [encoder setBytes:&n_slots_u32 length:sizeof(n_slots_u32) atIndex:5];

        const NSUInteger threads = static_cast<NSUInteger>(total);
        const NSUInteger group = std::min<NSUInteger>(
            256,
            static_cast<NSUInteger>(state.pipeline.maxTotalThreadsPerThreadgroup));
        [encoder dispatchThreads:MTLSizeMake(threads, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            [queries_buffer release];
            [scores_buffer release];
            return -99;
        }

        std::memcpy(scores, [scores_buffer contents], total * sizeof(float));

        [queries_buffer release];
        [scores_buffer release];
    }

    return 0;
}

int ggml_vec_index_metal_topk_f32(
        ggml_vec_index_metal_index * cache,
        const float   * queries,
        int             n_q,
        int             k,
        int             block_size,
        float         * candidate_scores,
        uint32_t      * candidate_slots) {
    if (cache == nullptr || queries == nullptr ||
        candidate_scores == nullptr || candidate_slots == nullptr ||
        n_q <= 0 || k <= 0 || k > 64 || block_size <= 0 ||
        cache->vectors == nil || cache->active == nil ||
        cache->n_slots == 0 || cache->dim <= 0) {
        return -2;
    }

    const size_t n_slots = cache->n_slots;
    const size_t n_q_sz = static_cast<size_t>(n_q);
    const size_t k_sz = static_cast<size_t>(k);
    const size_t block_size_sz = static_cast<size_t>(block_size);
    const size_t n_blocks = (n_slots + block_size_sz - 1) / block_size_sz;
    if (n_blocks == 0 ||
        n_blocks > static_cast<size_t>(UINT32_MAX) ||
        n_blocks > SIZE_MAX / n_q_sz) {
        return -2;
    }

    const size_t total_blocks = n_blocks * n_q_sz;
    if (total_blocks > static_cast<size_t>(UINT32_MAX) ||
        total_blocks > SIZE_MAX / k_sz) {
        return -2;
    }

    const size_t total_candidates = total_blocks * k_sz;
    if (total_candidates > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_topk_unlocked(state) || state.topk_pipeline == nil) {
        return -2;
    }
    if (block_size > 256 ||
        static_cast<NSUInteger>(block_size) > state.topk_pipeline.maxTotalThreadsPerThreadgroup) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(cache->dim);
        if (dim_sz != 0 && n_q_sz > SIZE_MAX / dim_sz / sizeof(float)) {
            return -2;
        }
        if (total_candidates > SIZE_MAX / sizeof(float) ||
            total_candidates > SIZE_MAX / sizeof(uint32_t)) {
            return -2;
        }

        id<MTLBuffer> queries_buffer =
            [state.device newBufferWithBytes:queries
                                      length:n_q_sz * dim_sz * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(float)
                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> slots_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];

        if (queries_buffer == nil || scores_buffer == nil || slots_buffer == nil) {
            [queries_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -7;
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            [queries_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        uint32_t dim_u32 = static_cast<uint32_t>(cache->dim);
        uint32_t n_slots_u32 = static_cast<uint32_t>(n_slots);
        uint32_t k_u32 = static_cast<uint32_t>(k);
        uint32_t block_size_u32 = static_cast<uint32_t>(block_size);
        uint32_t n_blocks_u32 = static_cast<uint32_t>(n_blocks);

        [encoder setComputePipelineState:state.topk_pipeline];
        [encoder setBuffer:cache->vectors offset:0 atIndex:0];
        [encoder setBuffer:cache->active  offset:0 atIndex:1];
        [encoder setBuffer:queries_buffer offset:0 atIndex:2];
        [encoder setBuffer:scores_buffer  offset:0 atIndex:3];
        [encoder setBuffer:slots_buffer   offset:0 atIndex:4];
        [encoder setBytes:&dim_u32 length:sizeof(dim_u32) atIndex:5];
        [encoder setBytes:&n_slots_u32 length:sizeof(n_slots_u32) atIndex:6];
        [encoder setBytes:&k_u32 length:sizeof(k_u32) atIndex:7];
        [encoder setBytes:&block_size_u32 length:sizeof(block_size_u32) atIndex:8];
        [encoder setBytes:&n_blocks_u32 length:sizeof(n_blocks_u32) atIndex:9];

        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(total_blocks), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(block_size), 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            [queries_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        std::memcpy(candidate_scores, [scores_buffer contents], total_candidates * sizeof(float));
        std::memcpy(candidate_slots, [slots_buffer contents], total_candidates * sizeof(uint32_t));

        [queries_buffer release];
        [scores_buffer release];
        [slots_buffer release];
    }

    return 0;
}

int ggml_vec_index_metal_topk_filter_f32(
        ggml_vec_index_metal_index * cache,
        const float   * queries,
        const uint32_t * filter_slots,
        size_t          n_filter,
        int             n_q,
        int             k,
        int             block_size,
        float         * candidate_scores,
        uint32_t      * candidate_slots) {
    if (cache == nullptr || queries == nullptr || filter_slots == nullptr ||
        candidate_scores == nullptr || candidate_slots == nullptr ||
        n_filter == 0 || n_filter > static_cast<size_t>(UINT32_MAX) ||
        n_q <= 0 || k <= 0 || k > 64 || block_size <= 0 ||
        cache->vectors == nil || cache->active == nil ||
        cache->n_slots == 0 || cache->dim <= 0) {
        return -2;
    }

    const size_t n_q_sz = static_cast<size_t>(n_q);
    const size_t k_sz = static_cast<size_t>(k);
    const size_t block_size_sz = static_cast<size_t>(block_size);
    const size_t n_blocks = (n_filter + block_size_sz - 1) / block_size_sz;
    if (n_blocks == 0 ||
        n_blocks > static_cast<size_t>(UINT32_MAX) ||
        n_blocks > SIZE_MAX / n_q_sz) {
        return -2;
    }

    const size_t total_blocks = n_blocks * n_q_sz;
    if (total_blocks > static_cast<size_t>(UINT32_MAX) ||
        total_blocks > SIZE_MAX / k_sz) {
        return -2;
    }

    const size_t total_candidates = total_blocks * k_sz;
    if (total_candidates > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_topk_filter_unlocked(state) || state.topk_filter_pipeline == nil) {
        return -2;
    }
    if (block_size > 256 ||
        static_cast<NSUInteger>(block_size) > state.topk_filter_pipeline.maxTotalThreadsPerThreadgroup) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(cache->dim);
        if (dim_sz != 0 && n_q_sz > SIZE_MAX / dim_sz / sizeof(float)) {
            return -2;
        }
        if (n_filter > SIZE_MAX / sizeof(uint32_t) ||
            total_candidates > SIZE_MAX / sizeof(float) ||
            total_candidates > SIZE_MAX / sizeof(uint32_t)) {
            return -2;
        }

        id<MTLBuffer> queries_buffer =
            [state.device newBufferWithBytes:queries
                                      length:n_q_sz * dim_sz * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> filter_buffer =
            [state.device newBufferWithBytes:filter_slots
                                      length:n_filter * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(float)
                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> slots_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];

        if (queries_buffer == nil || filter_buffer == nil || scores_buffer == nil || slots_buffer == nil) {
            [queries_buffer release];
            [filter_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -7;
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            [queries_buffer release];
            [filter_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        uint32_t dim_u32 = static_cast<uint32_t>(cache->dim);
        uint32_t n_filter_u32 = static_cast<uint32_t>(n_filter);
        uint32_t k_u32 = static_cast<uint32_t>(k);
        uint32_t block_size_u32 = static_cast<uint32_t>(block_size);
        uint32_t n_blocks_u32 = static_cast<uint32_t>(n_blocks);

        [encoder setComputePipelineState:state.topk_filter_pipeline];
        [encoder setBuffer:cache->vectors offset:0 atIndex:0];
        [encoder setBuffer:cache->active  offset:0 atIndex:1];
        [encoder setBuffer:queries_buffer offset:0 atIndex:2];
        [encoder setBuffer:filter_buffer  offset:0 atIndex:3];
        [encoder setBuffer:scores_buffer  offset:0 atIndex:4];
        [encoder setBuffer:slots_buffer   offset:0 atIndex:5];
        [encoder setBytes:&dim_u32 length:sizeof(dim_u32) atIndex:6];
        [encoder setBytes:&n_filter_u32 length:sizeof(n_filter_u32) atIndex:7];
        [encoder setBytes:&k_u32 length:sizeof(k_u32) atIndex:8];
        [encoder setBytes:&block_size_u32 length:sizeof(block_size_u32) atIndex:9];
        [encoder setBytes:&n_blocks_u32 length:sizeof(n_blocks_u32) atIndex:10];

        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(total_blocks), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(block_size), 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            [queries_buffer release];
            [filter_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        std::memcpy(candidate_scores, [scores_buffer contents], total_candidates * sizeof(float));
        std::memcpy(candidate_slots, [slots_buffer contents], total_candidates * sizeof(uint32_t));

        [queries_buffer release];
        [filter_buffer release];
        [scores_buffer release];
        [slots_buffer release];
    }

    return 0;
}

int ggml_vec_index_metal_topk_filter_blocks_f32(
        ggml_vec_index_metal_index * cache,
        const float   * queries,
        const uint32_t * filter_slots,
        const uint32_t * block_queries,
        const uint32_t * block_offsets,
        const uint32_t * block_counts,
        size_t          n_filter,
        size_t          n_blocks,
        int             n_q,
        int             k,
        int             block_size,
        float         * candidate_scores,
        uint32_t      * candidate_slots) {
    if (cache == nullptr || queries == nullptr || filter_slots == nullptr ||
        block_queries == nullptr || block_offsets == nullptr || block_counts == nullptr ||
        candidate_scores == nullptr || candidate_slots == nullptr ||
        n_filter == 0 || n_filter > static_cast<size_t>(UINT32_MAX) ||
        n_blocks == 0 || n_blocks > static_cast<size_t>(UINT32_MAX) ||
        n_q <= 0 || k <= 0 || k > 64 || block_size <= 0 ||
        cache->vectors == nil || cache->active == nil ||
        cache->n_slots == 0 || cache->dim <= 0) {
        return -2;
    }

    const size_t n_q_sz = static_cast<size_t>(n_q);
    const size_t k_sz = static_cast<size_t>(k);
    if (n_blocks > SIZE_MAX / k_sz) {
        return -2;
    }

    const size_t total_candidates = n_blocks * k_sz;
    if (total_candidates > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_topk_filter_blocks_unlocked(state) || state.topk_filter_blocks_pipeline == nil) {
        return -2;
    }
    if (block_size > 256 ||
        static_cast<NSUInteger>(block_size) > state.topk_filter_blocks_pipeline.maxTotalThreadsPerThreadgroup) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(cache->dim);
        if (dim_sz != 0 && n_q_sz > SIZE_MAX / dim_sz / sizeof(float)) {
            return -2;
        }
        if (n_filter > SIZE_MAX / sizeof(uint32_t) ||
            n_blocks > SIZE_MAX / sizeof(uint32_t) ||
            total_candidates > SIZE_MAX / sizeof(float) ||
            total_candidates > SIZE_MAX / sizeof(uint32_t)) {
            return -2;
        }

        id<MTLBuffer> queries_buffer =
            [state.device newBufferWithBytes:queries
                                      length:n_q_sz * dim_sz * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> filter_buffer =
            [state.device newBufferWithBytes:filter_slots
                                      length:n_filter * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> block_queries_buffer =
            [state.device newBufferWithBytes:block_queries
                                      length:n_blocks * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> block_offsets_buffer =
            [state.device newBufferWithBytes:block_offsets
                                      length:n_blocks * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> block_counts_buffer =
            [state.device newBufferWithBytes:block_counts
                                      length:n_blocks * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(float)
                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> slots_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];

        if (queries_buffer == nil || filter_buffer == nil ||
            block_queries_buffer == nil || block_offsets_buffer == nil ||
            block_counts_buffer == nil || scores_buffer == nil || slots_buffer == nil) {
            [queries_buffer release];
            [filter_buffer release];
            [block_queries_buffer release];
            [block_offsets_buffer release];
            [block_counts_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -7;
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            [queries_buffer release];
            [filter_buffer release];
            [block_queries_buffer release];
            [block_offsets_buffer release];
            [block_counts_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        uint32_t dim_u32 = static_cast<uint32_t>(cache->dim);
        uint32_t k_u32 = static_cast<uint32_t>(k);
        uint32_t block_size_u32 = static_cast<uint32_t>(block_size);

        [encoder setComputePipelineState:state.topk_filter_blocks_pipeline];
        [encoder setBuffer:cache->vectors        offset:0 atIndex:0];
        [encoder setBuffer:cache->active         offset:0 atIndex:1];
        [encoder setBuffer:queries_buffer        offset:0 atIndex:2];
        [encoder setBuffer:filter_buffer         offset:0 atIndex:3];
        [encoder setBuffer:block_queries_buffer  offset:0 atIndex:4];
        [encoder setBuffer:block_offsets_buffer  offset:0 atIndex:5];
        [encoder setBuffer:block_counts_buffer   offset:0 atIndex:6];
        [encoder setBuffer:scores_buffer         offset:0 atIndex:7];
        [encoder setBuffer:slots_buffer          offset:0 atIndex:8];
        [encoder setBytes:&dim_u32 length:sizeof(dim_u32) atIndex:9];
        [encoder setBytes:&k_u32 length:sizeof(k_u32) atIndex:10];
        [encoder setBytes:&block_size_u32 length:sizeof(block_size_u32) atIndex:11];

        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_blocks), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(block_size), 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            [queries_buffer release];
            [filter_buffer release];
            [block_queries_buffer release];
            [block_offsets_buffer release];
            [block_counts_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        std::memcpy(candidate_scores, [scores_buffer contents], total_candidates * sizeof(float));
        std::memcpy(candidate_slots, [slots_buffer contents], total_candidates * sizeof(uint32_t));

        [queries_buffer release];
        [filter_buffer release];
        [block_queries_buffer release];
        [block_offsets_buffer release];
        [block_counts_buffer release];
        [scores_buffer release];
        [slots_buffer release];
    }

    return 0;
}

int ggml_vec_index_metal_topk_q8(
        ggml_vec_index_metal_index * cache,
        const float   * queries,
        int             n_q,
        int             k,
        int             block_size,
        float         * candidate_scores,
        uint32_t      * candidate_slots) {
    if (cache == nullptr || queries == nullptr ||
        candidate_scores == nullptr || candidate_slots == nullptr ||
        n_q <= 0 || k <= 0 || k > 64 || block_size <= 0 ||
        cache->q8_codes == nil || cache->q8_scales == nil || cache->active == nil ||
        cache->n_slots == 0 || cache->dim <= 0) {
        return -2;
    }

    const size_t n_slots = cache->n_slots;
    const size_t n_q_sz = static_cast<size_t>(n_q);
    const size_t k_sz = static_cast<size_t>(k);
    const size_t block_size_sz = static_cast<size_t>(block_size);
    const size_t n_blocks = (n_slots + block_size_sz - 1) / block_size_sz;
    if (n_blocks == 0 ||
        n_blocks > static_cast<size_t>(UINT32_MAX) ||
        n_blocks > SIZE_MAX / n_q_sz) {
        return -2;
    }

    const size_t total_blocks = n_blocks * n_q_sz;
    if (total_blocks > static_cast<size_t>(UINT32_MAX) ||
        total_blocks > SIZE_MAX / k_sz) {
        return -2;
    }

    const size_t total_candidates = total_blocks * k_sz;
    if (total_candidates > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_topk_q8_unlocked(state) || state.topk_q8_pipeline == nil) {
        return -2;
    }
    if (block_size > 256 ||
        static_cast<NSUInteger>(block_size) > state.topk_q8_pipeline.maxTotalThreadsPerThreadgroup) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(cache->dim);
        if (dim_sz != 0 && n_q_sz > SIZE_MAX / dim_sz / sizeof(float)) {
            return -2;
        }
        if (total_candidates > SIZE_MAX / sizeof(float) ||
            total_candidates > SIZE_MAX / sizeof(uint32_t)) {
            return -2;
        }

        id<MTLBuffer> queries_buffer =
            [state.device newBufferWithBytes:queries
                                      length:n_q_sz * dim_sz * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(float)
                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> slots_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];

        if (queries_buffer == nil || scores_buffer == nil || slots_buffer == nil) {
            [queries_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -7;
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            [queries_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        uint32_t dim_u32 = static_cast<uint32_t>(cache->dim);
        uint32_t n_slots_u32 = static_cast<uint32_t>(n_slots);
        uint32_t k_u32 = static_cast<uint32_t>(k);
        uint32_t block_size_u32 = static_cast<uint32_t>(block_size);
        uint32_t n_blocks_u32 = static_cast<uint32_t>(n_blocks);

        [encoder setComputePipelineState:state.topk_q8_pipeline];
        [encoder setBuffer:cache->q8_codes  offset:0 atIndex:0];
        [encoder setBuffer:cache->q8_scales offset:0 atIndex:1];
        [encoder setBuffer:cache->active    offset:0 atIndex:2];
        [encoder setBuffer:queries_buffer   offset:0 atIndex:3];
        [encoder setBuffer:scores_buffer    offset:0 atIndex:4];
        [encoder setBuffer:slots_buffer     offset:0 atIndex:5];
        [encoder setBytes:&dim_u32 length:sizeof(dim_u32) atIndex:6];
        [encoder setBytes:&n_slots_u32 length:sizeof(n_slots_u32) atIndex:7];
        [encoder setBytes:&k_u32 length:sizeof(k_u32) atIndex:8];
        [encoder setBytes:&block_size_u32 length:sizeof(block_size_u32) atIndex:9];
        [encoder setBytes:&n_blocks_u32 length:sizeof(n_blocks_u32) atIndex:10];

        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(total_blocks), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(block_size), 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            [queries_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        std::memcpy(candidate_scores, [scores_buffer contents], total_candidates * sizeof(float));
        std::memcpy(candidate_slots, [slots_buffer contents], total_candidates * sizeof(uint32_t));

        [queries_buffer release];
        [scores_buffer release];
        [slots_buffer release];
    }

    return 0;
}

int ggml_vec_index_metal_topk_filter_q8(
        ggml_vec_index_metal_index * cache,
        const float   * queries,
        const uint32_t * filter_slots,
        size_t          n_filter,
        int             n_q,
        int             k,
        int             block_size,
        float         * candidate_scores,
        uint32_t      * candidate_slots) {
    if (cache == nullptr || queries == nullptr || filter_slots == nullptr ||
        candidate_scores == nullptr || candidate_slots == nullptr ||
        n_filter == 0 || n_filter > static_cast<size_t>(UINT32_MAX) ||
        n_q <= 0 || k <= 0 || k > 64 || block_size <= 0 ||
        cache->q8_codes == nil || cache->q8_scales == nil || cache->active == nil ||
        cache->n_slots == 0 || cache->dim <= 0) {
        return -2;
    }

    const size_t n_q_sz = static_cast<size_t>(n_q);
    const size_t k_sz = static_cast<size_t>(k);
    const size_t block_size_sz = static_cast<size_t>(block_size);
    const size_t n_blocks = (n_filter + block_size_sz - 1) / block_size_sz;
    if (n_blocks == 0 ||
        n_blocks > static_cast<size_t>(UINT32_MAX) ||
        n_blocks > SIZE_MAX / n_q_sz) {
        return -2;
    }

    const size_t total_blocks = n_blocks * n_q_sz;
    if (total_blocks > static_cast<size_t>(UINT32_MAX) ||
        total_blocks > SIZE_MAX / k_sz) {
        return -2;
    }

    const size_t total_candidates = total_blocks * k_sz;
    if (total_candidates > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_topk_filter_q8_unlocked(state) || state.topk_filter_q8_pipeline == nil) {
        return -2;
    }
    if (block_size > 256 ||
        static_cast<NSUInteger>(block_size) > state.topk_filter_q8_pipeline.maxTotalThreadsPerThreadgroup) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(cache->dim);
        if (dim_sz != 0 && n_q_sz > SIZE_MAX / dim_sz / sizeof(float)) {
            return -2;
        }
        if (n_filter > SIZE_MAX / sizeof(uint32_t) ||
            total_candidates > SIZE_MAX / sizeof(float) ||
            total_candidates > SIZE_MAX / sizeof(uint32_t)) {
            return -2;
        }

        id<MTLBuffer> queries_buffer =
            [state.device newBufferWithBytes:queries
                                      length:n_q_sz * dim_sz * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> filter_buffer =
            [state.device newBufferWithBytes:filter_slots
                                      length:n_filter * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(float)
                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> slots_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];

        if (queries_buffer == nil || filter_buffer == nil || scores_buffer == nil || slots_buffer == nil) {
            [queries_buffer release];
            [filter_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -7;
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            [queries_buffer release];
            [filter_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        uint32_t dim_u32 = static_cast<uint32_t>(cache->dim);
        uint32_t n_filter_u32 = static_cast<uint32_t>(n_filter);
        uint32_t k_u32 = static_cast<uint32_t>(k);
        uint32_t block_size_u32 = static_cast<uint32_t>(block_size);
        uint32_t n_blocks_u32 = static_cast<uint32_t>(n_blocks);

        [encoder setComputePipelineState:state.topk_filter_q8_pipeline];
        [encoder setBuffer:cache->q8_codes  offset:0 atIndex:0];
        [encoder setBuffer:cache->q8_scales offset:0 atIndex:1];
        [encoder setBuffer:cache->active    offset:0 atIndex:2];
        [encoder setBuffer:queries_buffer   offset:0 atIndex:3];
        [encoder setBuffer:filter_buffer    offset:0 atIndex:4];
        [encoder setBuffer:scores_buffer    offset:0 atIndex:5];
        [encoder setBuffer:slots_buffer     offset:0 atIndex:6];
        [encoder setBytes:&dim_u32 length:sizeof(dim_u32) atIndex:7];
        [encoder setBytes:&n_filter_u32 length:sizeof(n_filter_u32) atIndex:8];
        [encoder setBytes:&k_u32 length:sizeof(k_u32) atIndex:9];
        [encoder setBytes:&block_size_u32 length:sizeof(block_size_u32) atIndex:10];
        [encoder setBytes:&n_blocks_u32 length:sizeof(n_blocks_u32) atIndex:11];

        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(total_blocks), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(block_size), 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            [queries_buffer release];
            [filter_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        std::memcpy(candidate_scores, [scores_buffer contents], total_candidates * sizeof(float));
        std::memcpy(candidate_slots, [slots_buffer contents], total_candidates * sizeof(uint32_t));

        [queries_buffer release];
        [filter_buffer release];
        [scores_buffer release];
        [slots_buffer release];
    }

    return 0;
}

int ggml_vec_index_metal_topk_filter_blocks_q8(
        ggml_vec_index_metal_index * cache,
        const float   * queries,
        const uint32_t * filter_slots,
        const uint32_t * block_queries,
        const uint32_t * block_offsets,
        const uint32_t * block_counts,
        size_t          n_filter,
        size_t          n_blocks,
        int             n_q,
        int             k,
        int             block_size,
        float         * candidate_scores,
        uint32_t      * candidate_slots) {
    if (cache == nullptr || queries == nullptr || filter_slots == nullptr ||
        block_queries == nullptr || block_offsets == nullptr || block_counts == nullptr ||
        candidate_scores == nullptr || candidate_slots == nullptr ||
        n_filter == 0 || n_filter > static_cast<size_t>(UINT32_MAX) ||
        n_blocks == 0 || n_blocks > static_cast<size_t>(UINT32_MAX) ||
        n_q <= 0 || k <= 0 || k > 64 || block_size <= 0 ||
        cache->q8_codes == nil || cache->q8_scales == nil || cache->active == nil ||
        cache->n_slots == 0 || cache->dim <= 0) {
        return -2;
    }

    const size_t n_q_sz = static_cast<size_t>(n_q);
    const size_t k_sz = static_cast<size_t>(k);
    if (n_blocks > SIZE_MAX / k_sz) {
        return -2;
    }

    const size_t total_candidates = n_blocks * k_sz;
    if (total_candidates > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_topk_filter_blocks_q8_unlocked(state) || state.topk_filter_blocks_q8_pipeline == nil) {
        return -2;
    }
    if (block_size > 256 ||
        static_cast<NSUInteger>(block_size) > state.topk_filter_blocks_q8_pipeline.maxTotalThreadsPerThreadgroup) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(cache->dim);
        if (dim_sz != 0 && n_q_sz > SIZE_MAX / dim_sz / sizeof(float)) {
            return -2;
        }
        if (n_filter > SIZE_MAX / sizeof(uint32_t) ||
            n_blocks > SIZE_MAX / sizeof(uint32_t) ||
            total_candidates > SIZE_MAX / sizeof(float) ||
            total_candidates > SIZE_MAX / sizeof(uint32_t)) {
            return -2;
        }

        id<MTLBuffer> queries_buffer =
            [state.device newBufferWithBytes:queries
                                      length:n_q_sz * dim_sz * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> filter_buffer =
            [state.device newBufferWithBytes:filter_slots
                                      length:n_filter * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> block_queries_buffer =
            [state.device newBufferWithBytes:block_queries
                                      length:n_blocks * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> block_offsets_buffer =
            [state.device newBufferWithBytes:block_offsets
                                      length:n_blocks * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> block_counts_buffer =
            [state.device newBufferWithBytes:block_counts
                                      length:n_blocks * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(float)
                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> slots_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];

        if (queries_buffer == nil || filter_buffer == nil ||
            block_queries_buffer == nil || block_offsets_buffer == nil ||
            block_counts_buffer == nil || scores_buffer == nil || slots_buffer == nil) {
            [queries_buffer release];
            [filter_buffer release];
            [block_queries_buffer release];
            [block_offsets_buffer release];
            [block_counts_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -7;
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            [queries_buffer release];
            [filter_buffer release];
            [block_queries_buffer release];
            [block_offsets_buffer release];
            [block_counts_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        uint32_t dim_u32 = static_cast<uint32_t>(cache->dim);
        uint32_t k_u32 = static_cast<uint32_t>(k);
        uint32_t block_size_u32 = static_cast<uint32_t>(block_size);

        [encoder setComputePipelineState:state.topk_filter_blocks_q8_pipeline];
        [encoder setBuffer:cache->q8_codes       offset:0 atIndex:0];
        [encoder setBuffer:cache->q8_scales      offset:0 atIndex:1];
        [encoder setBuffer:cache->active         offset:0 atIndex:2];
        [encoder setBuffer:queries_buffer        offset:0 atIndex:3];
        [encoder setBuffer:filter_buffer         offset:0 atIndex:4];
        [encoder setBuffer:block_queries_buffer  offset:0 atIndex:5];
        [encoder setBuffer:block_offsets_buffer  offset:0 atIndex:6];
        [encoder setBuffer:block_counts_buffer   offset:0 atIndex:7];
        [encoder setBuffer:scores_buffer         offset:0 atIndex:8];
        [encoder setBuffer:slots_buffer          offset:0 atIndex:9];
        [encoder setBytes:&dim_u32 length:sizeof(dim_u32) atIndex:10];
        [encoder setBytes:&k_u32 length:sizeof(k_u32) atIndex:11];
        [encoder setBytes:&block_size_u32 length:sizeof(block_size_u32) atIndex:12];

        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_blocks), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(block_size), 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            [queries_buffer release];
            [filter_buffer release];
            [block_queries_buffer release];
            [block_offsets_buffer release];
            [block_counts_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        std::memcpy(candidate_scores, [scores_buffer contents], total_candidates * sizeof(float));
        std::memcpy(candidate_slots, [slots_buffer contents], total_candidates * sizeof(uint32_t));

        [queries_buffer release];
        [filter_buffer release];
        [block_queries_buffer release];
        [block_offsets_buffer release];
        [block_counts_buffer release];
        [scores_buffer release];
        [slots_buffer release];
    }

    return 0;
}

int ggml_vec_index_metal_topk_q4(
        ggml_vec_index_metal_index * cache,
        const float   * queries,
        int             n_q,
        int             k,
        int             block_size,
        float         * candidate_scores,
        uint32_t      * candidate_slots) {
    if (cache == nullptr || queries == nullptr ||
        candidate_scores == nullptr || candidate_slots == nullptr ||
        n_q <= 0 || k <= 0 || k > 64 || block_size <= 0 ||
        cache->q4_codes == nil || cache->q4_scales == nil || cache->active == nil ||
        cache->n_slots == 0 || cache->dim <= 0) {
        return -2;
    }

    const size_t n_slots = cache->n_slots;
    const size_t n_q_sz = static_cast<size_t>(n_q);
    const size_t k_sz = static_cast<size_t>(k);
    const size_t block_size_sz = static_cast<size_t>(block_size);
    const size_t n_blocks = (n_slots + block_size_sz - 1) / block_size_sz;
    if (n_blocks == 0 ||
        n_blocks > static_cast<size_t>(UINT32_MAX) ||
        n_blocks > SIZE_MAX / n_q_sz) {
        return -2;
    }

    const size_t total_blocks = n_blocks * n_q_sz;
    if (total_blocks > static_cast<size_t>(UINT32_MAX) ||
        total_blocks > SIZE_MAX / k_sz) {
        return -2;
    }

    const size_t total_candidates = total_blocks * k_sz;
    if (total_candidates > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_topk_q4_unlocked(state) || state.topk_q4_pipeline == nil) {
        return -2;
    }
    if (block_size > 256 ||
        static_cast<NSUInteger>(block_size) > state.topk_q4_pipeline.maxTotalThreadsPerThreadgroup) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(cache->dim);
        const size_t row_bytes = (dim_sz + 1) / 2;
        if (dim_sz != 0 && n_q_sz > SIZE_MAX / dim_sz / sizeof(float)) {
            return -2;
        }
        if (row_bytes > static_cast<size_t>(UINT32_MAX) ||
            total_candidates > SIZE_MAX / sizeof(float) ||
            total_candidates > SIZE_MAX / sizeof(uint32_t)) {
            return -2;
        }

        id<MTLBuffer> queries_buffer =
            [state.device newBufferWithBytes:queries
                                      length:n_q_sz * dim_sz * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(float)
                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> slots_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];

        if (queries_buffer == nil || scores_buffer == nil || slots_buffer == nil) {
            [queries_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -7;
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            [queries_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        uint32_t dim_u32 = static_cast<uint32_t>(cache->dim);
        uint32_t row_bytes_u32 = static_cast<uint32_t>(row_bytes);
        uint32_t n_slots_u32 = static_cast<uint32_t>(n_slots);
        uint32_t k_u32 = static_cast<uint32_t>(k);
        uint32_t block_size_u32 = static_cast<uint32_t>(block_size);
        uint32_t n_blocks_u32 = static_cast<uint32_t>(n_blocks);

        [encoder setComputePipelineState:state.topk_q4_pipeline];
        [encoder setBuffer:cache->q4_codes  offset:0 atIndex:0];
        [encoder setBuffer:cache->q4_scales offset:0 atIndex:1];
        [encoder setBuffer:cache->active    offset:0 atIndex:2];
        [encoder setBuffer:queries_buffer   offset:0 atIndex:3];
        [encoder setBuffer:scores_buffer    offset:0 atIndex:4];
        [encoder setBuffer:slots_buffer     offset:0 atIndex:5];
        [encoder setBytes:&dim_u32 length:sizeof(dim_u32) atIndex:6];
        [encoder setBytes:&row_bytes_u32 length:sizeof(row_bytes_u32) atIndex:7];
        [encoder setBytes:&n_slots_u32 length:sizeof(n_slots_u32) atIndex:8];
        [encoder setBytes:&k_u32 length:sizeof(k_u32) atIndex:9];
        [encoder setBytes:&block_size_u32 length:sizeof(block_size_u32) atIndex:10];
        [encoder setBytes:&n_blocks_u32 length:sizeof(n_blocks_u32) atIndex:11];

        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(total_blocks), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(block_size), 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            [queries_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        std::memcpy(candidate_scores, [scores_buffer contents], total_candidates * sizeof(float));
        std::memcpy(candidate_slots, [slots_buffer contents], total_candidates * sizeof(uint32_t));

        [queries_buffer release];
        [scores_buffer release];
        [slots_buffer release];
    }

    return 0;
}

int ggml_vec_index_metal_topk_filter_q4(
        ggml_vec_index_metal_index * cache,
        const float   * queries,
        const uint32_t * filter_slots,
        size_t          n_filter,
        int             n_q,
        int             k,
        int             block_size,
        float         * candidate_scores,
        uint32_t      * candidate_slots) {
    if (cache == nullptr || queries == nullptr || filter_slots == nullptr ||
        candidate_scores == nullptr || candidate_slots == nullptr ||
        n_filter == 0 || n_filter > static_cast<size_t>(UINT32_MAX) ||
        n_q <= 0 || k <= 0 || k > 64 || block_size <= 0 ||
        cache->q4_codes == nil || cache->q4_scales == nil || cache->active == nil ||
        cache->n_slots == 0 || cache->dim <= 0) {
        return -2;
    }

    const size_t n_q_sz = static_cast<size_t>(n_q);
    const size_t k_sz = static_cast<size_t>(k);
    const size_t block_size_sz = static_cast<size_t>(block_size);
    const size_t n_blocks = (n_filter + block_size_sz - 1) / block_size_sz;
    if (n_blocks == 0 ||
        n_blocks > static_cast<size_t>(UINT32_MAX) ||
        n_blocks > SIZE_MAX / n_q_sz) {
        return -2;
    }

    const size_t total_blocks = n_blocks * n_q_sz;
    if (total_blocks > static_cast<size_t>(UINT32_MAX) ||
        total_blocks > SIZE_MAX / k_sz) {
        return -2;
    }

    const size_t total_candidates = total_blocks * k_sz;
    if (total_candidates > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_topk_filter_q4_unlocked(state) || state.topk_filter_q4_pipeline == nil) {
        return -2;
    }
    if (block_size > 256 ||
        static_cast<NSUInteger>(block_size) > state.topk_filter_q4_pipeline.maxTotalThreadsPerThreadgroup) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(cache->dim);
        const size_t row_bytes = (dim_sz + 1) / 2;
        if (dim_sz != 0 && n_q_sz > SIZE_MAX / dim_sz / sizeof(float)) {
            return -2;
        }
        if (row_bytes > static_cast<size_t>(UINT32_MAX) ||
            n_filter > SIZE_MAX / sizeof(uint32_t) ||
            total_candidates > SIZE_MAX / sizeof(float) ||
            total_candidates > SIZE_MAX / sizeof(uint32_t)) {
            return -2;
        }

        id<MTLBuffer> queries_buffer =
            [state.device newBufferWithBytes:queries
                                      length:n_q_sz * dim_sz * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> filter_buffer =
            [state.device newBufferWithBytes:filter_slots
                                      length:n_filter * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(float)
                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> slots_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];

        if (queries_buffer == nil || filter_buffer == nil || scores_buffer == nil || slots_buffer == nil) {
            [queries_buffer release];
            [filter_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -7;
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            [queries_buffer release];
            [filter_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        uint32_t dim_u32 = static_cast<uint32_t>(cache->dim);
        uint32_t row_bytes_u32 = static_cast<uint32_t>(row_bytes);
        uint32_t n_filter_u32 = static_cast<uint32_t>(n_filter);
        uint32_t k_u32 = static_cast<uint32_t>(k);
        uint32_t block_size_u32 = static_cast<uint32_t>(block_size);
        uint32_t n_blocks_u32 = static_cast<uint32_t>(n_blocks);

        [encoder setComputePipelineState:state.topk_filter_q4_pipeline];
        [encoder setBuffer:cache->q4_codes  offset:0 atIndex:0];
        [encoder setBuffer:cache->q4_scales offset:0 atIndex:1];
        [encoder setBuffer:cache->active    offset:0 atIndex:2];
        [encoder setBuffer:queries_buffer   offset:0 atIndex:3];
        [encoder setBuffer:filter_buffer    offset:0 atIndex:4];
        [encoder setBuffer:scores_buffer    offset:0 atIndex:5];
        [encoder setBuffer:slots_buffer     offset:0 atIndex:6];
        [encoder setBytes:&dim_u32 length:sizeof(dim_u32) atIndex:7];
        [encoder setBytes:&row_bytes_u32 length:sizeof(row_bytes_u32) atIndex:8];
        [encoder setBytes:&n_filter_u32 length:sizeof(n_filter_u32) atIndex:9];
        [encoder setBytes:&k_u32 length:sizeof(k_u32) atIndex:10];
        [encoder setBytes:&block_size_u32 length:sizeof(block_size_u32) atIndex:11];
        [encoder setBytes:&n_blocks_u32 length:sizeof(n_blocks_u32) atIndex:12];

        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(total_blocks), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(block_size), 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            [queries_buffer release];
            [filter_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        std::memcpy(candidate_scores, [scores_buffer contents], total_candidates * sizeof(float));
        std::memcpy(candidate_slots, [slots_buffer contents], total_candidates * sizeof(uint32_t));

        [queries_buffer release];
        [filter_buffer release];
        [scores_buffer release];
        [slots_buffer release];
    }

    return 0;
}

int ggml_vec_index_metal_topk_filter_blocks_q4(
        ggml_vec_index_metal_index * cache,
        const float   * queries,
        const uint32_t * filter_slots,
        const uint32_t * block_queries,
        const uint32_t * block_offsets,
        const uint32_t * block_counts,
        size_t          n_filter,
        size_t          n_blocks,
        int             n_q,
        int             k,
        int             block_size,
        float         * candidate_scores,
        uint32_t      * candidate_slots) {
    if (cache == nullptr || queries == nullptr || filter_slots == nullptr ||
        block_queries == nullptr || block_offsets == nullptr || block_counts == nullptr ||
        candidate_scores == nullptr || candidate_slots == nullptr ||
        n_filter == 0 || n_filter > static_cast<size_t>(UINT32_MAX) ||
        n_blocks == 0 || n_blocks > static_cast<size_t>(UINT32_MAX) ||
        n_q <= 0 || k <= 0 || k > 64 || block_size <= 0 ||
        cache->q4_codes == nil || cache->q4_scales == nil || cache->active == nil ||
        cache->n_slots == 0 || cache->dim <= 0) {
        return -2;
    }

    const size_t n_q_sz = static_cast<size_t>(n_q);
    const size_t k_sz = static_cast<size_t>(k);
    if (n_blocks > SIZE_MAX / k_sz) {
        return -2;
    }

    const size_t total_candidates = n_blocks * k_sz;
    if (total_candidates > static_cast<size_t>(UINT32_MAX)) {
        return -2;
    }

    std::lock_guard<std::mutex> lock(metal_mutex());
    MetalState & state = metal_state();
    if (!init_topk_filter_blocks_q4_unlocked(state) || state.topk_filter_blocks_q4_pipeline == nil) {
        return -2;
    }
    if (block_size > 256 ||
        static_cast<NSUInteger>(block_size) > state.topk_filter_blocks_q4_pipeline.maxTotalThreadsPerThreadgroup) {
        return -2;
    }

    @autoreleasepool {
        const size_t dim_sz = static_cast<size_t>(cache->dim);
        const size_t row_bytes = (dim_sz + 1) / 2;
        if (dim_sz != 0 && n_q_sz > SIZE_MAX / dim_sz / sizeof(float)) {
            return -2;
        }
        if (row_bytes > static_cast<size_t>(UINT32_MAX) ||
            n_filter > SIZE_MAX / sizeof(uint32_t) ||
            n_blocks > SIZE_MAX / sizeof(uint32_t) ||
            total_candidates > SIZE_MAX / sizeof(float) ||
            total_candidates > SIZE_MAX / sizeof(uint32_t)) {
            return -2;
        }

        id<MTLBuffer> queries_buffer =
            [state.device newBufferWithBytes:queries
                                      length:n_q_sz * dim_sz * sizeof(float)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> filter_buffer =
            [state.device newBufferWithBytes:filter_slots
                                      length:n_filter * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> block_queries_buffer =
            [state.device newBufferWithBytes:block_queries
                                      length:n_blocks * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> block_offsets_buffer =
            [state.device newBufferWithBytes:block_offsets
                                      length:n_blocks * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> block_counts_buffer =
            [state.device newBufferWithBytes:block_counts
                                      length:n_blocks * sizeof(uint32_t)
                                     options:MTLResourceStorageModeShared];
        id<MTLBuffer> scores_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(float)
                                      options:MTLResourceStorageModeShared];
        id<MTLBuffer> slots_buffer =
            [state.device newBufferWithLength:total_candidates * sizeof(uint32_t)
                                      options:MTLResourceStorageModeShared];

        if (queries_buffer == nil || filter_buffer == nil ||
            block_queries_buffer == nil || block_offsets_buffer == nil ||
            block_counts_buffer == nil || scores_buffer == nil || slots_buffer == nil) {
            [queries_buffer release];
            [filter_buffer release];
            [block_queries_buffer release];
            [block_offsets_buffer release];
            [block_counts_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -7;
        }

        id<MTLCommandBuffer> command_buffer = [state.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [command_buffer computeCommandEncoder];
        if (command_buffer == nil || encoder == nil) {
            [queries_buffer release];
            [filter_buffer release];
            [block_queries_buffer release];
            [block_offsets_buffer release];
            [block_counts_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        uint32_t dim_u32 = static_cast<uint32_t>(cache->dim);
        uint32_t row_bytes_u32 = static_cast<uint32_t>(row_bytes);
        uint32_t k_u32 = static_cast<uint32_t>(k);
        uint32_t block_size_u32 = static_cast<uint32_t>(block_size);

        [encoder setComputePipelineState:state.topk_filter_blocks_q4_pipeline];
        [encoder setBuffer:cache->q4_codes       offset:0 atIndex:0];
        [encoder setBuffer:cache->q4_scales      offset:0 atIndex:1];
        [encoder setBuffer:cache->active         offset:0 atIndex:2];
        [encoder setBuffer:queries_buffer        offset:0 atIndex:3];
        [encoder setBuffer:filter_buffer         offset:0 atIndex:4];
        [encoder setBuffer:block_queries_buffer  offset:0 atIndex:5];
        [encoder setBuffer:block_offsets_buffer  offset:0 atIndex:6];
        [encoder setBuffer:block_counts_buffer   offset:0 atIndex:7];
        [encoder setBuffer:scores_buffer         offset:0 atIndex:8];
        [encoder setBuffer:slots_buffer          offset:0 atIndex:9];
        [encoder setBytes:&dim_u32 length:sizeof(dim_u32) atIndex:10];
        [encoder setBytes:&row_bytes_u32 length:sizeof(row_bytes_u32) atIndex:11];
        [encoder setBytes:&k_u32 length:sizeof(k_u32) atIndex:12];
        [encoder setBytes:&block_size_u32 length:sizeof(block_size_u32) atIndex:13];

        [encoder dispatchThreadgroups:MTLSizeMake(static_cast<NSUInteger>(n_blocks), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(static_cast<NSUInteger>(block_size), 1, 1)];
        [encoder endEncoding];
        [command_buffer commit];
        [command_buffer waitUntilCompleted];

        if (command_buffer.status == MTLCommandBufferStatusError) {
            [queries_buffer release];
            [filter_buffer release];
            [block_queries_buffer release];
            [block_offsets_buffer release];
            [block_counts_buffer release];
            [scores_buffer release];
            [slots_buffer release];
            return -99;
        }

        std::memcpy(candidate_scores, [scores_buffer contents], total_candidates * sizeof(float));
        std::memcpy(candidate_slots, [slots_buffer contents], total_candidates * sizeof(uint32_t));

        [queries_buffer release];
        [filter_buffer release];
        [block_queries_buffer release];
        [block_offsets_buffer release];
        [block_counts_buffer release];
        [scores_buffer release];
        [slots_buffer release];
    }

    return 0;
}
