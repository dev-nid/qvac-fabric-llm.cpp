#pragma once

#include <stddef.h>
#include <stdint.h>

struct ggml_vec_index_metal_index;

bool ggml_vec_index_metal_available(void);

void ggml_vec_index_metal_free(ggml_vec_index_metal_index * cache);

int ggml_vec_index_metal_prepare_f32(
    ggml_vec_index_metal_index ** cache,
    const float   * vectors,
    const uint8_t * active,
    size_t          n_slots,
    int             dim);

int ggml_vec_index_metal_prepare_q8(
    ggml_vec_index_metal_index ** cache,
    const int8_t  * codes,
    const float   * scales,
    const uint8_t * active,
    size_t          n_slots,
    int             dim);

int ggml_vec_index_metal_prepare_q4(
    ggml_vec_index_metal_index ** cache,
    const uint8_t * codes,
    const float   * scales,
    const uint8_t * active,
    size_t          n_slots,
    int             dim);

int ggml_vec_index_metal_score_f32(
    ggml_vec_index_metal_index * cache,
    const float   * queries,
    int             n_q,
    float         * scores);

int ggml_vec_index_metal_topk_f32(
    ggml_vec_index_metal_index * cache,
    const float   * queries,
    int             n_q,
    int             k,
    int             block_size,
    float         * candidate_scores,
    uint32_t      * candidate_slots);

int ggml_vec_index_metal_topk_filter_f32(
    ggml_vec_index_metal_index * cache,
    const float   * queries,
    const uint32_t * filter_slots,
    size_t          n_filter,
    int             n_q,
    int             k,
    int             block_size,
    float         * candidate_scores,
    uint32_t      * candidate_slots);

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
    uint32_t      * candidate_slots);

int ggml_vec_index_metal_topk_q8(
    ggml_vec_index_metal_index * cache,
    const float   * queries,
    int             n_q,
    int             k,
    int             block_size,
    float         * candidate_scores,
    uint32_t      * candidate_slots);

int ggml_vec_index_metal_topk_filter_q8(
    ggml_vec_index_metal_index * cache,
    const float   * queries,
    const uint32_t * filter_slots,
    size_t          n_filter,
    int             n_q,
    int             k,
    int             block_size,
    float         * candidate_scores,
    uint32_t      * candidate_slots);

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
    uint32_t      * candidate_slots);

int ggml_vec_index_metal_topk_q4(
    ggml_vec_index_metal_index * cache,
    const float   * queries,
    int             n_q,
    int             k,
    int             block_size,
    float         * candidate_scores,
    uint32_t      * candidate_slots);

int ggml_vec_index_metal_topk_filter_q4(
    ggml_vec_index_metal_index * cache,
    const float   * queries,
    const uint32_t * filter_slots,
    size_t          n_filter,
    int             n_q,
    int             k,
    int             block_size,
    float         * candidate_scores,
    uint32_t      * candidate_slots);

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
    uint32_t      * candidate_slots);
