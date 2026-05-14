// Unit test for llama_memory_keep_positions_range (DFlash tree compaction).
//
// Cases:
//   1. basic compact (3 cells, keep 2) with rename to p_min = 0
//   2. rename to non-zero p_min (committed prefix preserved)
//   3. sibling-duplicate handling (kept first under tree-mode bypass)
//   4. n_positions == 0 edge case (delegates to seq_rm)
//   5. seq_id < 0 rejection
//
// Uses the public llama.h C API end-to-end against the standard test model
// (tinyllamas/stories15M). Cells are populated via llama_decode; cell state
// is verified via llama_memory_seq_pos_min/max. Sibling-duplicate construction
// requires llama_set_tree_mask (the bypass on apply_ubatch must be active
// for two same-position writes to land in distinct cells).

#include "arg.h"
#include "common.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

static int n_failed = 0;

#define EXPECT(cond, msg) do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s (%s)\n", msg, #cond); \
            ++n_failed; \
        } else { \
            fprintf(stderr, "  ok: %s\n", msg); \
        } \
    } while (0)

static void clear_seq(llama_memory_t mem, llama_seq_id seq) {
    llama_memory_seq_rm(mem, seq, -1, -1);
}

// Decode `n` tokens at sequential positions starting at `pos0`, all on `seq`.
// Uses arbitrary token id 1 (any valid id is fine — we never read logits).
static bool decode_chain(llama_context * ctx, llama_seq_id seq, llama_pos pos0, int n) {
    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        common_batch_add(batch, 1, pos0 + i, { seq }, false);
    }
    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

// Decode `n` tokens at positions[] (potentially with duplicates) on `seq`,
// with a tree mask installed so the contiguity-invariant purge is bypassed.
// All tokens see the root: a fully-visible NxN mask.
static bool decode_tree(llama_context * ctx, llama_seq_id seq, const llama_pos * positions, int n) {
    std::vector<uint8_t> mask((size_t) n * n, 1);
    llama_set_tree_mask(ctx, mask.data(), n);

    llama_batch batch = llama_batch_init(n, 0, 1);
    for (int i = 0; i < n; ++i) {
        common_batch_add(batch, 1, positions[i], { seq }, false);
    }
    int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);

    llama_clear_tree_mask(ctx);
    return rc == 0;
}

static void test_basic_compact_to_zero(llama_context * ctx, llama_memory_t mem) {
    fprintf(stderr, "\n[test_basic_compact_to_zero] decode 3 cells at [0,1,2], keep [0,2] -> [0,1]\n");
    clear_seq(mem, 0);

    EXPECT(decode_chain(ctx, 0, 0, 3), "decode 3 chain tokens");
    EXPECT(llama_memory_seq_pos_min(mem, 0) == 0, "pre: pos_min == 0");
    EXPECT(llama_memory_seq_pos_max(mem, 0) == 2, "pre: pos_max == 2");

    const llama_pos kept[] = { 0, 2 };
    EXPECT(llama_memory_keep_positions_range(mem, 0, kept, 2, /*p_min=*/0),
            "keep_positions_range succeeds");
    EXPECT(llama_memory_seq_pos_min(mem, 0) == 0, "post: pos_min == 0");
    EXPECT(llama_memory_seq_pos_max(mem, 0) == 1,
            "post: pos_max == 1 (kept cells renamed to [0,1])");
}

static void test_rename_with_committed_prefix(llama_context * ctx, llama_memory_t mem) {
    fprintf(stderr, "\n[test_rename_with_committed_prefix] keep cells with p_min = 5 (prefix [0..4] untouched)\n");
    clear_seq(mem, 0);

    // Decode 8 tokens at positions [0..7]. Then compact the suffix [5..7]
    // keeping positions [5, 7] -> renamed to [5, 6]. Prefix [0..4] stays.
    EXPECT(decode_chain(ctx, 0, 0, 8), "decode 8 chain tokens");
    EXPECT(llama_memory_seq_pos_max(mem, 0) == 7, "pre: pos_max == 7");

    const llama_pos kept[] = { 5, 7 };
    EXPECT(llama_memory_keep_positions_range(mem, 0, kept, 2, /*p_min=*/5),
            "keep_positions_range succeeds");
    EXPECT(llama_memory_seq_pos_min(mem, 0) == 0,
            "post: pos_min == 0 (committed prefix untouched)");
    EXPECT(llama_memory_seq_pos_max(mem, 0) == 6,
            "post: pos_max == 6 (5 -> 5, 7 -> 6, dropped 6)");
}

static void test_sibling_duplicates(llama_context * ctx, llama_memory_t mem) {
    fprintf(stderr, "\n[test_sibling_duplicates] tree-write two cells at same pos, keep first\n");
    clear_seq(mem, 0);

    // Prefix: 3 chain cells at [0,1,2]. Then a tree-mode write of two
    // tokens at the SAME pos (3) — only possible because the bypass is
    // active (set_tree_mask) and both tokens see the prefix via the mask.
    EXPECT(decode_chain(ctx, 0, 0, 3), "decode 3-token prefix");
    EXPECT(llama_memory_seq_pos_max(mem, 0) == 2, "pre-tree: pos_max == 2");

    const llama_pos tree_pos[] = { 3, 3 };
    EXPECT(decode_tree(ctx, 0, tree_pos, 2), "decode 2 sibling tokens at pos=3");
    EXPECT(llama_memory_seq_pos_max(mem, 0) == 3,
            "post-tree: pos_max == 3 (both siblings at same pos)");

    // Compact the suffix [3, ...]: keep position 3, rename to 3. The first
    // cell at pos=3 wins; the duplicate is dropped.
    const llama_pos kept[] = { 3 };
    EXPECT(llama_memory_keep_positions_range(mem, 0, kept, 1, /*p_min=*/3),
            "keep_positions_range succeeds (first-wins on duplicate)");
    EXPECT(llama_memory_seq_pos_min(mem, 0) == 0, "post: prefix preserved");
    EXPECT(llama_memory_seq_pos_max(mem, 0) == 3, "post: pos_max == 3");

    // After compaction the seq is contiguous from 0..3. We can extend it.
    EXPECT(decode_chain(ctx, 0, 4, 1), "post-compact decode at pos=4 succeeds");
    EXPECT(llama_memory_seq_pos_max(mem, 0) == 4, "post-extend: pos_max == 4");
}

static void test_n_zero_edge(llama_context * ctx, llama_memory_t mem) {
    fprintf(stderr, "\n[test_n_zero_edge] n_positions == 0 drops everything in [p_min, +inf)\n");
    clear_seq(mem, 0);

    EXPECT(decode_chain(ctx, 0, 0, 5), "decode 5 chain tokens");

    EXPECT(llama_memory_keep_positions_range(mem, 0, nullptr, 0, /*p_min=*/3),
            "keep_positions_range with n=0 succeeds");
    EXPECT(llama_memory_seq_pos_min(mem, 0) == 0, "post: pos_min == 0");
    EXPECT(llama_memory_seq_pos_max(mem, 0) == 2, "post: pos_max == 2 (cells [3,4] dropped)");

    EXPECT(llama_memory_keep_positions_range(mem, 0, nullptr, 0, /*p_min=*/0),
            "keep_positions_range with n=0, p_min=0 wipes the seq");
    EXPECT(llama_memory_seq_pos_min(mem, 0) == -1, "post: seq is empty (pos_min == -1)");
}

static void test_seq_id_negative(llama_context * ctx, llama_memory_t mem) {
    fprintf(stderr, "\n[test_seq_id_negative] seq_id < 0 is rejected\n");
    clear_seq(mem, 0);

    EXPECT(decode_chain(ctx, 0, 0, 3), "decode 3 chain tokens (seed for the test)");

    const llama_pos kept[] = { 0, 1 };
    EXPECT(!llama_memory_keep_positions_range(mem, /*seq_id=*/-1, kept, 2, /*p_min=*/0),
            "keep_positions_range rejects seq_id == -1");
    // Sanity: state unchanged.
    EXPECT(llama_memory_seq_pos_max(mem, 0) == 2, "state unchanged after rejection");
}

int main(int argc, char ** argv) {
    common_params params;
    params.n_ctx = 128;
    params.n_batch = 32;
    params.n_ubatch = 32;
    params.n_parallel = 1;
    params.kv_unified = true;

    common_init();
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    ggml_backend_load_all();

    common_init_result_ptr llama_init = common_init_from_params(params);

    llama_context * ctx = llama_init->context();
    if (!ctx) {
        fprintf(stderr, "%s: failed to init context\n", __func__);
        return 1;
    }

    llama_memory_t mem = llama_get_memory(ctx);
    if (!mem) {
        fprintf(stderr, "%s: ctx has no memory\n", __func__);
        return 1;
    }

    test_basic_compact_to_zero(ctx, mem);
    test_rename_with_committed_prefix(ctx, mem);
    test_sibling_duplicates(ctx, mem);
    test_n_zero_edge(ctx, mem);
    test_seq_id_negative(ctx, mem);

    fprintf(stderr, "\n==== %d failures ====\n", n_failed);
    return n_failed == 0 ? 0 : 1;
}
