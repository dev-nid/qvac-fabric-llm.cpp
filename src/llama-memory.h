#pragma once

#include "llama.h"

#include <map>
#include <memory>
#include <functional>

struct llama_ubatch;

class llama_batch_allocr;

class llama_io_write_i;
class llama_io_read_i;

struct llama_memory_params {
    // kv cache
    ggml_type type_k;
    ggml_type type_v;

    // use full-size SWA cache
    bool swa_full;
};

enum llama_memory_status {
    LLAMA_MEMORY_STATUS_SUCCESS = 0,
    LLAMA_MEMORY_STATUS_NO_UPDATE,
    LLAMA_MEMORY_STATUS_FAILED_PREPARE,
    LLAMA_MEMORY_STATUS_FAILED_COMPUTE,
};

// helper function for combining the status of two memory contexts
// useful for implementing hybrid memory types (e.g. iSWA)
llama_memory_status llama_memory_status_combine(llama_memory_status s0, llama_memory_status s1);

// helper function for checking if a memory status indicates a failure
bool llama_memory_status_is_fail(llama_memory_status status);

// the interface for managing the memory context during batch processing
// this interface is implemented per memory type. see:
//   - llama_kv_cache_context
//   - llama_kv_cache_iswa_context
//   ...
//
// the only method that should mutate the memory and the memory context is llama_memory_i::apply()
struct llama_memory_context_i {
    virtual ~llama_memory_context_i() = default;

    // consume the current ubatch from the context and proceed to the next one
    // return false if we are done
    virtual bool next() = 0;

    // apply the memory state for the current ubatch to the memory object
    // return false on failure
    virtual bool apply() = 0;

    // get the current ubatch
    virtual const llama_ubatch & get_ubatch() const = 0;

    // get the status of the memory context - used for error handling and checking if any updates would be applied
    virtual llama_memory_status get_status() const = 0;
};

using llama_memory_context_ptr = std::unique_ptr<llama_memory_context_i>;

// general concept of LLM memory
// the KV cache is a type of LLM memory, but there can be other types
struct llama_memory_i {
    // this callback is used to filter out layers that should not be included in the cache
    using layer_filter_cb = std::function<bool(int32_t il)>;

    // this callback is used to specify which layers should reuse memory from other layers
    // return negative value to indicate that the layer il should not reuse memory
    using layer_reuse_cb = std::function<int32_t(int32_t il)>;

    virtual ~llama_memory_i() = default;

    // split the input batch into a set of ubatches and verify that they can fit into the cache
    // return a context object containing the ubatches and memory state required to process them
    // check the llama_memory_context_i::get_status() for the result
    virtual llama_memory_context_ptr init_batch(
            llama_batch_allocr & balloc,
            uint32_t n_ubatch,
            bool embd_all) = 0;

    // simulate full cache, used for allocating worst-case compute buffers
    virtual llama_memory_context_ptr init_full() = 0;

    // prepare for any pending memory updates, such as shifts, copies, etc.
    // status == LLAMA_MEMORY_STATUS_NO_UPDATE if there is nothing to update
    virtual llama_memory_context_ptr init_update(llama_context * lctx, bool optimize) = 0;

    // getters
    virtual bool get_can_shift() const = 0;

    //
    // ops
    //

    // if data == true, the data buffers will also be cleared together with the metadata
    virtual void clear(bool data) = 0;

    virtual bool seq_rm  (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1) = 0;
    virtual void seq_cp  (llama_seq_id seq_id_src, llama_seq_id seq_id_dst, llama_pos p0, llama_pos p1) = 0;
    virtual void seq_keep(llama_seq_id seq_id) = 0;
    virtual void seq_add (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, llama_pos shift) = 0;
    virtual void seq_div (llama_seq_id seq_id,                              llama_pos p0, llama_pos p1, int d) = 0;

    virtual llama_pos seq_pos_min(llama_seq_id seq_id) const = 0;
    virtual llama_pos seq_pos_max(llama_seq_id seq_id) const = 0;

    virtual std::map<ggml_backend_buffer_type_t, size_t> memory_breakdown() const = 0;

    //
    // state write/read
    //

    virtual void state_write(llama_io_write_i & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) const = 0;
    virtual void state_read (llama_io_read_i  & io, llama_seq_id seq_id = -1, llama_state_seq_flags flags = 0) = 0;

    //
    // DFlash tree-aware ubatch write path
    //

    // When tree mode is active on the unified KV cache, apply_ubatch skips
    // the contiguity-invariant purge step (see src/llama-kv-cache.cpp around
    // the seq_pos_max_rm loop): the caller has installed a tree mask via
    // llama_set_tree_mask and accepts ownership of intra-batch position
    // semantics (sibling tree nodes legitimately share positions). After
    // the accept walk, the caller restores contiguity by calling
    // llama_memory_keep_positions_range.
    //
    // Default: no-op. Memory backends without a unified KV cache (recurrent)
    // ignore this; composite backends (hybrid, iswa) propagate to the attn
    // half. Chain-mode callers never set this, so the bypass is dormant.
    virtual void set_tree_mode_active(bool active) { (void) active; }

    // Compact the cache to keep only the listed (seq_id, pos) cells in the
    // suffix [p_min, +inf). Cells with seq_id and pos in positions[0..n-1]
    // are renamed to consecutive positions starting at p_min, in the order
    // supplied. Cells with seq_id and pos >= p_min that are NOT in
    // positions[] are dropped. Cells with pos < p_min are untouched.
    //
    // Used by the tree-mode spec driver after the accept walk: the verify
    // batch wrote tree-depth positions (with sibling duplicates); after
    // the sampler picks the accepted path, this call restores the
    // monotonic-positions invariant the rest of the cache assumes.
    //
    // Edge cases:
    //   - n_positions == 0 -> equivalent to seq_rm(seq_id, p_min, -1)
    //   - seq_id < 0       -> rejected (returns false); tree compaction is
    //                          always single-seq
    //
    // Default: no-op returning true (recurrent half is no-op since the
    // GDN+conv state was already fixed up via the state_select APIs).
    virtual bool keep_positions_range(
            llama_seq_id      seq_id,
            const llama_pos * positions,
            int32_t           n_positions,
            llama_pos         p_min) {
        (void) seq_id; (void) positions; (void) n_positions; (void) p_min;
        return true;
    }

    // Tree-mode compaction by cell-walk ordinal (= DFS allocation order from
    // the immediately-preceding tree-mode ubatch write). Like
    // keep_positions_range, but disambiguates sibling cells that share a
    // position: for each cell with seq_id and pos >= p_min encountered in
    // cell-index order, increment a counter; if counter is in dfs_keep[],
    // keep the cell (renamed to p_min + rank-in-dfs_keep), else drop.
    //
    // dfs_keep[] must be strictly increasing. Tree-mode write is single-seq
    // by construction (seq_id == 0 in the spec driver), seq_id < 0 is
    // rejected.
    //
    // Defaults to no-op = true (recurrent half: GDN+conv state is fixed up
    // via the state_select / conv_state_history_select APIs and needs no
    // separate compaction here).
    virtual bool keep_cells_dfs_ordinals_range(
            llama_seq_id    seq_id,
            const int32_t * dfs_keep,
            int32_t         n_keep,
            llama_pos       p_min) {
        (void) seq_id; (void) dfs_keep; (void) n_keep; (void) p_min;
        return true;
    }
};

using llama_memory_ptr = std::unique_ptr<llama_memory_i>;
