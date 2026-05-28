#pragma once

#include "llama.h"
#include "common.h"

#include <unordered_map>
#include <vector>

struct common_speculative;

// tree of likely continuations the DFlash drafter emits in a single forward
// pass, used to feed a tree-shaped target verify batch (one decode = up to
// L+1 accepted tokens for L tree depth)
//
// Indexing convention:
//   * Index 0 is the implicit root (id_last). Carries no entry in
//     `tokens` / `depths` / `branch_ids` (only in `parents` and
//     `child_maps` so the accept walk can start there).
//   * Indices 1..n_nodes are the real tree nodes:
//       tokens[i-1]      = the draft token at index i
//       parents[i]       = the parent index (0 = root, j = node j)
//       depths[i-1]      = 1-based depth from the root
//       branch_ids[i-1]  = which branch this node belongs to (= which
//                          seq_id the spec-simple driver should tag it
//                          with). 0 = main path; 1..n_branches-1 = alts.
//       child_maps[i]    = token → child-node-index map at node i
//   * `parents[0]` is set to -1 by construction.
//   * `visibility` is a row-major `(n_nodes + 1)²` byte matrix:
//       visibility[i*(n+1) + j] = 1 iff node i is allowed to attend to j
//   * `main_path_len` is the length of the chain seed.
//   * `n_branches` is the number of distinct seq_ids in `branch_ids`.
struct common_speculative_tree {
    std::vector<llama_token> tokens;
    std::vector<int>         parents;
    std::vector<int>         depths;
    std::vector<int>         branch_ids;
    std::vector<std::unordered_map<llama_token, int>> child_maps;
    std::vector<uint8_t>     visibility;
    int                      n_nodes       = 0;
    int                      main_path_len = 0;
    int                      n_branches    = 1;

    // parent_ids in row-major [n_tokens, n_seqs] layout
    // (out[t + s * n_tokens] = parent of token t in seq s),
    // matching the layout expected by ggml_gated_delta_net_with_history_tree
    // and ggml_ssm_conv_tree.
    //
    // n_tokens must equal n_nodes + 1 (root + tree nodes). n_seqs is
    // typically 1 for the single-seq DFS-flattened tree-verify batch the
    // spec driver constructs; widened entries (s > 0) are set to -1 so
    // any unused per-seq slabs in the gdn_history buffer treat their
    // pre-block state as the start (kernel TREE_MODE root sentinel).
    //
    // Returns false if `out` is null or the n_tokens argument doesn't
    // match the tree shape.
    bool write_parent_ids(int32_t * out, int n_tokens, int n_seqs) const;
};

// comma separated list of all types
std::string common_speculative_type_name_str();

// convert string to type
enum common_speculative_type common_speculative_type_from_name(const std::string & name);

// convert type to string
std::string common_speculative_type_to_str(enum common_speculative_type type);

common_speculative * common_speculative_init(
        common_params_speculative & params,
        llama_context             * ctx_tgt);

void common_speculative_free(common_speculative * spec);

// optionally call once at the beginning of a new generation
void common_speculative_begin(common_speculative * spec, const llama_tokens & prompt);

// sample up to n_draft tokens and add them to the batch using the draft model
llama_tokens common_speculative_draft(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// DFlash spec-round profile (env DFLASH_PROFILE=1). No-op when unset.
// Accumulates encoder / draft_decode / draft_other ns per round inside
// the DFlash draft() path. Pair with caller-side timers around
// llama_decode(ctx_tgt, ...) to get the full target / draft / driver
// breakdown.
void common_dflash_prof_print();

// tree-shaped variant of common_speculative_draft. Runs the draft once
// (same compute as draft) and returns a tree of candidate continuations
// built from the draft's per-position top-K. The
// caller is responsible for tagging the verify batch with multi-seq seq_ids
// (one per branch_id), running ONE target verify, walking the accept tree
// to pick the longest accepted chain, and the per-branch KV rollback.
//
// Returns an empty tree (n_nodes == 0) for non-DFlash speculative states or
// on any failure path. The caller should fall back to chain-mode in that
// case.
common_speculative_tree common_speculative_draft_tree(
                     common_speculative * spec,
        const common_params_speculative & params,
                     const llama_tokens & prompt,
                            llama_token   id_last);

// alt-accept fast path: tells the speculative state that the most recent
// tree-verify decode's
// accept walk descended into an alt branch at tree index `alt_capture_idx`
// (= the alt's output index in the tree-decode batch) and tree depth
// `alt_depth` (1-based). The next draft_tree call's side-store extend will
// pull captures `[0, 1, ..., alt_depth - 1, alt_capture_idx]` instead of
// the linear `[0..alt_depth]` slice — this eliminates the
// "redecode `[id_last, m_1, ..., m_{d-1}, alt_token]` round-trip" the
// driver would otherwise need.
//
// Default impl is a no-op (DRAFT speculative state has no captures buffer).
void common_speculative_record_alt_accept(
        common_speculative * spec,
        int                  alt_capture_idx,
        int                  alt_depth);

// informs the speculative decoder that n_accepted tokens were accepted by the target model
void common_speculative_accept(common_speculative * spec, uint16_t n_accepted);

int32_t common_speculative_n_max(const common_speculative * spec, const common_params_speculative & params);
int32_t common_speculative_n_min(const common_speculative * spec, const common_params_speculative & params);

// print statistics about the speculative decoding
void common_speculative_print_stats(const common_speculative * spec);

struct common_speculative_deleter {
    void operator()(common_speculative * s) { common_speculative_free(s); }
};

typedef std::unique_ptr<common_speculative, common_speculative_deleter> common_speculative_ptr;
