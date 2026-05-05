#pragma once

#include "llama.h"
#include "common.h"

#include <cstdint>
#include <unordered_map>
#include <vector>

struct common_speculative;

struct common_speculative_params {
    int n_draft = 16;  // max drafted tokens (DRAFT) / DFlash block size (DFLASH; capped at the trained value)
    int n_reuse = 256;

    float p_min = 0.75f; // min probability required to accept a token in the draft (DRAFT only)
};

// DDTree (DFlash Phase 2 Stage B): tree of likely continuations the DFlash
// drafter emits in a single forward pass, used to feed a tree-shaped target
// verify batch (one decode = up to L+1 accepted tokens for L tree depth).
//
// Indexing convention (mirrors buun fork's representation; written from
// scratch):
//   * Index 0 is the implicit root (id_last). It carries no entry in
//     `tokens` / `depths` (only in `parents` and `child_maps` so the
//     accept walk can start there).
//   * Indices 1..n_nodes are the real tree nodes:
//       tokens[i-1]      = the draft token at index i
//       parents[i]       = the parent index (0 = root, j = node j)
//       depths[i-1]      = 1-based depth from the root (depth 1 = direct
//                          child of root; depth 2 = grandchild; ...)
//       child_maps[i]    = token → child-node-index map at node i (used by
//                          the accept walk for O(1) "does target's argmax
//                          match a child of `current`?" lookup)
//   * `parents[0]` is set to -1 by construction.
//   * `visibility` is a row-major `(n_nodes + 1)²` byte matrix:
//       visibility[i*(n+1) + j] = 1 iff node i is allowed to attend to
//                                       node j (parent-pointer reachability)
//     This is what `llama_set_tree_mask` consumes verbatim.
//   * `main_path_len` is the length of the chain seed (= number of nodes on
//     the top-1 chain from depth 1). Useful for telemetry; not load-bearing.
struct common_speculative_tree {
    std::vector<llama_token> tokens;
    std::vector<int>         parents;
    std::vector<int>         depths;
    std::vector<std::unordered_map<llama_token, int>> child_maps;
    std::vector<uint8_t>     visibility;
    int                      n_nodes       = 0;
    int                      main_path_len = 0;
};

// Initialise a speculative decoder using the AUTO algorithm picker:
// inspect the draft model and pick DFLASH if it carries DFlash GGUF metadata
// (`llama_model_dflash_block_size(model_dft) > 0`), otherwise DRAFT.
struct common_speculative * common_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft
);

// Same as common_speculative_init, but lets the caller force a specific
// algorithm (or pass COMMON_SPECULATIVE_TYPE_AUTO to keep auto-detect).
// Returns nullptr if the requested algorithm is incompatible with the
// loaded draft model (e.g. DFLASH on a non-DFlash draft).
struct common_speculative * common_speculative_init_typed(
        struct llama_context *      ctx_tgt,
        struct llama_context *      ctx_dft,
        enum common_speculative_type type);

void common_speculative_free(struct common_speculative * spec);

bool common_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft);

void common_speculative_add_replacement_tgt_dft(
        struct common_speculative * spec,
        const char *source, const char *dest);

// Run the prompt prefill on the *target* context.
//
// For COMMON_SPECULATIVE_TYPE_DRAFT this is functionally
//     llama_decode(ctx_tgt, llama_batch_get_one(prompt[0..n-1]))
// i.e. the last prompt token is left unconsumed and the caller's verify
// loop is expected to commit `prompt.back()` (= id_last) on the next decode.
//
// For COMMON_SPECULATIVE_TYPE_DFLASH this runs an *all-positions-logits*
// prefill on `prompt[0..n-1]` (required for full DFlash capture coverage)
// and then extends the draft's K/V side store with the captured target
// features for those positions. The caller's verify loop still commits
// `prompt.back()` (= id_last) on the next decode, exactly the same as
// for DRAFT — the only externally-visible difference is that the prompt
// prefill cost is paid up-front via this call rather than being deferred
// to the first verify batch.
//
// Returns true on success, false on llama_decode / dflash_extend failure.
bool common_speculative_target_prefill(
        struct common_speculative * spec,
        const llama_tokens &        prompt);

// Returns the speculative-decoding algorithm that this spec is using
// (after AUTO has been resolved). Useful for callers that need to size
// `--draft-min`-style batch logic per algorithm.
enum common_speculative_type common_speculative_get_type(
        const struct common_speculative * spec);

// sample up to n_draft tokens and add them to the batch using the draft model
//
// For DRAFT: returns up to `params.n_draft` tokens sampled greedily from
//            the draft model's per-position logits.
// For DFLASH: returns exactly `params.n_draft` tokens (= block_size - 1),
//             being the draft model's argmax predictions at intra-block
//             positions [1..block_size-1]. The caller's verify batch must
//             then prepend `id_last` and decode all `block_size` positions
//             on the target.
llama_tokens common_speculative_gen_draft(
               struct common_speculative * spec,
        struct common_speculative_params   params,
                      const llama_tokens & prompt,
                             llama_token   id_last);

// DDTree (DFlash Phase 2 Stage B): tree-shaped variant of
// common_speculative_gen_draft. Runs the draft once (same compute as
// gen_draft, same K/V side store extend) and returns a tree of candidate
// continuations built from the draft's per-position top-K. The caller is
// responsible for installing the resulting visibility matrix via
// `llama_set_tree_mask` before the verify decode and clearing it after,
// for the tree-walk accept logic, and for the post-verify KV rollback
// (drop verify slots + re-decode the accepted chain in causal mode).
//
// Returns an empty tree (n_nodes == 0) on any failure path or for
// non-DFlash speculative states. The caller should fall back to
// chain-mode in that case.
//
// Stage B scope: tree shape is "main path (top-1 at each depth) + one
// alternate at depth 1 (top-2 at draft position 1)" — total n_nodes =
// block_size, total visibility size = (block_size + 1)² bytes. Stage C
// will generalise the shape via a budget parameter.
common_speculative_tree common_speculative_gen_draft_tree(
               struct common_speculative * spec,
        struct common_speculative_params   params,
                      const llama_tokens & prompt,
                             llama_token   id_last);
