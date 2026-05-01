#pragma once

#include "llama.h"
#include "common.h"

struct common_speculative;

struct common_speculative_params {
    int n_draft = 16;  // max drafted tokens (DRAFT) / DFlash block size (DFLASH; capped at the trained value)
    int n_reuse = 256;

    float p_min = 0.75f; // min probability required to accept a token in the draft (DRAFT only)
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
