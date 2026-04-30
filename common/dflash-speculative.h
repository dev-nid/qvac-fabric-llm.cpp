#pragma once

#include "llama.h"
#include "common.h"

#include <string>
#include <vector>

// -------------------------------------------------------------------------
// DFlash speculative-decoding driver
// -------------------------------------------------------------------------
//
// DFlash is a "block diffusion" speculative-decoding method: a small draft
// model produces an entire block of tokens in a single forward pass (using
// mask tokens), conditioned on the target model's intermediate hidden
// states. The target model then verifies the block in parallel; the
// longest matching prefix is accepted plus one bonus token from the
// target.
//
// Reference (Python): dflash/dflash/model_mlx.py: stream_generate
// Reference (Python): dflash/dflash/model.py:    dflash_generate
//
// Usage:
//
//     auto * spec = common_dflash_speculative_init(ctx_tgt, ctx_dft);
//     common_dflash_speculative_params p;
//     p.block_size = 16;
//     auto out = common_dflash_speculative_generate(spec, p, prompt, eos_ids,
//                                                   smpl_tgt);
//     common_dflash_speculative_free(spec);
//

struct common_dflash_speculative;

struct common_dflash_speculative_params {
    int   block_size = 0;          // 0 = use the value stored in the draft GGUF
    int   n_max_predict = 256;     // upper bound on generated tokens (post-prompt)
    bool  use_color = false;
};

struct common_dflash_speculative_callbacks {
    // Optional callback fired after each accepted token. If it returns false,
    // generation halts immediately. user_data is forwarded.
    bool (*on_token)(llama_token id, void * user_data) = nullptr;
    void * user_data = nullptr;
};

struct common_dflash_speculative_stats {
    int n_blocks      = 0;
    int n_drafted     = 0;     // total tokens drafted (excluding the leading committed token)
    int n_accept      = 0;     // total drafted tokens accepted by target
    int n_predict     = 0;     // total tokens emitted post-prompt (prompt excluded)
    double t_decode_s = 0.0;   // wall-clock time in the decode loop (seconds)
    int    n_input    = 0;     // number of prompt tokens
};

struct common_dflash_speculative * common_dflash_speculative_init(
        struct llama_context * ctx_tgt,
        struct llama_context * ctx_dft);

void common_dflash_speculative_free(struct common_dflash_speculative * spec);

// Compatibility check between target and draft.
// Returns true if the two models share the same vocabulary AND the draft is
// a DFlash architecture (i.e. its GGUF carries the dflash.* metadata).
bool common_dflash_speculative_are_compatible(
        const struct llama_context * ctx_tgt,
        const struct llama_context * ctx_dft);

// Run end-to-end DFlash speculative decoding.
// Inputs:
//   - prompt:        tokens already in the target vocab (typically the result
//                    of common_tokenize on the user's prompt).
//   - eos_ids:       early-stop tokens.
//   - smpl_tgt:      target sampler (e.g. created by common_sampler_init);
//                    used to sample the verification tokens. The draft
//                    sampler is fixed at greedy/argmax in v1.
// Output:
//   - returns the full token sequence INCLUDING the prompt, with the new
//     tokens appended. EOS / stop tokens (if any) are *included* at the end.
llama_tokens common_dflash_speculative_generate(
        struct common_dflash_speculative * spec,
        struct common_dflash_speculative_params params,
        const llama_tokens & prompt,
        const std::vector<llama_token> & eos_ids,
        struct common_sampler * smpl_tgt,
        struct common_dflash_speculative_callbacks cbs = {},
        struct common_dflash_speculative_stats * stats_out = nullptr);
