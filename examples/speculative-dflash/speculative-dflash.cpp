// llama-speculative-dflash: end-to-end demo of DFlash block-parallel speculative
// decoding ported from the dflash Python package (z-lab/dflash) into llama.cpp.
//
// Usage:
//
//     llama-speculative-dflash \
//         -m  Qwen3-8B.gguf \
//         -md Qwen3-8B-DFlash.gguf \
//         -p  "How many positive whole-number divisors does 196 have?" \
//         --n-predict 256 \
//         --dflash-block-size 16
//
// The DFlash draft GGUF must have been produced by the converter additions
// landed alongside this file (DFlashModel in convert_hf_to_gguf.py).

#include "arg.h"
#include "common.h"
#include "dflash-speculative.h"
#include "log.h"
#include "sampling.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int main(int argc, char ** argv) {
    common_params params;

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required (pass a DFlash draft GGUF)\n", __func__);
        return 1;
    }

    common_init();

    llama_backend_init();
    llama_numa_init(params.numa);

    // -----------------------------------------------------------------
    // Load the target model
    // -----------------------------------------------------------------
    common_init_result llama_init_tgt = common_init_from_params(params);
    llama_model   * model_tgt = llama_init_tgt.model.get();
    llama_context * ctx_tgt   = llama_init_tgt.context.get();

    if (model_tgt == nullptr || ctx_tgt == nullptr) {
        LOG_ERR("%s: failed to load target model\n", __func__);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // -----------------------------------------------------------------
    // Load the draft (DFlash) model
    // -----------------------------------------------------------------
    {
        params.devices      = params.speculative.devices;
        params.model        = params.speculative.model;
        params.n_ctx        = params.speculative.n_ctx;
        params.n_batch      = params.speculative.n_ctx > 0 ? params.speculative.n_ctx : params.n_batch;
        params.n_gpu_layers = params.speculative.n_gpu_layers;

        if (params.speculative.cpuparams.n_threads > 0) {
            params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
        }
        params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
        params.tensor_buft_overrides     = params.speculative.tensor_buft_overrides;
    }
    common_init_result llama_init_dft = common_init_from_params(params);
    llama_context * ctx_dft = llama_init_dft.context.get();

    if (ctx_dft == nullptr) {
        LOG_ERR("%s: failed to load draft model\n", __func__);
        return 1;
    }

    if (!common_dflash_speculative_are_compatible(ctx_tgt, ctx_dft)) {
        LOG_ERR("%s: draft '%s' is not a compatible DFlash draft for target '%s'\n",
                __func__, params.speculative.model.path.c_str(), params.model.path.c_str());
        return 1;
    }

    // -----------------------------------------------------------------
    // Tokenize prompt
    // -----------------------------------------------------------------
    std::vector<llama_token> prompt_ids = common_tokenize(ctx_tgt, params.prompt, /*add_special=*/true, /*parse_special=*/true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) prompt_ids.size()) {
        LOG_ERR("%s: prompt exceeds target context (%d tokens, ctx %d)\n",
            __func__, (int) prompt_ids.size(), llama_n_ctx(ctx_tgt));
        return 1;
    }

    LOG("\n\n");
    for (auto id : prompt_ids) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    // -----------------------------------------------------------------
    // Run speculative decoding
    // -----------------------------------------------------------------
    common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    auto * spec = common_dflash_speculative_init(ctx_tgt, ctx_dft);

    common_dflash_speculative_params spec_params;
    spec_params.block_size    = params.speculative.n_max > 0 ? params.speculative.n_max : 0;  // 0 = use draft default
    spec_params.n_max_predict = params.n_predict > 0 ? params.n_predict : 256;
    spec_params.use_color     = params.use_color;

    common_dflash_speculative_callbacks cbs;
    cbs.user_data = ctx_tgt;
    cbs.on_token  = [](llama_token id, void * ud) -> bool {
        auto * ctx = (llama_context *) ud;
        const std::string s = common_token_to_piece(ctx, id);
        LOG("%s", s.c_str());
        fflush(stdout);
        return true;
    };

    common_dflash_speculative_stats stats {};
    auto out = common_dflash_speculative_generate(
        spec, spec_params, prompt_ids,
        /*eos_ids=*/{ llama_vocab_eos(vocab) },
        smpl,
        cbs,
        &stats);

    LOG("\n\n");

    // -----------------------------------------------------------------
    // Stats
    // -----------------------------------------------------------------
    LOG_INF("encoded %4d prompt tokens\n", stats.n_input);
    LOG_INF("decoded %4d tokens in %8.3f s, speed: %8.3f t/s\n",
        stats.n_predict, stats.t_decode_s,
        stats.n_predict / std::max(1e-6, stats.t_decode_s));
    LOG_INF("\n");
    LOG_INF("n_blocks   = %d\n", stats.n_blocks);
    LOG_INF("n_drafted  = %d\n", stats.n_drafted);
    LOG_INF("n_accept   = %d\n", stats.n_accept);
    LOG_INF("acceptance = %.3f%%\n",
        stats.n_drafted > 0 ? 100.0 * stats.n_accept / stats.n_drafted : 0.0);

    common_dflash_speculative_free(spec);
    common_sampler_free(smpl);

    llama_backend_free();
    return 0;
}
