#include "arg.h"
#include "common.h"
#include "dflash-profile.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>
#include <utility>

struct spec_checkpoint {
    int64_t n_tokens = 0;

    std::vector<uint8_t> data;

    size_t size() const {
        return data.size();
    }

    bool empty() const {
        return data.empty();
    }
};

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SPECULATIVE)) {
        return 1;
    }

    if (params.n_predict < -1) {
        LOG_ERR("%s: --n-predict must be >= -1\n", __func__);
        return 1;
    }

    // DFlash per-op profiler (opt-in via DFLASH_PROF=1, see common/dflash-profile.h).
    const auto dflash_prof_cfg = dflash_prof::read_env_config();
    if (dflash_prof_cfg.enabled) {
        auto & prof = dflash_prof::profiler::instance();
        prof.set_enabled(true);
        params.cb_eval           = dflash_prof::profiler::eval_callback;
        params.cb_eval_user_data = &prof;
        params.warmup            = false;
        LOG_INF("%s: DFLASH_PROF=1: per-op profiling enabled (CPU-only; "
                "single-node compute distorts absolute throughput)\n", __func__);
    }

    // Companion memory profiler (opt-in via DFLASH_MEM=1).
    const auto dflash_mem_cfg = dflash_prof::read_mem_env_config();
    if (dflash_mem_cfg.enabled) {
        dflash_prof::memory_profiler::instance().set_enabled(true);
        dflash_prof::memory_profiler::instance().snapshot("before_load", nullptr, nullptr);
        LOG_INF("%s: DFLASH_MEM=1: memory profiling enabled\n", __func__);
    }

    if (params.speculative.draft.mparams.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // DDTree (DFlash Phase 2): when --dflash-tree is set, bump n_parallel
    // and enable kv_unified BEFORE target context creation. Tree-mode
    // verify uses one seq_id per branch, all sharing a single KV stream.
    if (params.speculative.dflash_tree) {
        const int budget = params.speculative.dflash_tree_budget;
        // Worst-case n_branches with K=2 uniform expansion is 1 + budget;
        // we don't know block_size yet (draft model isn't loaded), so we
        // use this upper bound. Over-allocating n_seq_max is cheap.
        const int min_parallel_tree = std::max(2, (budget > 0) ? budget + 1 : 2);
        if (params.n_parallel < min_parallel_tree) {
            LOG_INF("%s: --dflash-tree (budget=%d): bumping n_parallel from %d to %d "
                    "so ctx_tgt's n_seq_max fits all alt-branch seq_ids\n",
                    __func__, budget, params.n_parallel, min_parallel_tree);
            params.n_parallel = min_parallel_tree;
        }
        if (!params.kv_unified) {
            LOG_INF("%s: --dflash-tree: enabling kv_unified (required for multi-seq tree-verify pattern)\n",
                    __func__);
            params.kv_unified = true;
        }
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;

    // load the target model
    auto llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt->model();
    ctx_tgt   = llama_init_tgt->context();

    // check if the context supports partial sequence removal
    const auto ctx_seq_rm = common_context_can_seq_rm(ctx_tgt);
    const bool use_ckpt = (ctx_seq_rm == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    if (use_ckpt) {
        LOG_INF("speculative decoding will use checkpoints (context does not support partial sequence removal)\n");
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // load the draft model
    llama_model_ptr model_dft;

    // TODO: simplify this logic
    {
        const auto & params_spec = params.speculative.draft;

        auto params_dft = params;

        params_dft.n_parallel   = 1;
        params_dft.n_ctx        = params_spec.n_ctx;
        params_dft.n_batch      = llama_n_ctx_seq(ctx_tgt);
        params_dft.devices      = params_spec.devices;
        params_dft.model        = params_spec.mparams;
        params_dft.n_gpu_layers = params_spec.n_gpu_layers;

        if (params_spec.cpuparams.n_threads > 0) {
            params_dft.cpuparams.n_threads       = params.speculative.draft.cpuparams.n_threads;
            params_dft.cpuparams_batch.n_threads = params.speculative.draft.cpuparams_batch.n_threads;
        }

        params_dft.tensor_buft_overrides = params.speculative.draft.tensor_buft_overrides;

        auto mparams_dft = common_model_params_to_llama(params_dft);

        model_dft.reset(llama_model_load_from_file(params_dft.model.path.c_str(), mparams_dft));
        if (model_dft == nullptr) {
            LOG_ERR("failed to load draft model, '%s'\n", params_dft.model.path.c_str());
            return 1;
        }

        params.speculative.draft.model = model_dft.get();
        params.speculative.draft.cparams = common_context_params_to_llama(params_dft);

        // Forward DFlash-specific cparams to the draft context.
        params.speculative.draft.cparams.dflash_max_ctx = params.speculative.dflash_max_ctx;
        params.speculative.draft.cparams.dflash_topk    = params.speculative.dflash_topk;
    }

    // Tokenize the prompt
    std::vector<llama_token> inp;
    inp = common_tokenize(ctx_tgt, params.prompt, true, true);

    if (llama_n_ctx(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the context size (%d tokens, ctx %d)\n", __func__, (int) inp.size(), llama_n_ctx(ctx_tgt));

        return 1;
    }

    if (llama_n_batch(ctx_tgt) < (uint32_t) inp.size()) {
        LOG_ERR("%s: the prompt exceeds the batch size (%d tokens, batch %d)\n", __func__, (int) inp.size(), llama_n_batch(ctx_tgt));

        return 1;
    }

    LOG("\n\n");

    for (auto id : inp) {
        LOG("%s", common_token_to_piece(ctx_tgt, id).c_str());
    }

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // used to determine end of generation
    bool has_eos = false;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    common_sampler_ptr smpl(common_sampler_init(model_tgt, params.sampling));

    // eval the prompt
    llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    // init the speculator
    const auto & params_spec = params.speculative;

    if (dflash_mem_cfg.enabled) {
        dflash_prof::memory_profiler::instance().snapshot("after_init", nullptr, ctx_tgt);
    }
    if (dflash_prof_cfg.enabled) dflash_prof::profiler::instance().set_phase("prompt");

    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);

    common_speculative_begin(spec, prompt_tgt);

    if (dflash_mem_cfg.enabled) {
        dflash_prof::memory_profiler::instance().snapshot("after_prefill", nullptr, ctx_tgt);
    }

    // Tree mode multi-seq: each batch token may carry up to n_branches
    // seq_ids, so size the per-slot seq_id arrays accordingly. Chain mode
    // and Stage B keep the historical size 1; Stage C with budget B has
    // n_branches <= 1 + B.
    const int batch_n_seq_max = params.speculative.dflash_tree
        ? std::max(2, params.speculative.dflash_tree_budget + 1)
        : 1;
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, batch_n_seq_max);

    size_t n_draft = 0;

    llama_tokens draft;
    spec_checkpoint spec_ckpt;

    // DDTree Phase 2 tree-mode statistics.
    int64_t n_tree_iters     = 0;
    int64_t n_tree_alt_taken = 0;

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    const bool use_tree = params.speculative.dflash_tree;

    while (true) {
        if (dflash_prof_cfg.enabled) dflash_prof::profiler::instance().set_phase("draft");
        // ============================================================
        // DDTree Phase 2: multi-seq tree-shaped verify branch
        // ============================================================
        // 1. Build a small tree from the draft's per-position top-K
        //    (Stage B: chain seed + 1 alt at depth 1; Stage C with
        //    --dflash-tree-budget B: chain seed + uniform round-robin
        //    sibling expansion until B nodes total).
        // 2. Multi-seq batch construction: seq_cp(0->b) broadcasts the
        //    prefix to alt branches; main-path nodes carry seqs
        //    {0} ∪ {alt branches at deeper depths}; alt nodes carry
        //    only their own branch_id.
        // 3. Decode (no tree mask — seq-id-based mask handles separation).
        // 4. Tree-walk accept (root → child by target argmax, repeat).
        // 5. Rollback: drop alt seqs; for main-accept drop rejected
        //    main-path tail; for alt-accept use commit-35 KV surgery
        //    (seq_rm/seq_cp metadata-only) + record_alt_accept hint.
        if (use_tree) {
            common_speculative_tree tree =
                common_speculative_draft_tree(spec, params_spec, prompt_tgt, id_last);

            const int n_past_before = n_past;
            auto * mem = llama_get_memory(ctx_tgt);

            common_batch_clear(batch_tgt);
            llama_tokens ids;

            if (tree.n_nodes == 0) {
                // Drafter returned no tree (non-DFlash, or topk read failed).
                // Decode just id_last and emit the bonus token.
                common_batch_add(batch_tgt, id_last, n_past_before, { 0 }, /*logits=*/true);
                llama_decode(ctx_tgt, batch_tgt);

                const llama_token bonus = common_sampler_sample(smpl.get(), ctx_tgt, 0);
                common_sampler_accept(smpl.get(), bonus, true);
                ids.push_back(bonus);
                n_past = n_past_before + 1;
            } else {
                // Multi-seq prefix tagging: copy seq 0 -> seq b for each alt branch.
                for (int b = 1; b < tree.n_branches; ++b) {
                    llama_memory_seq_cp(mem, 0, b, -1, -1);
                }

                // For each main-path node at depth d, compute which alt seqs
                // it must additionally carry (every alt at deeper depths).
                std::vector<std::vector<llama_seq_id>> main_extra_seqs(tree.main_path_len);
                for (int i = 0; i < tree.n_nodes; ++i) {
                    if (tree.branch_ids[i] == 0) continue;
                    const int alt_depth = tree.depths[i];
                    const llama_seq_id alt_seq = (llama_seq_id) tree.branch_ids[i];
                    for (int d_main = 1; d_main < alt_depth; ++d_main) {
                        main_extra_seqs[d_main - 1].push_back(alt_seq);
                    }
                }

                // id_last (root) — visible from every branch.
                std::vector<llama_seq_id> root_seqs(tree.n_branches);
                for (int b = 0; b < tree.n_branches; ++b) root_seqs[b] = b;
                common_batch_add(batch_tgt, id_last, n_past_before, root_seqs, /*logits=*/true);

                // Tree nodes.
                for (int i = 0; i < tree.n_nodes; ++i) {
                    const int d  = tree.depths[i];
                    const int bi = tree.branch_ids[i];
                    std::vector<llama_seq_id> seqs;
                    if (bi == 0) {
                        seqs.reserve(1 + main_extra_seqs[d - 1].size());
                        seqs.push_back(0);
                        for (auto s : main_extra_seqs[d - 1]) seqs.push_back(s);
                    } else {
                        seqs.push_back((llama_seq_id) bi);
                    }
                    common_batch_add(batch_tgt, tree.tokens[i],
                                     n_past_before + d, seqs, /*logits=*/true);
                }

                if (dflash_prof_cfg.enabled) dflash_prof::profiler::instance().set_phase("verify");
                llama_decode(ctx_tgt, batch_tgt);

                // Tree-walk accept: root → ... → commit_n following target argmax.
                int current  = 0;
                int commit_n = 0;
                while (true) {
                    const llama_token target_token =
                        common_sampler_sample(smpl.get(), ctx_tgt, current);
                    common_sampler_accept(smpl.get(), target_token, true);
                    ids.push_back(target_token);

                    auto it = tree.child_maps[current].find(target_token);
                    if (it == tree.child_maps[current].end()) {
                        break; // bonus token
                    }
                    current  = it->second;
                    commit_n = current;
                }

                const bool alt_accepted = (commit_n > tree.main_path_len);

                if (alt_accepted) {
                    n_tree_alt_taken++;

                    // Stage C alt-accept: KV surgery (no compute).
                    const int          d          = tree.depths[commit_n - 1];
                    const llama_seq_id alt_branch = (llama_seq_id) tree.branch_ids[commit_n - 1];

                    // (a) Drop main-path tail m_d..m_main_path_len from seq 0.
                    llama_memory_seq_rm(mem, 0, n_past_before + d, -1);
                    // (b) Promote alt slot at N0+d into seq 0 (metadata-only under kv_unified).
                    llama_memory_seq_cp(mem, alt_branch, 0,
                                        n_past_before + d,
                                        n_past_before + d + 1);
                    // (c) Drop every alt seq.
                    for (int b = 1; b < tree.n_branches; ++b) {
                        llama_memory_seq_rm(mem, b, -1, -1);
                    }
                    // (d) Hint the spec state to remap captures on next extend.
                    common_speculative_record_alt_accept(spec,
                        /*alt_capture_idx=*/commit_n, /*alt_depth=*/d);

                    n_past = n_past_before + d + 1;
                } else {
                    // Main-path accept (commit_n == 0 or 1..main_path_len).
                    const int L = (commit_n > 0) ? tree.depths[commit_n - 1] : 0;
                    llama_memory_seq_rm(mem, 0, n_past_before + 1 + L, -1);
                    for (int b = 1; b < tree.n_branches; ++b) {
                        llama_memory_seq_rm(mem, b, -1, -1);
                    }
                    n_past = n_past_before + 1 + L;
                }
            }

            n_tree_iters++;
            common_speculative_accept(spec, ids.empty() ? 0 : (uint16_t)(ids.size() - 1));

            n_drafted += tree.n_nodes;
            n_accept  += ids.empty() ? 0 : (int) ids.size() - 1;

            // Cap at exactly params.n_predict tokens (per-token check) so
            // byte-exact match against the greedy reference holds.
            const int n_predict_pre_loop = n_predict;
            for (size_t i = 0; i < ids.size(); ++i) {
                if (params.n_predict >= 0 && (n_predict_pre_loop + (int) i) >= params.n_predict) {
                    has_eos = true;
                    break;
                }
                prompt_tgt.push_back(id_last);
                id_last = ids[i];
                ++n_predict;
                if (llama_vocab_is_eog(vocab, id_last)) {
                    has_eos = true;
                    break;
                }
                const std::string token_str = common_token_to_piece(ctx_tgt, id_last);
                if (params.use_color && i + 1 < ids.size()) {
                    LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
                } else {
                    LOG("%s", token_str.c_str());
                }
            }

            if ((params.n_predict >= 0 && n_predict >= params.n_predict) || has_eos) {
                break;
            }
            continue;
        }

        // generate or reuse draft tokens
        //
        // this is the most important part of the speculation. the more probable tokens that are provided here
        // the better the performance will be. in theory, this computation can be performed asynchronously and even
        // offloaded to a remote device. it doesn't even have to be based on an LLM. instead, it can provide tokens
        // from a cache or lookup tables.
        //
        if (draft.empty()) {
            // generate a new draft
            draft = common_speculative_draft(spec, params_spec, prompt_tgt, id_last);

            // save the original draft size
            n_draft = draft.size();

            // save a checkpoint of the target context before evaluating the draft
            // this allows us to restore the state if partial draft acceptance occurs
            if (!draft.empty() && use_ckpt) {
                const size_t ckpt_size = llama_state_seq_get_size_ext(ctx_tgt, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                spec_ckpt.data.resize(ckpt_size);

                const size_t n = llama_state_seq_get_data_ext(ctx_tgt, spec_ckpt.data.data(), ckpt_size, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
                GGML_ASSERT(n == ckpt_size);

                spec_ckpt.n_tokens = (int64_t) prompt_tgt.size();
                LOG_DBG("created speculative checkpoint (n_tokens = %" PRId64 ", size = %.3f MiB)\n",
                        spec_ckpt.n_tokens, (float) spec_ckpt.data.size() / 1024 / 1024);
            }
        } else {
            // we have a previous (partial) draft to reuse from checkpoint restoration
            if (use_ckpt) {
                GGML_ASSERT(!spec_ckpt.empty());
            }
        }

        // always have a token to evaluate from before - id_last
        common_batch_clear(batch_tgt);
        common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            if (dflash_prof_cfg.enabled) dflash_prof::profiler::instance().set_phase("verify");
            llama_decode(ctx_tgt, batch_tgt);
        }

        // only save the sampler sampler state if we use checkpoints
        common_sampler_ptr smpl_save;
        if (use_ckpt) {
            smpl_save.reset(common_sampler_clone(smpl.get()));
        }

        // sample from the full target batch and return the accepted tokens based on the target sampler
        //
        // for each token to be accepted, the sampler would have to sample that same token
        // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
        // available logits from the batch and sample the next token until we run out of logits or the sampler
        // disagrees with the draft
        //
        auto ids = common_sampler_sample_and_accept_n(smpl.get(), ctx_tgt, draft);

        //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        // check for partial draft acceptance:
        // if the context doesn't support partial sequence removal, restore the checkpoint
        // and make the accepted tokens the new partial draft for the next iteration
        if (use_ckpt && ids.size() - 1 < draft.size()) {
            LOG_DBG("partial acceptance: %zu < %zu, restoring checkpoint\n", ids.size() - 1, draft.size());

            draft = std::move(ids);

            const size_t n = llama_state_seq_set_data_ext(ctx_tgt, spec_ckpt.data.data(), spec_ckpt.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            GGML_ASSERT(n == spec_ckpt.size());

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, spec_ckpt.n_tokens, -1);

            prompt_tgt.resize(spec_ckpt.n_tokens);
            smpl = std::move(smpl_save);

            n_past = (int) prompt_tgt.size();

            continue;
        }

        common_speculative_accept(spec, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;

        // Cap at exactly params.n_predict tokens (per-token check inside
        // the loop) so byte-exact comparison against the greedy reference
        // holds. Without this, the last accepted draft chunk can overshoot
        // by up to ids.size() - 1 tokens.
        const int n_predict_pre_loop = n_predict;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (params.n_predict >= 0 && (n_predict_pre_loop + (int) i) >= params.n_predict) {
                has_eos = true;
                break;
            }

            prompt_tgt.push_back(id_last);

            id_last = ids[i];
            ++n_predict;

            if (llama_vocab_is_eog(vocab, id_last)) {
                has_eos = true;
                break;
            }

            const std::string token_str = common_token_to_piece(ctx_tgt, id_last);

            if (params.use_color && i + 1 < ids.size()) {
                LOG("\u001b[%dm%s\u001b[37m", (36 - 0 % 6), token_str.c_str());
            } else {
                LOG("%s", token_str.c_str());
            }
        }

        LOG_DBG("accepted %d/%d draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, (int) draft.size(), id_last);

        // clear the draft since it has been consumed
        draft.clear();

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        }

        if ((params.n_predict >= 0 && n_predict >= params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    if (dflash_mem_cfg.enabled) {
        dflash_prof::memory_profiler::instance().snapshot("after_gen", nullptr, ctx_tgt);
    }

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("n_draft   = %d\n", params_spec.draft.n_max);
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);
    if (use_tree) {
        LOG_INF("tree iters    = %lld\n", (long long) n_tree_iters);
        LOG_INF("tree alt taken = %lld\n", (long long) n_tree_alt_taken);
    }

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl.get());

    llama_batch_free(batch_tgt);

    common_speculative_free(spec);

    if (dflash_prof_cfg.enabled) {
        FILE * out = stderr;
        if (!dflash_prof_cfg.out_file.empty()) {
            FILE * f = std::fopen(dflash_prof_cfg.out_file.c_str(), "w");
            if (f) out = f;
        }
        dflash_prof::profiler::instance().dump(out, dflash_prof_cfg.top_n);
        if (out != stderr) std::fclose(out);
    }
    if (dflash_mem_cfg.enabled) {
        FILE * out = stderr;
        if (!dflash_mem_cfg.out_file.empty()) {
            FILE * f = std::fopen(dflash_mem_cfg.out_file.c_str(), "w");
            if (f) out = f;
        }
        dflash_prof::memory_profiler::instance().dump(out);
        if (out != stderr) std::fclose(out);
    }

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
