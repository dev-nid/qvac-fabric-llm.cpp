#include "arg.h"
#include "chat.h"
#include "common.h"
#include "dflash-gpu-log.h"
#include "dflash-profile.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <clocale>
#include <cstdio>
#include <cstdlib>
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

    // GPU-aware host-observed wall-time logger (opt-in via DFLASH_GPU_LOG=1).
    // Designed for Vulkan / Metal / CUDA where the dflash-profile cb_eval
    // path would distort timings. See common/dflash-gpu-log.h.
    const auto dflash_gpu_log_cfg = dflash_gpu_log::read_env_config();
    if (dflash_gpu_log_cfg.enabled) {
        dflash_gpu_log::logger::instance().set_enabled(true);
        LOG_INF("%s: DFLASH_GPU_LOG=1: GPU host-observed phase logger enabled\n", __func__);
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
    bool use_ckpt = (ctx_seq_rm == COMMON_CONTEXT_SEQ_RM_TYPE_FULL);

    // Experimental: disable the checkpoint+re-verify path on partial acceptance
    // even for FULL-only seq_rm contexts (e.g. Qwen3.5 GatedDeltaNet). The
    // verify decode advanced the SSM state through all draft positions, and
    // skipping the re-decode leaves the SSM state polluted by rejected
    // tokens; downstream quality may drift. Enabled by DFLASH_SKIP_CKPT=1
    // for back-to-back comparison against the default behaviour.
    if (use_ckpt) {
        const char * e = std::getenv("DFLASH_SKIP_CKPT");
        if (e && e[0] != '\0' && std::strcmp(e, "0") != 0) {
            use_ckpt = false;
            LOG_INF("DFLASH_SKIP_CKPT=1: bypassing checkpoint+re-verify on partial acceptance "
                    "(SSM state may drift)\n");
        }
    }

    // Phase 4 GDN history kernel (Session 19 — partial-tail seq_rm path).
    //
    // When set: target's GDN op writes per-token recurrent state to a
    // persistent buffer; qwen35.cpp emits an in-graph state_select fixup
    // before each GDN layer's state read; on partial acceptance, the
    // spec driver drives k_index = K - 1 to roll the GDN state back to
    // the last accepted token in the chain. The recurrent-cache metadata
    // is rewound via llama_dflash_memory_seq_rm_partial_tail_state_managed_externally.
    // Together this replaces the FULL-seq_rm checkpoint + re-verify
    // round-trip — eliminating ~20 % of decode wall on Qwen3.5-27B
    // chain mode (Session 17 estimate).
    const bool use_gdn_history = params.speculative.dflash_gdn_history;
    if (use_gdn_history) {
        if (use_ckpt) {
            LOG_INF("--dflash-gdn-history: replacing checkpoint+re-verify path "
                    "with in-graph GDN-state fixup + partial-tail seq_rm on "
                    "partial acceptance\n");
        }
        use_ckpt = false;
    }

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
        params.speculative.draft.cparams.dflash_max_ctx     = params.speculative.dflash_max_ctx;
        params.speculative.draft.cparams.dflash_topk        = params.speculative.dflash_topk;
        // dflash_inline_encoder is target-only; explicitly clear it on the
        // draft's cparams so the draft context construction doesn't reject
        // it via the target-only validation.
        params.speculative.draft.cparams.dflash_inline_encoder = false;
        // dflash_gdn_history is target-only too (the draft has no GDN layers,
        // and the target-only validation would reject it anyway).
        params.speculative.draft.cparams.dflash_gdn_history    = false;
        // Best-first DDTree needs full-vocab logits so the host can compute
        // per-position log-probs for the heap-based tree builder. Default
        // mode (round-robin uniform expansion) keeps the bs * n_vocab * 4 byte
        // PCIe optimisation.
        params.speculative.draft.cparams.dflash_emit_logits = params.speculative.dflash_tree_best_first;
    }

    // DFlash: auto-wrap raw prompt with the target model's chat template.
    // Qwen3 / Qwen3.5 drafters are instruct-trained; feeding raw text runs
    // slightly OOD which lowers acceptance length. Gated on (1) DFlash being
    // active and (2) chat templating not explicitly disabled via
    // --no-chat-template, so non-DFlash callers and raw-completion users are
    // unaffected. Falls through to the original raw-prompt path if the model
    // has no template or templating fails.
    const bool dflash_active_for_prompt = llama_model_dflash_block_size(model_dft.get()) > 0;
    if (dflash_active_for_prompt && params.enable_chat_template && !params.prompt.empty()) {
        try {
            auto tmpls = common_chat_templates_init(model_tgt, params.chat_template);
            common_chat_msg msg;
            msg.role    = "user";
            msg.content = params.prompt;

            common_chat_templates_inputs inputs;
            inputs.messages              = { msg };
            inputs.add_generation_prompt = true;
            inputs.use_jinja             = params.use_jinja;
            // Disable thinking mode (Qwen3 / Qwen3.5): chain-of-thought adds
            // exploratory text that drops draft acceptance rate. z-lab's
            // benchmark convention matches direct-answer mode.
            inputs.enable_thinking       = false;

            const std::string wrapped = common_chat_templates_apply(tmpls.get(), inputs).prompt;
            if (!wrapped.empty()) {
                LOG_INF("%s: DFlash: chat-templated prompt (raw=%zu -> wrapped=%zu chars). "
                        "Disable with --no-chat-template.\n",
                        __func__, params.prompt.size(), wrapped.size());
                params.prompt = wrapped;
            } else {
                LOG_INF("%s: DFlash: chat template returned empty result, using raw prompt\n", __func__);
            }
        } catch (const std::exception & e) {
            LOG_INF("%s: DFlash: chat template not applied (%s), using raw prompt\n", __func__, e.what());
        }
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

    // init the speculator BEFORE the user prefill so DFlash capture is
    // installed in time. Pre-V2 the prefill ran first (without capture)
    // and `begin()` re-decoded the prompt to populate captures — a full
    // duplicate target prefill per generation, scaling with prompt
    // length. With init moved up, capture fires during the user's
    // prefill and `begin()` skips its re-decode.
    const auto & params_spec = params.speculative;

    if (dflash_mem_cfg.enabled) {
        dflash_prof::memory_profiler::instance().snapshot("after_init", nullptr, ctx_tgt);
    }
    if (dflash_prof_cfg.enabled) dflash_prof::profiler::instance().set_phase("prompt");

    struct common_speculative * spec = common_speculative_init(params.speculative, ctx_tgt);

    // DFlash detection: positive block size on the draft means a DFlash
    // drafter was loaded and capture was installed by `init` above; the
    // prefill must request output at every prompt position so capture
    // fires for each.
    const bool dflash_active = llama_model_dflash_block_size(model_dft.get()) > 0;

    llama_token id_last;
    llama_tokens prompt_tgt;
    int n_past;
    {
        DFLASH_GPU_TIMER("user prompt prefill (target)", ctx_tgt);
        if (dflash_active) {
            // DFlash: prefill the FULL prompt with logits=true at every
            // position so the drafter sees captures for all committed
            // target tokens (including the last). The anchor token
            // (id_last) is then sampled from the target's prediction at
            // the last prompt position, matching how the DFlash drafter
            // was trained — same convention as the reference Python
            // implementation (z-lab/dflash:dflash/model.py) and the
            // upstream-PR-candidate fork. The pre-fix path prefilled
            // inp.size()-1 tokens and used inp.back() as the anchor,
            // which left a one-position gap in captured features and
            // significantly hurt accept rate on long prompts at large
            // target model sizes (see logs/16_27b_accept_rate_divergence.md).
            const int32_t n_prefill = (int32_t) inp.size();
            llama_batch prefill_batch = llama_batch_init(n_prefill, 0, 1);
            for (int32_t i = 0; i < n_prefill; ++i) {
                common_batch_add(prefill_batch, inp[i], (llama_pos) i, { 0 }, /*logits=*/true);
            }
            // Phase 3 inline encoder: the prefill captures every prompt
            // position. Write them into the draft's side store starting at
            // slot 0, with RoPE positions also starting at 0.
            if (params.speculative.dflash_inline_encoder) {
                llama_dflash_set_inline_encode_state(ctx_tgt,
                    /*write_offset=*/ 0,
                    /*pos_start=*/    0);
            }
            if (llama_decode(ctx_tgt, prefill_batch) != 0) {
                LOG_ERR("%s: target prompt prefill (with capture) failed\n", __func__);
                llama_batch_free(prefill_batch);
                return 1;
            }
            llama_batch_free(prefill_batch);

            id_last = common_sampler_sample(smpl.get(), ctx_tgt, /*idx=*/-1);
            common_sampler_accept(smpl.get(), id_last, true);

            prompt_tgt.assign(inp.begin(), inp.end());
            prompt_tgt.reserve(llama_n_ctx(ctx_tgt));
            n_past = (int) inp.size();
        } else {
            llama_decode(ctx_tgt, llama_batch_get_one(inp.data(), inp.size() - 1));
            id_last = inp.back();
            prompt_tgt.assign(inp.begin(), inp.end() - 1);
            prompt_tgt.reserve(llama_n_ctx(ctx_tgt));
            n_past = (int) inp.size() - 1;
        }
    }

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
    // Single-seq DFS tree-with-GDN-history is the default whenever both
    // --dflash-tree and --dflash-gdn-history are set. Phase 5 + Session 26
    // wired this path with tree-depth positions (matching lucebox), the
    // tree-mode apply_ubatch bypass plumbed via llama_set_tree_mask, and
    // post-accept compaction via llama_memory_keep_positions_range. The
    // legacy multi-seq tree path (without GDN history) below is still
    // selected for `--dflash-tree` alone.
    const bool use_tree_with_gdn_history = use_tree && use_gdn_history;

    // DDTree Session 30: alt-rate-aware budget heuristic. On prompts whose
    // generation pattern doesn't reward alt-accept (alt-trajectory drifts
    // the drafter into a poor-fit region, see Session 29 p1 analysis), the
    // user-supplied tree budget is dropped to alt_decide_fallback for the
    // rest of generation once the warmup window's alt rate exceeds
    // alt_decide_threshold. The decision is made ONCE after the warmup —
    // a sliding window proved too noisy because alt counts on bad-fit
    // prompts cluster and re-cluster, and per-iter flips lost the chain-
    // wins on good-fit prompts.
    //
    // Disabled when (a) the user-supplied budget is already small (<=
    // alt_decide_fallback), (b) tree mode without GDN history (legacy
    // multi-seq path), or (c) the warmup hasn't completed. Override with
    // DFLASH_TREE_ALT_RATE_DISABLE=1 to A/B against the fixed-budget
    // config.
    constexpr int   alt_decide_warmup       = 3;
    constexpr float alt_decide_threshold    = 0.34f;  // > 1 alt in 3 iters
    // Session 30 origin: 16 (block_size for Qwen3-DFlash, validated on
    // CUDA RTX 5090 via the published 5-prompt sweep numbers).
    // Session 35 measurement on Strix Halo gfx1151 Vulkan (HE-10,
    // n_gen=256, Qwen3.5-27B Q4_K_M) showed b=18 strictly dominates
    // b=16 on the 10-prompt mean (27.86 vs 27.21 t/s). To avoid any
    // behaviour change for CUDA / NVIDIA / Strix Halo HIP setups
    // (where Session 30's choice of 16 was validated), the fallback
    // is left at 16 by default and opt-in via env var:
    //   DFLASH_ALT_FALLBACK_BUDGET=18  → AMD Vulkan / Strix Halo
    //   DFLASH_ALT_FALLBACK_BUDGET=22  → keep at user-set (no-op
    //                                    fallback; effectively
    //                                    disables the heuristic)
    // Default 16 preserves the existing tested-on-5090 behaviour.
    int alt_decide_fallback = 16;
    if (const char * e = std::getenv("DFLASH_ALT_FALLBACK_BUDGET")) {
        if (e[0] != '\0') {
            const int v = std::atoi(e);
            if (v >= 1) {
                alt_decide_fallback = v;
            }
        }
    }
    const int       alt_orig_budget         = params.speculative.dflash_tree_budget;
    const bool      alt_heuristic_active = use_tree_with_gdn_history &&
                                           alt_orig_budget > alt_decide_fallback &&
                                           [] {
                                               const char * e = std::getenv("DFLASH_TREE_ALT_RATE_DISABLE");
                                               return !(e && e[0] != '\0' && std::strcmp(e, "0") != 0);
                                           }();
    int     alt_warmup_count           = 0;  // total iters seen so far during warmup
    int     alt_warmup_alts            = 0;  // alt-accept iters during warmup
    bool    alt_locked_to_fallback     = false;
    int64_t alt_n_downscaled_iters     = 0;
    if (alt_heuristic_active) {
        LOG_INF("--dflash-tree-budget %d: alt-rate-aware budget heuristic "
                "active (warmup=%d iters, threshold=%.2f, fallback=%d). "
                "Disable with DFLASH_TREE_ALT_RATE_DISABLE=1.\n",
                alt_orig_budget, alt_decide_warmup, (double) alt_decide_threshold,
                alt_decide_fallback);
    }

    // DDTree Session 32: warmup-decide between chain and tree mode. Run the
    // first wd_warmup_iters in chain mode, measure chain accept yield
    // (n_accept / n_drafted), then lock to chain or tree for the rest of
    // generation based on the threshold. The hypothesis from the Session
    // 29 / 30 cross-prompt data: high chain yield (>= wd_yield_threshold)
    // means the drafter is hitting well in chain — tree expansion just
    // adds per-iter cost. Low chain yield means the drafter needs the
    // tree's alternates to land more tokens — tree wins.
    //
    // OPT-IN via DFLASH_TREE_WARMUP_DECIDE=1. Default off because the
    // warmup chain phase costs peak throughput on tree-friendly prompts
    // (e.g. HE[0]: tree-22 alone 142 t/s vs warmup-decide 125 t/s) in
    // exchange for rescuing tree-unfriendly prompts (HE[1]: tree-22
    // alone 95 t/s vs warmup-decide 143 t/s). Workloads with mixed
    // prompt distributions where worst-case predictability matters
    // more than peak should enable.
    constexpr int   wd_warmup_iters       = 3;
    constexpr float wd_yield_threshold    = 0.50f;
    const bool      wd_active = use_tree_with_gdn_history && [] {
        const char * e = std::getenv("DFLASH_TREE_WARMUP_DECIDE");
        return e && e[0] != '\0' && std::strcmp(e, "0") != 0;
    }();
    bool runtime_use_tree_with_gdn_history = use_tree_with_gdn_history && !wd_active;
    bool wd_decision_locked      = !wd_active;
    int  wd_chain_iters_done     = 0;
    int  wd_chain_n_accept_base  = 0;
    int  wd_chain_n_drafted_base = 0;
    // Session 35: optional tree-budget override when warmup-decide locks
    // to tree mode. Lets AMD Vulkan / Strix Halo users opt into the
    // measured-best b=18 budget WITHOUT changing the user-facing
    // --dflash-tree-budget default (so CUDA / NVIDIA / Strix Halo HIP
    // call sites that set b=22 keep that exact behaviour). Default 0
    // means "use --dflash-tree-budget as-is" — fully back-compat.
    int wd_tree_budget_override = 0;
    if (const char * e = std::getenv("DFLASH_TREE_WD_TREE_BUDGET")) {
        if (e[0] != '\0') {
            const int v = std::atoi(e);
            if (v >= 1) {
                wd_tree_budget_override = v;
            }
        }
    }
    if (wd_active) {
        LOG_INF("--dflash-tree: DFLASH_TREE_WARMUP_DECIDE=1 — warmup-decide "
                "active (chain warmup=%d iters, yield threshold=%.2f). "
                "After warmup, locks to tree mode if chain yield <= "
                "threshold, else stays in chain.%s\n",
                wd_warmup_iters, (double) wd_yield_threshold,
                wd_tree_budget_override > 0
                    ? string_format(" Tree-side budget override "
                                    "DFLASH_TREE_WD_TREE_BUDGET=%d active.",
                                    wd_tree_budget_override).c_str()
                    : "");
    }

    while (true) {
        if (dflash_prof_cfg.enabled) dflash_prof::profiler::instance().set_phase("draft");

        // ============================================================
        // DDTree Phase 5: single-seq DFS-flattened tree-verify with
        // GDN history. Engaged when both --dflash-tree and
        // --dflash-gdn-history are set. Replaces the multi-seq Stage-C
        // tree path below with:
        //   1. Single-seq batch (all tree tokens in seq 0, in DFS
        //      order, positions = n_past_before + tree.depths[i]).
        //   2. tree_mask (visibility matrix) for attention isolation
        //      AND apply_ubatch contiguity-purge bypass on the unified
        //      KV cache (set_tree_mode_active fires from inside
        //      llama_set_tree_mask).
        //   3. parent_ids buffer for the GDN + conv tree kernels.
        //   4. Accept walk; record the DFS slot of each committed node.
        //   5. llama_memory_keep_positions_range(seq=0, accepted, p_min):
        //      keeps the root + accepted-path cells, drops every other
        //      verify-batch cell, and renames the kept ones to a
        //      contiguous monotonic block [n_past_before .. n_past_before+L].
        //      Recurrent half: rewind seq_pos via the partial-tail API
        //      (the GDN+conv recurrent state itself is fixed up by the
        //      next iter's state_select op via k_index = commit_n,
        //      pointing at gdn_history[deepest accepted DFS slot] which
        //      was just written by the tree-verify kernel).
        //
        // Reference: lucebox's single-seq tree-verify pattern uses
        // position = committed + depth (test_dflash.cpp:2560) and a
        // slot-rewire compaction (test_dflash.cpp:2783-2818). We
        // achieve the same end state via keep_positions_range, which
        // performs the equivalent rewrite within llama.cpp's unified
        // cache.
        //
        // Session 32: gated on runtime_use_tree_with_gdn_history (mutated
        // by the warmup-decide heuristic). During warmup the spec driver
        // takes the chain path below to measure chain yield; after the
        // decision it stays locked to chain or tree for the rest of
        // generation.
        if (runtime_use_tree_with_gdn_history) {
            // Pick this iter's tree budget. If the warmup decided to
            // lock to the fallback, every subsequent iter uses it.
            // Session 35: if warmup-decide is active AND a tree-budget
            // override is set, use that override (lets AMD Vulkan
            // users opt into b=18 without changing user-set b=22 on
            // 5090 / HIP). The override is composed AFTER alt-rate
            // fallback so a smaller alt-fallback still wins.
            int effective_budget = alt_orig_budget;
            if (wd_active && wd_decision_locked && wd_tree_budget_override > 0) {
                effective_budget = wd_tree_budget_override;
            }
            if (alt_locked_to_fallback) {
                effective_budget = alt_decide_fallback;
                ++alt_n_downscaled_iters;
            }
            common_params_speculative params_spec_iter = params_spec;
            params_spec_iter.dflash_tree_budget = effective_budget;

            common_speculative_tree tree =
                common_speculative_draft_tree(spec, params_spec_iter, prompt_tgt, id_last);

            const int n_past_before = n_past;
            auto * mem = llama_get_memory(ctx_tgt);

            common_batch_clear(batch_tgt);
            llama_tokens ids;
            // Whether this iter took an alt branch (filled inside the
            // tree-verify branch below; stays false for empty-fallback
            // iters which are 1-token chain decodes).
            bool iter_alt_taken = false;

            if (tree.n_nodes == 0) {
                // Drafter returned no tree. Fall back to a single-
                // token decode + bonus, same as the legacy tree path.
                // Clear the parent_ids buffer so any prior
                // tree-shaped value doesn't leak into the chain
                // fallback graph's set_input. We deliberately leave
                // gdn_history_k_index{,_per_seq} alone — if the
                // previous iter was a tree verify, the recurrent slot
                // currently holds the wrong "last DFS slot" state and
                // needs the state_select fixup to load gdn_history
                // [commit_n] = state-after-deepest-accepted before this
                // 1-token AR/fused decode reads it. After this decode,
                // n_seq_tokens=1 → AR/fused path writes the correct end
                // state back to the recurrent slot, so we clear k_index
                // for the next iter.
                llama_dflash_set_gdn_history_parent_ids(
                    ctx_tgt, nullptr, 0, 0);
                common_batch_add(batch_tgt, id_last, n_past_before, { 0 }, /*logits=*/true);
                {
                    DFLASH_GPU_TIMER("target tree-empty fallback decode", ctx_tgt);
                    llama_decode(ctx_tgt, batch_tgt);
                }
                const llama_token bonus =
                    common_sampler_sample(smpl.get(), ctx_tgt, 0);
                common_sampler_accept(smpl.get(), bonus, true);
                ids.push_back(bonus);
                n_past = n_past_before + 1;
                llama_dflash_set_gdn_history_k_index(ctx_tgt, -1);
            } else {
                const int n_tree_tokens = 1 + tree.n_nodes;

                // (1) Single-seq DFS-flattened verify batch. All tokens
                // belong to seq 0; attention separation comes from the
                // tree_mask we install below. Positions are tree depth
                // (root at n_past_before, depth-1 children at
                // n_past_before+1, ...) so ROPE rotation between Q at
                // depth d_q and K at depth d_k matches the d_q - d_k
                // relative offset the model was trained on. Sibling
                // tokens at the same depth share a position; the
                // duplicate writes are tolerated because
                // llama_set_tree_mask flips the KV cache into tree mode
                // (apply_ubatch's contiguity-invariant purge is bypassed
                // for this verify), and the accept walk's
                // keep_positions_range below compacts the cache back to
                // a monotonic block. See Session 24 design notes.
                common_batch_add(batch_tgt, id_last, n_past_before, { 0 }, /*logits=*/true);
                for (int i = 0; i < tree.n_nodes; ++i) {
                    common_batch_add(batch_tgt, tree.tokens[i],
                                     n_past_before + tree.depths[i],
                                     { 0 }, /*logits=*/true);
                }

                // (2) Tree mask. The visibility matrix is exactly the
                // (n_tree_tokens × n_tree_tokens) byte matrix already
                // built by common_speculative_draft_tree.
                llama_set_tree_mask(ctx_tgt, tree.visibility.data(),
                                    n_tree_tokens);

                // (3) parent_ids for GDN + conv tree kernels.
                // gdn_history_n_seqs_max sets the persistent buffer's
                // last dim; we always emit a [n_tree_tokens, 1]
                // parent_ids since the verify batch is single-seq, and
                // unused seq slabs (s > 0) in the kernel are dormant.
                std::vector<int32_t> parent_ids(
                    (size_t) n_tree_tokens * 1, -1);
                const bool ok = tree.write_parent_ids(
                    parent_ids.data(), n_tree_tokens, 1);
                GGML_ASSERT(ok && "write_parent_ids: tree shape mismatch");
                llama_dflash_set_gdn_history_parent_ids(
                    ctx_tgt, parent_ids.data(), n_tree_tokens, 1);

                // GDN-state fixup on entry: leave whatever the previous
                // iter set. After a tree verify the recurrent slot
                // holds the kernel's end-of-DFS state (= last DFS
                // slot's state, NOT the deepest-accepted state), so
                // the previous iter has set k_index_per_seq = [commit_n]
                // pointing at gdn_history[commit_n] = state-after-leaf;
                // the state_select op fires here to overwrite the
                // recurrent slot before with_history_tree reads it for
                // the new root. On the very first iter after prefill
                // the value is still the default (-1) so the op is a
                // no-op and the prefill state is used unchanged.

                if (dflash_prof_cfg.enabled) dflash_prof::profiler::instance().set_phase("verify");
                {
                    DFLASH_GPU_TIMER("target tree-verify decode (single-seq DFS)", ctx_tgt);
                    llama_decode(ctx_tgt, batch_tgt);
                }

                // (4) Tree-walk accept: root → ... → commit_n. Track
                // each accepted child's DFS slot so step (5) can hand
                // keep_positions_range the exact accept-path positions.
                std::vector<int> accepted_dfs;
                accepted_dfs.reserve((size_t) tree.n_nodes);
                int current  = 0;
                int commit_n = 0;
                while (true) {
                    const llama_token target_token =
                        common_sampler_sample(smpl.get(), ctx_tgt, current);
                    common_sampler_accept(smpl.get(), target_token, true);
                    ids.push_back(target_token);

                    auto it = tree.child_maps[current].find(target_token);
                    if (it == tree.child_maps[current].end()) {
                        break; // bonus token (off-tree)
                    }
                    current  = it->second;
                    commit_n = current;
                    accepted_dfs.push_back(current);
                }

                // Done with the tree-mode visibility + parent_ids — clear
                // them now so the next iteration's set_input doesn't
                // pick up stale tree state. clear_tree_mask also flips
                // set_tree_mode_active(false) on the KV cache so the
                // next non-tree decode path sees standard contiguity
                // semantics.
                llama_clear_tree_mask(ctx_tgt);
                llama_dflash_set_gdn_history_parent_ids(
                    ctx_tgt, nullptr, 0, 0);

                // L = number of accepted children (= depth of deepest
                // accepted leaf along its tree-ancestry, since the accept
                // walk descends one tree edge per step).
                const int L = (int) accepted_dfs.size();
                const bool alt_accepted = (commit_n > tree.main_path_len);
                iter_alt_taken = alt_accepted;

                if (alt_accepted) {
                    n_tree_alt_taken++;
                    // Hint the spec state to remap captures on next
                    // extend. The alt_capture_idx is the alt leaf's DFS
                    // batch slot — the draft uses it to slice its
                    // target-feat captures from the verify in the
                    // alt-ancestor order. alt_depth = L (= the number of
                    // committed tokens along the alt branch).
                    common_speculative_record_alt_accept(spec,
                        /*alt_capture_idx=*/commit_n,
                        /*alt_depth=*/L);
                }

                // (5) KV-cache compaction by DFS-write ordinal. Cells
                // identified by their allocation order (= DFS slot)
                // disambiguates same-position siblings; the kept
                // cells are renamed to a contiguous monotonic block
                // [n_past_before .. n_past_before + L].
                {
                    std::vector<int32_t> dfs_keep;
                    dfs_keep.reserve(1 + accepted_dfs.size());
                    dfs_keep.push_back(0);
                    for (int dfs : accepted_dfs) {
                        dfs_keep.push_back((int32_t) dfs);
                    }
                    const bool ok = llama_memory_keep_cells_dfs_ordinals_range(
                        mem, /*seq_id=*/0,
                        dfs_keep.data(),
                        (int32_t) dfs_keep.size(),
                        /*p_min=*/n_past_before);
                    GGML_ASSERT(ok);
                }

                // Recurrent half: rewind seq_pos to match the
                // compacted attention cache. The recurrent state
                // itself is fixed up by the next iter's state_select
                // op (k_index = commit_n picks gdn_history[deepest
                // accepted DFS slot]).
                {
                    const bool ok = llama_dflash_memory_seq_rm_partial_tail_state_managed_externally(
                        mem, 0, n_past_before + 1 + L, -1);
                    GGML_ASSERT(ok);
                }
                {
                    int32_t k_leaf = commit_n;
                    llama_dflash_set_gdn_history_k_index_per_seq(
                        ctx_tgt, &k_leaf, 1);
                }
                n_past = n_past_before + 1 + L;
            }

            n_tree_iters++;
            common_speculative_accept(spec,
                ids.empty() ? 0 : (uint16_t)(ids.size() - 1));

            n_drafted += tree.n_nodes;
            n_accept  += ids.empty() ? 0 : (int) ids.size() - 1;

            // Drive the warmup-decide heuristic. Empty-fallback iters
            // (tree.n_nodes == 0, iter_alt_taken stays false) count
            // against the alt-rate.
            if (alt_heuristic_active && !alt_locked_to_fallback) {
                ++alt_warmup_count;
                if (iter_alt_taken) ++alt_warmup_alts;
                if (alt_warmup_count >= alt_decide_warmup) {
                    const float alt_rate = (float) alt_warmup_alts /
                                           (float) alt_warmup_count;
                    if (alt_rate > alt_decide_threshold) {
                        alt_locked_to_fallback = true;
                        LOG_INF("alt-rate heuristic: warmup %d iters "
                                "saw %d alts (rate %.2f > %.2f); locking "
                                "tree budget to %d for the rest of "
                                "generation.\n",
                                alt_warmup_count, alt_warmup_alts,
                                (double) alt_rate,
                                (double) alt_decide_threshold,
                                alt_decide_fallback);
                    }
                }
            }

            // Commit accepted tokens + bonus to id_last, capped at
            // params.n_predict. Same shape as the legacy tree branch
            // below so byte-exact match against greedy is preserved.
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
        //
        // Session 32: gated on `!use_gdn_history` so this strictly
        // handles the legacy `--dflash-tree` (without GDN history)
        // case. When use_gdn_history is set, the use_tree_with_gdn_history
        // path above owns the tree decision (possibly via the warmup-
        // decide runtime flag) and we don't want to fall through to the
        // legacy path during the warmup phase.
        if (use_tree && !use_gdn_history) {
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
                {
                    DFLASH_GPU_TIMER("target tree-empty fallback decode", ctx_tgt);
                    llama_decode(ctx_tgt, batch_tgt);
                }

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
                {
                    DFLASH_GPU_TIMER("target tree-verify decode", ctx_tgt);
                    llama_decode(ctx_tgt, batch_tgt);
                }

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
        // Phase 3 inline encoder: this iteration's captured rows land in
        // the side store starting at slot prompt_tgt.size() (= n_committed_total),
        // with RoPE positions starting at the current n_past (the slot
        // id_last lands in inside the target's batch). Must be set BEFORE
        // n_past++ below.
        if (params.speculative.dflash_inline_encoder) {
            llama_dflash_set_inline_encode_state(ctx_tgt,
                /*write_offset=*/ (int64_t) prompt_tgt.size(),
                /*pos_start=*/    (int64_t) n_past);
        }
        common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

        // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
        {
            for (size_t i = 0; i < draft.size(); ++i) {
                common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
            }

            //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

            if (dflash_prof_cfg.enabled) dflash_prof::profiler::instance().set_phase("verify");
            {
                DFLASH_GPU_TIMER("target chain-verify decode", ctx_tgt);
                llama_decode(ctx_tgt, batch_tgt);
            }
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

        // Phase 4 GDN history fixup: between target decodes, set the
        // host k_index so the next verify graph's state_select op
        // restores the GDN recurrent state to "right after the last
        // accepted token" instead of "after the entire draft".
        //
        // Verify batch was [id_last, draft[0], ..., draft[N-1]] at
        // positions [P, P+1, ..., P+N]. state_history slot t holds the
        // GDN state right after position P+t. The sampler accepts
        // `accepted_count = ids.size() - 1` draft tokens (id_last is
        // never counted as a draft accept). Last accepted token sits
        // at position P + accepted_count → k_index = accepted_count.
        //
        //   Partial: accepted_count < draft.size() →
        //     k_index = accepted_count (rolls state back, fixup runs).
        //   Full:    accepted_count == draft.size() →
        //     k_index = -1 (no-op; current slot already holds the
        //     correct final state).
        if (use_gdn_history) {
            const int accepted_count = (int) ids.size() - 1;
            if (accepted_count < (int) draft.size()) {
                llama_dflash_set_gdn_history_k_index(ctx_tgt,
                    (int32_t) accepted_count);
            } else {
                llama_dflash_set_gdn_history_k_index(ctx_tgt, -1);
            }
        }

        common_speculative_accept(spec, ids.size() - 1);

        // full acceptance: consume the draft and commit accepted tokens
        n_past    += ids.size() - 1;
        n_drafted += n_draft; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;

        // Session 32 warmup-decide: drive the chain → tree/chain lock.
        // We tracked n_accept/n_drafted via global counters above; subtract
        // baselines to get the warmup window's deltas.
        if (wd_active && !wd_decision_locked) {
            ++wd_chain_iters_done;
            if (wd_chain_iters_done >= wd_warmup_iters) {
                const int warmup_accept  = n_accept  - wd_chain_n_accept_base;
                const int warmup_drafted = n_drafted - wd_chain_n_drafted_base;
                const float yield = warmup_drafted > 0
                                  ? (float) warmup_accept / (float) warmup_drafted
                                  : 0.0f;
                const bool stay_chain = yield > wd_yield_threshold;
                if (!stay_chain) {
                    runtime_use_tree_with_gdn_history = true;
                }
                wd_decision_locked = true;
                LOG_INF("warmup-decide: chain yield %.2f over %d iters "
                        "(%d/%d accept/drafted); %s for the rest of "
                        "generation.\n",
                        (double) yield, wd_chain_iters_done,
                        warmup_accept, warmup_drafted,
                        stay_chain ? "staying in chain mode"
                                   : "switching to tree mode");
            }
        }

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

            if (use_gdn_history) {
                // The GDN state for positions [n_past, end-of-batch) is
                // about to be rolled back by the next decode's
                // state_select fixup, so allow the recurrent half to
                // rewind its tail-cell pos rather than rejecting the
                // partial-tail removal.
                const bool ok = llama_dflash_memory_seq_rm_partial_tail_state_managed_externally(
                    llama_get_memory(ctx_tgt), 0, n_past, -1);
                GGML_ASSERT(ok);
            } else {
                llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
            }
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
        if (alt_heuristic_active) {
            LOG_INF("tree budget downscaled iters = %lld / %lld (heuristic: warmup=%d iters, threshold=%.2f, fallback=%d, locked=%s)\n",
                    (long long) alt_n_downscaled_iters, (long long) n_tree_iters,
                    alt_decide_warmup, (double) alt_decide_threshold,
                    alt_decide_fallback,
                    alt_locked_to_fallback ? "yes" : "no");
        }
        if (wd_active) {
            LOG_INF("warmup-decide: locked to %s after %d warmup iters\n",
                    runtime_use_tree_with_gdn_history ? "tree" : "chain",
                    wd_chain_iters_done);
        }
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
    if (dflash_gpu_log_cfg.enabled) {
        FILE * out = stderr;
        if (!dflash_gpu_log_cfg.out_file.empty()) {
            FILE * f = std::fopen(dflash_gpu_log_cfg.out_file.c_str(), "w");
            if (f) out = f;
        }
        dflash_gpu_log::dump_summary(out);
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
