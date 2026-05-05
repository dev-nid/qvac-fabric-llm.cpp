#include "arg.h"
#include "common.h"
#include "sampling.h"
#include "speculative.h"
#include "log.h"
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <utility>
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

    common_init();

    if (params.speculative.model.path.empty()) {
        LOG_ERR("%s: --model-draft is required\n", __func__);
        return 1;
    }

    // DDTree Phase 2 Stage B (Option C: multi-seq storage):
    //
    // Tree-mode verify uses TWO sequence IDs on ctx_tgt — seq 0 holds
    // the "main path" branch (= the chain prediction at each draft
    // depth), seq 1 holds the "alternate" branch (= a single top-2
    // node at depth 1 for Stage B; up to n_seq_max-1 branches in
    // future Stage C). The standard seq-id-based attention mask in
    // `set_input_kq_mask` then handles cross-branch isolation
    // automatically — no tree-mask override needed for our shape, no
    // n_ubatch bump needed (works at -ub 1, including the canonical
    // gate's -ub 1 invocation), no per-iter re-decode needed for
    // main-path-accepted iters (the verify-decode captures at output
    // indices [0..L] are mathematically equivalent to chain-mode
    // captures because the seq-id mask gives main path nodes the
    // same prefix as a chain decode would).
    //
    // We bump `params.n_parallel` (which becomes ctx_tgt's
    // `n_seq_max`) to at least 2 so the second seq is allocatable.
    // Chain mode is unaffected (only runs when --dflash-tree is on).
    if (params.speculative.dflash_tree) {
        // Stage C: n_branches at runtime depends on the budget. Worst-case
        // upper bound (= what we need n_seq_max for):
        //   n_branches = 1 (main) + min(budget - chain_len, (K-1) * chain_len)
        // where chain_len = min(budget, block_size - 1). For Stage B
        // default (budget=0 -> shape with 1 alt) n_branches = 2. For
        // budget=22, K=2: chain_len = 15, n_alts = 7, n_branches = 8.
        // For budget=45, K=4: chain_len = 15, n_alts = 30, n_branches = 31.
        // We don't know block_size at this point (the draft model isn't
        // loaded yet), so we use the worst-case value max(2, budget+1).
        // Over-allocating n_seq_max is cheap (small per-slot metadata).
        const int budget = params.speculative.dflash_tree_budget;
        const int min_parallel_tree = std::max(2, (budget > 0) ? budget + 1 : 2);
        if (params.n_parallel < min_parallel_tree) {
            LOG_INF("%s: --dflash-tree (budget=%d): bumping n_parallel from %d to %d "
                    "so ctx_tgt's n_seq_max fits all alt-branch seq_ids\n",
                    __func__, budget, params.n_parallel, min_parallel_tree);
            params.n_parallel = min_parallel_tree;
        }
        // Tree mode requires the unified KV cache (n_stream=1): the
        // multi-seq attention mask path expects all seqs to share a
        // single stream so tokens in alt seqs can attend (via the
        // standard seq-id-based mask) to KV slots written under seq 0
        // after `seq_cp(0 -> b)` broadcasts the prefix. Without
        // kv_unified, n_seq_max > 1 gives each seq its own stream and
        // cross-seq attention requires an actual data copy.
        if (!params.kv_unified) {
            LOG_INF("%s: --dflash-tree: enabling kv_unified (required for multi-seq "
                    "tree-verify pattern)\n",
                    __func__);
            params.kv_unified = true;
        }
    }

    // init llama.cpp
    llama_backend_init();
    llama_numa_init(params.numa);

    llama_model * model_tgt = NULL;

    llama_context * ctx_tgt = NULL;
    llama_context * ctx_dft = NULL;

    // load the target model
    common_init_result llama_init_tgt = common_init_from_params(params);

    model_tgt = llama_init_tgt.model.get();
    ctx_tgt   = llama_init_tgt.context.get();

    if (model_tgt == nullptr || ctx_tgt == nullptr) {
        LOG_ERR("%s: failed to load target model '%s'\n", __func__, params.model.path.c_str());
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model_tgt);

    // -----------------------------------------------------------------
    // Load the draft model.
    //
    // We use a load → bind → init split (rather than the one-shot
    // common_init_from_params used by the original speculative-simple
    // driver) because DFlash drafts share their tok_embd / lm_head with
    // the target model and the binding has to happen *before* the draft
    // context is constructed (graph_reserve runs at construction time).
    //
    // For non-DFlash drafts, llama_dflash_bind_target() is a no-op, so
    // the split is harmless. The rest of the params shuffling (devices,
    // n_ctx, n_gpu_layers, cpuparams, tensor_buft_overrides) is identical
    // to what the original driver did via common_init_from_params.
    // -----------------------------------------------------------------
    {
        params.devices         = params.speculative.devices;
        params.model           = params.speculative.model;
        params.n_ctx           = params.speculative.n_ctx;
        params.n_batch         = params.speculative.n_ctx > 0 ? params.speculative.n_ctx : params.n_batch;
        params.n_gpu_layers    = params.speculative.n_gpu_layers;
        // Forward the speculative.dflash_max_ctx into the top-level params so
        // common_context_params_to_llama() picks it up for the draft context.
        params.dflash_max_ctx  = params.speculative.dflash_max_ctx;
        params.dflash_topk     = params.speculative.dflash_topk;
        // Forward the per-draft KV-cache quantisation flags (-ctkd / -ctvd)
        // so common_context_params_to_llama() applies them to the draft
        // context (it copies into cparams.type_k / type_v at common.cpp:1306).
        // Without this, -ctkd / -ctvd are silently ignored on this binary
        // even though the server path (tools/server/server-context.cpp)
        // already does the equivalent plumbing.
        params.cache_type_k    = params.speculative.cache_type_k;
        params.cache_type_v    = params.speculative.cache_type_v;
        // DDTree Phase 2 Stage B: when --dflash-tree is set, the
        // single-iteration loop below installs a tree mask before the
        // target's verify decode and uses a tree-walk accept. The mask
        // installation API is on ctx_tgt (no draft-context plumbing
        // needed); we surface the flag here as a convenience for the
        // dflash_topk auto-bump (already applied in arg.cpp).

        if (params.speculative.cpuparams.n_threads > 0) {
            params.cpuparams.n_threads = params.speculative.cpuparams.n_threads;
        }
        params.cpuparams_batch.n_threads = params.speculative.cpuparams_batch.n_threads;
        params.tensor_buft_overrides     = params.speculative.tensor_buft_overrides;
    }

    common_init_result llama_init_dft;
    {
        auto mparams = common_model_params_to_llama(params);
        llama_model * model_dft = llama_model_load_from_file(params.model.path.c_str(), mparams);
        if (model_dft == nullptr) {
            LOG_ERR("%s: failed to load draft model '%s'\n", __func__, params.model.path.c_str());
            return 1;
        }

        // Paper §4.2: the DFlash draft shares tok_embd / lm_head with the
        // target. The bind must happen *before* the draft context is
        // created. No-op for non-DFlash drafts.
        llama_dflash_bind_target(model_dft, model_tgt);

        llama_init_dft = common_init_from_model_and_params(model_dft, std::move(llama_init_dft), params);
    }
    ctx_dft = llama_init_dft.context.get();

    if (ctx_dft == nullptr) {
        LOG_ERR("%s: failed to create draft context\n", __func__);
        return 1;
    }

    if (!common_speculative_are_compatible(ctx_tgt, ctx_dft)) {
        LOG_INF("the draft model '%s' is not compatible with the target model '%s'. tokens will be translated between the draft and target models.\n", params.speculative.model.path.c_str(), params.model.path.c_str());
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

    // how many tokens to draft each time
    int n_draft     = params.speculative.n_max;
    int n_draft_min = params.speculative.n_min;

    float p_min = params.speculative.p_min;

    int n_predict = 0;
    int n_drafted = 0;
    int n_accept  = 0;

    // DDTree Phase 2 Stage B counters (only meaningful in tree mode):
    int n_tree_iters     = 0; // # verify iterations that took the tree path
    int n_tree_alt_taken = 0; // # iterations where the alternate branch was accepted
    int n_tree_redecoded = 0; // total # tokens re-decoded in causal mode for KV cleanup

    // Per-phase wall-clock timing (microseconds; both modes accumulate).
    int64_t t_draft_total    = 0;  // gen_draft / gen_draft_tree
    int64_t t_verify_total   = 0;  // ctx_tgt verify decode
    int64_t t_accept_total   = 0;  // sample-and-accept (chain or tree-walk)
    int64_t t_redecode_total = 0;  // tree-mode re-decode + seq_rm rollback
    int64_t t_iter_total     = 0;  // sum of per-iter wall-clock

    // used to determine end of generation
    bool has_eos = false;

    // ================================================
    // everything until here is standard initialization
    // the relevant stuff for speculative decoding starts here

    const auto t_enc_start = ggml_time_us();

    // target model sampling context
    struct common_sampler * smpl = common_sampler_init(model_tgt, params.sampling);

    // init the speculator. AUTO picks DFLASH iff the draft GGUF carries
    // dflash.* metadata; otherwise DRAFT (the original llama.cpp path).
    struct common_speculative * spec = common_speculative_init_typed(
            ctx_tgt, ctx_dft, params.speculative.type);
    if (spec == nullptr) {
        LOG_ERR("%s: failed to initialise speculative decoder (--draft-type "
                "incompatible with the loaded draft model)\n", __func__);
        return 1;
    }
    for (auto &pair : params.speculative.replacements) {
        common_speculative_add_replacement_tgt_dft(spec, pair.first.c_str(), pair.second.c_str());
    }

    const enum common_speculative_type spec_type = common_speculative_get_type(spec);
    const bool is_dflash = (spec_type == COMMON_SPECULATIVE_TYPE_DFLASH);
    if (is_dflash) {
        // The DFlash drafter always returns exactly block_size-1 tokens; the
        // n_draft_min discard logic in the loop below would throw them all
        // away if the user (or the default n_min=0 case) ever raised this
        // above 0. Force-disable it for DFlash to keep a single shape.
        n_draft_min = 0;
        // For DFlash, n_draft is the block size (number of mask positions in
        // each forward pass). Default 16 matches the Qwen3-4B-DFlash-b16
        // training value; n_min has no semantic for block-parallel drafting.
        LOG_INF("%s: --draft-type dflash: block_size=%d (capped at the trained value)\n",
                __func__, n_draft);
    }

    // Run the prompt prefill on the target context. For DRAFT this is the
    // legacy llama_decode(batch_get_one(prompt[0..n-1])); for DFLASH it
    // also pushes the prompt-wide captures through the encoder so the
    // first gen_draft has a populated K/V side store.
    if (!common_speculative_target_prefill(spec, inp)) {
        LOG_ERR("%s: target prompt prefill failed\n", __func__);
        return 1;
    }

    // note: keep the last token separate!
    llama_token id_last = inp.back();

    // all tokens currently in the target context
    llama_tokens prompt_tgt(inp.begin(), inp.end() - 1);
    prompt_tgt.reserve(llama_n_ctx(ctx_tgt));

    int n_past = inp.size() - 1;

    // init the speculator's per-call params
    struct common_speculative_params params_spec;
    params_spec.n_draft            = n_draft;
    params_spec.n_reuse            = llama_n_ctx(ctx_dft) - n_draft;
    params_spec.p_min              = p_min;
    params_spec.dflash_tree_budget = params.speculative.dflash_tree_budget;

    // Tree mode multi-seq: each batch token may carry up to n_branches
    // seq_ids (root has all branches; main-path nodes carry seq 0 plus
    // the branches of every alt at deeper depths). Per-slot seq_id
    // arrays are allocated by llama_batch_init at this fixed size, so
    // we size to the worst-case n_branches up front. For chain mode and
    // Stage B (n_branches <= 2), keep the historical size 1; for Stage
    // C with --dflash-tree-budget B, n_branches <= 1 + budget.
    const int batch_n_seq_max = params.speculative.dflash_tree
        ? std::max(2, params.speculative.dflash_tree_budget + 1)
        : 1;
    llama_batch batch_tgt = llama_batch_init(llama_n_batch(ctx_tgt), 0, batch_n_seq_max);

    const auto t_enc_end = ggml_time_us();

    const auto t_dec_start = ggml_time_us();

    const bool use_tree = is_dflash && params.speculative.dflash_tree;

    while (true) {
        llama_tokens             ids;
        size_t                   n_drafted_this_iter = 0;
        common_speculative_tree  tree;

        const int64_t t_iter_start = ggml_time_us();
        if (use_tree) {
            // ============================================================
            // DDTree Phase 2 Stage B (Option C: multi-seq storage)
            // ============================================================
            // 1. Build a small tree from the draft's per-position top-K
            //    (Stage B shape: chain seed = bs-1 main-path nodes at
            //    depths 1..bs-1, plus 1 alternate at depth 1 = total
            //    bs+1 tree nodes including the implicit root).
            //
            // 2. Multi-seq batch construction:
            //    a. seq_cp(0 -> 1, -1, -1) broadcasts the prefix into
            //       seq 1 so alt-branch nodes can attend to it via the
            //       standard seq-id-based mask.
            //    b. id_last is added with seq_id={0, 1} (visible from
            //       both branches; carries no draft information of its
            //       own, just the anchor).
            //    c. Main-path nodes get seq_id={0}; alt nodes (indices
            //       > main_path_len) get seq_id={1}. Positions are
            //       depth-based as before — siblings at the same tree
            //       depth share a position, but they live in DIFFERENT
            //       seqs so the standard mask separates them.
            //
            // 3. Decode (NO tree mask — the seq-id-based mask handles
            //    every cross-branch separation correctly because each
            //    branch's nodes only have its own branch's seq_id).
            //    At -ub 1 each token gets its own ubatch and the
            //    standard mask runs per ubatch; at larger -ub the
            //    same per-row seq-id check runs on the batched mask.
            //
            // 4. Tree-walk accept (unchanged from chain-mode behavior:
            //    walk root -> ... -> commit_n following target argmax).
            //
            // 5. Rollback (clean, no re-decode for main-path accepts):
            //    - For commit_n == 0 (bonus only): seq_rm(0, N0+1, -1)
            //      drops m1..m15 from seq 0; seq_rm(1, -1, -1) drops
            //      seq 1 from everywhere (un-tags prefix + id_last,
            //      drops alt slot entirely). id_last stays in seq 0.
            //      No re-decode. Captures[0] is id_last's correct
            //      hidden state.
            //    - For main-accepted L tokens (commit_n in 1..main_path_len):
            //      seq_rm(0, N0+L+1, -1) drops m_{L+1}..m_15;
            //      seq_rm(1, -1, -1) drops seq 1. Cache has prefix +
            //      id_last + m1..m_L. Captures[0..L] are correct (main
            //      path's tree-decode prefix is the chain prefix).
            //      No re-decode.
            //    - For alt-accepted (commit_n > main_path_len): drop
            //      everything in seq 0 from N0 onward AND drop seq 1,
            //      then RE-DECODE [id_last, alt] in seq 0 to populate
            //      KV + captures correctly. Re-decode is 2 tokens; this
            //      branch fires only when target's argmax at root is
            //      the alt token (~5-15% of iters in our prompt grid).
            const int64_t t_d0 = ggml_time_us();
            tree = common_speculative_gen_draft_tree(spec, params_spec, prompt_tgt, id_last);
            t_draft_total += ggml_time_us() - t_d0;

            const int n_past_before = n_past;
            auto * mem = llama_get_memory(ctx_tgt);

            common_batch_clear(batch_tgt);
            if (tree.n_nodes == 0) {
                // Drafter failed (no top-K data or trivial). Decode
                // just id_last in seq 0 and emit the bonus.
                common_batch_add(batch_tgt, id_last, n_past_before, { 0 }, /*logits=*/true);
                const int64_t t_v0 = ggml_time_us();
                llama_decode(ctx_tgt, batch_tgt);
                t_verify_total += ggml_time_us() - t_v0;

                const int64_t t_a0 = ggml_time_us();
                const llama_token bonus = common_sampler_sample(smpl, ctx_tgt, 0);
                common_sampler_accept(smpl, bonus, true);
                ids.push_back(bonus);
                t_accept_total += ggml_time_us() - t_a0;

                n_past = n_past_before + 1;
            } else {
                // Multi-seq prefix tagging: copy seq 0 -> seq b for each
                // alt branch b in [1..n_branches-1] so each alt's standard
                // attention mask reaches the shared prefix. `seq_cp` with
                // `kv_unified=true` is metadata-only (just adds the dst
                // seq_id to every existing seq-0 slot's seq list — no
                // data copy).
                for (int b = 1; b < tree.n_branches; ++b) {
                    llama_memory_seq_cp(mem, 0, b, -1, -1);
                }

                // For each MAIN-PATH node at depth d, compute which alt
                // branch seq_ids it must additionally carry: every alt at
                // depth > d (i.e., where main_d is on the alt's ancestor
                // chain). This is required so the KV cache's per-seq
                // position-consistency check sees a contiguous run of
                // positions in every alt seq from the prefix through the
                // alt's depth. Without it, an alt at depth d=3 in seq 3
                // would see seq 3's last position = N0 (root only), and
                // adding a slot at N0+3 would fail the Y = X+1 invariant.
                //
                // For Stage C uniform expansion: alt at depth d_a sits in
                // seq b = (alt index within depths) so we pre-bin by
                // depth. Computed in O(n_nodes) and indexed in O(1) per
                // main-path token.
                std::vector<std::vector<llama_seq_id>> main_extra_seqs(tree.main_path_len);
                for (int i = 0; i < tree.n_nodes; ++i) {
                    if (tree.branch_ids[i] == 0) continue; // skip main
                    const int alt_depth = tree.depths[i];
                    const llama_seq_id alt_seq = (llama_seq_id) tree.branch_ids[i];
                    // main_d_{1..alt_depth-1} need to be visible to alt_seq.
                    for (int d_main = 1; d_main < alt_depth; ++d_main) {
                        main_extra_seqs[d_main - 1].push_back(alt_seq);
                    }
                }

                // id_last is the shared root: visible from every branch.
                std::vector<llama_seq_id> root_seqs(tree.n_branches);
                for (int b = 0; b < tree.n_branches; ++b) root_seqs[b] = b;
                common_batch_add(batch_tgt, id_last, n_past_before,
                                 root_seqs, /*logits=*/true);

                // Tree nodes. Main-path nodes at depth d carry seqs
                // {0} ∪ {alt branches at deeper depths}. Alt-leaf nodes
                // carry just their own branch_id.
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

                const int64_t t_v0 = ggml_time_us();
                llama_decode(ctx_tgt, batch_tgt);
                t_verify_total += ggml_time_us() - t_v0;

                const int64_t t_a0 = ggml_time_us();
                int current  = 0; // root (= id_last in tree indexing)
                int commit_n = 0;
                while (true) {
                    const llama_token target_token = common_sampler_sample(smpl, ctx_tgt, current);
                    common_sampler_accept(smpl, target_token, true);
                    ids.push_back(target_token);

                    auto it = tree.child_maps[current].find(target_token);
                    if (it == tree.child_maps[current].end()) {
                        break; // bonus token
                    }
                    current  = it->second;
                    commit_n = current;
                }
                t_accept_total += ggml_time_us() - t_a0;

                const int64_t t_r0 = ggml_time_us();
                const bool alt_accepted = (commit_n > tree.main_path_len);

                if (alt_accepted) {
                    n_tree_alt_taken++;

                    // Drop everything in seq 0 from N0 onward (id_last
                    // + main path) and drop every alt seq entirely
                    // (alt slots that were tagged with that seq alone
                    // become empty; the prefix's tags shrink back to
                    // seq 0 only).
                    llama_memory_seq_rm(mem, 0, n_past_before, -1);
                    for (int b = 1; b < tree.n_branches; ++b) {
                        llama_memory_seq_rm(mem, b, -1, -1);
                    }

                    // Re-decode [id_last, alt_token] in seq 0 to
                    // populate KV + captures with correct chain-prefix
                    // hidden states. Stage C alts are leaves at depth
                    // `tree.depths[commit_n - 1]` hanging off
                    // `main_path[depth-1]`; the accept walk only ever
                    // descends one step into an alt branch (alts have
                    // no children in the current Stage C tree shape),
                    // so `[id_last, alt_token]` covers the full
                    // accepted prefix.
                    //
                    // For accepted alt at depth d > 1, we'd want to
                    // re-decode [id_last, m1, ..., m_{d-1}, alt_token]
                    // — d+1 tokens — to match the chain prefix the alt
                    // was committed under. The accept walk's `current`
                    // chain follows the tree.parents pointers from
                    // `commit_n` back to the root, so we can build
                    // the actual prefix from there.
                    common_batch_clear(batch_tgt);
                    common_batch_add(batch_tgt, id_last, n_past_before,
                                     { 0 }, /*logits=*/true);
                    // Walk root -> ... -> commit_n in tree order.
                    std::vector<int> path;
                    for (int node = commit_n; node > 0; node = tree.parents[node]) {
                        path.push_back(node);
                    }
                    std::reverse(path.begin(), path.end());
                    for (size_t i = 0; i < path.size(); ++i) {
                        common_batch_add(batch_tgt, tree.tokens[path[i] - 1],
                                         n_past_before + 1 + (llama_pos) i,
                                         { 0 }, /*logits=*/true);
                    }
                    llama_decode(ctx_tgt, batch_tgt);
                    n_tree_redecoded += (int) batch_tgt.n_tokens;
                    n_past = n_past_before + 1 + (int) path.size();
                } else {
                    // Main-path accept (commit_n == 0 or 1..main_path_len).
                    // Drop the rejected main-path tail, drop every alt
                    // seq entirely. No re-decode needed: captures at
                    // output indices [0..L] are correct (id_last +
                    // accepted main path's tree-decode prefix is the
                    // chain prefix at that depth).
                    const int L = (commit_n > 0) ? tree.depths[commit_n - 1] : 0;
                    llama_memory_seq_rm(mem, 0, n_past_before + 1 + L, -1);
                    for (int b = 1; b < tree.n_branches; ++b) {
                        llama_memory_seq_rm(mem, b, -1, -1);
                    }
                    n_past = n_past_before + 1 + L;
                }
                t_redecode_total += ggml_time_us() - t_r0;
            }

            n_tree_iters++;
            n_drafted_this_iter = (size_t) tree.n_nodes;
        } else {
            // ============================================================
            // Chain mode (unchanged from pre-Stage-B behaviour)
            // ============================================================
            const int64_t t_d0 = ggml_time_us();
            llama_tokens draft = common_speculative_gen_draft(spec, params_spec, prompt_tgt, id_last);
            t_draft_total += ggml_time_us() - t_d0;

            //LOG_DBG("draft: %s\n", string_from(ctx_dft, draft).c_str());

            // always have a token to evaluate from before - id_last
            common_batch_clear(batch_tgt);
            common_batch_add  (batch_tgt, id_last, n_past++, { 0 }, true);

            // evaluate the target model on [id_last, draft0, draft1, ..., draftN-1]
            const int64_t t_v0 = ggml_time_us();
            {
                // do not waste time on small drafts (DRAFT only — DFlash blocks
                // are always exactly block_size-1 tokens by construction)
                if (draft.size() < (size_t) n_draft_min) {
                    draft.clear();
                }

                for (size_t i = 0; i < draft.size(); ++i) {
                    common_batch_add(batch_tgt, draft[i], n_past + i, { 0 }, true);
                }

                //LOG_DBG("target batch: %s\n", string_from(ctx_tgt, batch_tgt).c_str());

                llama_decode(ctx_tgt, batch_tgt);
            }
            t_verify_total += ggml_time_us() - t_v0;

            // sample from the full target batch and return the accepted tokens based on the target sampler
            //
            // for each token to be accepted, the sampler would have to sample that same token
            // in such cases, instead of decoding the sampled token as we normally do, we simply continue with the
            // available logits from the batch and sample the next token until we run out of logits or the sampler
            // disagrees with the draft
            //
            const int64_t t_a0 = ggml_time_us();
            ids = common_sampler_sample_and_accept_n(smpl, ctx_tgt, draft);
            t_accept_total += ggml_time_us() - t_a0;

            //LOG_DBG("ids: %s\n", string_from(ctx_tgt, ids).c_str());

            n_past += ids.size() - 1;
            n_drafted_this_iter = draft.size();
        }
        t_iter_total += ggml_time_us() - t_iter_start;

        GGML_ASSERT(ids.size() > 0); // there will always be at least one accepted token

        n_drafted += n_drafted_this_iter; // note: we ignore the discarded small drafts
        n_accept  += ids.size() - 1;

        // process the accepted tokens and update contexts
        //
        // this is the standard token post-processing that we normally do
        // in this case, we do it for a group of accepted tokens at once
        //
        // We cap at exactly params.n_predict tokens (rather than
        // n_predict_pre_loop + ids.size(), which can overshoot by up to
        // ids.size() - 1 and produces an extra "trailing token" that's
        // confusing for byte-exact comparisons against llama-cli output).
        // The cap is shared across DRAFT and DFLASH so that
        // `--n-predict N` always produces exactly N generated tokens
        // (or stops early on EOG).
        const int n_predict_pre_loop = n_predict;
        for (size_t i = 0; i < ids.size(); ++i) {
            if (params.n_predict >= 0 && (n_predict_pre_loop + (int) i) >= params.n_predict) {
                has_eos = true; // sentinel to break the outer loop
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

        LOG_DBG("accepted %d/%zu draft tokens, the last target token is: (%d)\n", (int) ids.size() - 1, n_drafted_this_iter, id_last);

        {
            LOG_DBG("clear kv cache from any extra tokens, n_past = %d\n", n_past);

            // In tree mode the per-iteration KV rollback (drop tree + re-decode
            // accepted chain) already left seq 0 with exactly n_past slots, so
            // this seq_rm is a no-op there. In chain mode it trims any
            // unconsumed draft slots beyond n_past. Keeping the call
            // unconditional matches the pre-Stage-B behaviour for chain mode
            // and is harmless for tree mode.
            llama_memory_seq_rm(llama_get_memory(ctx_tgt), 0, n_past, -1);
        }

        if ((params.n_predict >= 0 && n_predict >= params.n_predict) || has_eos) {
            break;
        }
    }

    auto t_dec_end = ggml_time_us();

    const int n_input = inp.size();

    LOG("\n\n");

    LOG_INF("encoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_input,   (t_enc_end - t_enc_start) / 1e6f, inp.size() / ((t_enc_end - t_enc_start) / 1e6f));
    LOG_INF("decoded %4d tokens in %8.3f seconds, speed: %8.3f t/s\n", n_predict, (t_dec_end - t_dec_start) / 1e6f, n_predict  / ((t_dec_end - t_dec_start) / 1e6f));

    LOG_INF("\n");
    LOG_INF("draft-type = %s%s\n", is_dflash ? "dflash" : "draft",
            use_tree ? " (tree, Stage B)" : "");
    LOG_INF("n_draft   = %d%s\n", n_draft, is_dflash ? " (block_size)" : "");
    LOG_INF("n_predict = %d\n", n_predict);
    LOG_INF("n_drafted = %d\n", n_drafted);
    LOG_INF("n_accept  = %d\n", n_accept);
    LOG_INF("accept    = %.3f%%\n", n_drafted > 0 ? 100.0f * n_accept / n_drafted : 0.0f);
    if (use_tree) {
        LOG_INF("tree iters       = %d\n", n_tree_iters);
        LOG_INF("tree alt taken   = %d (%.1f%%)\n", n_tree_alt_taken,
                n_tree_iters > 0 ? 100.0f * n_tree_alt_taken / n_tree_iters : 0.0f);
        LOG_INF("tree re-decoded  = %d tokens (%.2f/iter avg)\n", n_tree_redecoded,
                n_tree_iters > 0 ? (float) n_tree_redecoded / n_tree_iters : 0.0f);
    }
    {
        const int64_t accounted = t_draft_total + t_verify_total + t_accept_total + t_redecode_total;
        const int64_t other     = t_iter_total > accounted ? t_iter_total - accounted : 0;
        LOG_INF("\nper-phase timing (loop only):\n");
        LOG_INF("  draft     = %8.2f ms (%5.1f%%)\n", t_draft_total / 1e3,
                t_iter_total > 0 ? 100.0 * t_draft_total / t_iter_total : 0.0);
        LOG_INF("  verify    = %8.2f ms (%5.1f%%)\n", t_verify_total / 1e3,
                t_iter_total > 0 ? 100.0 * t_verify_total / t_iter_total : 0.0);
        LOG_INF("  accept    = %8.2f ms (%5.1f%%)\n", t_accept_total / 1e3,
                t_iter_total > 0 ? 100.0 * t_accept_total / t_iter_total : 0.0);
        LOG_INF("  re-decode = %8.2f ms (%5.1f%%)  %s\n", t_redecode_total / 1e3,
                t_iter_total > 0 ? 100.0 * t_redecode_total / t_iter_total : 0.0,
                use_tree ? "(tree mode only)" : "(chain mode = 0)");
        LOG_INF("  other     = %8.2f ms (%5.1f%%)\n", other / 1e3,
                t_iter_total > 0 ? 100.0 * other / t_iter_total : 0.0);
        LOG_INF("  TOTAL     = %8.2f ms\n", t_iter_total / 1e3);
    }

    LOG_INF("\n");
    LOG_INF("draft:\n\n");

    llama_perf_context_print(ctx_dft);

    LOG_INF("\n");
    LOG_INF("target:\n\n");
    common_perf_print(ctx_tgt, smpl);

    common_sampler_free(smpl);
    common_speculative_free(spec);

    llama_backend_free();

    LOG("\n\n");

    return 0;
}
