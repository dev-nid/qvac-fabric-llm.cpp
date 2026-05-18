#include "llama-context.h"

#include "ggml.h"
#include "llama-arch.h"
#include "llama-impl.h"
#include "llama-batch.h"
#include "llama-io.h"
#include "llama-memory.h"
#include "llama-memory-hybrid.h"
#include "llama-memory-recurrent.h"
#include "llama-mmap.h"
#include "llama-model.h"
#include "models/models.h"   // for llama_model_dflash::encode_graph
#include "llama-ext.h"
#include "llama.h"

#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

//
// llama_context
//

llama_context::llama_context(
        const llama_model & model,
              llama_context_params params) :
    model(model),
    cvec(std::make_unique<llama_adapter_cvec>()),
    loras(std::make_unique<llama_adapter_loras>()),
    balloc(std::make_unique<llama_batch_allocr>(model.hparams.n_pos_per_embd())) {
    // TODO warning when creating llama_context with awkward ctx size that is not a power of 2,
    //     may need to be backend-dependent
    LLAMA_LOG_INFO("%s: constructing llama_context\n", __func__);

    t_start_us = model.t_start_us;
    t_load_us  = model.t_load_us;

    const auto & hparams = model.hparams;

    cparams.n_seq_max = std::max(1u, params.n_seq_max);
    if (cparams.n_seq_max > LLAMA_MAX_SEQ) {
        throw std::runtime_error("n_seq_max must be <= " + std::to_string(LLAMA_MAX_SEQ));
    }

    cparams.n_threads        = params.n_threads;
    cparams.n_threads_batch  = params.n_threads_batch;
    cparams.yarn_ext_factor  = params.yarn_ext_factor  >= 0.0f ? params.yarn_ext_factor  : hparams.yarn_ext_factor;
    cparams.yarn_attn_factor = params.yarn_attn_factor >= 0.0f ? params.yarn_attn_factor : hparams.yarn_attn_factor;
    cparams.yarn_beta_fast   = params.yarn_beta_fast   >= 0.0f ? params.yarn_beta_fast   : hparams.yarn_beta_fast;
    cparams.yarn_beta_slow   = params.yarn_beta_slow   >= 0.0f ? params.yarn_beta_slow   : hparams.yarn_beta_slow;
    cparams.embeddings       = params.embeddings;
    cparams.offload_kqv      = params.offload_kqv;
    cparams.no_perf          = params.no_perf;
    cparams.pooling_type     = params.pooling_type;
    cparams.warmup           = false;

    cparams.n_ctx            = params.n_ctx           == 0    ? hparams.n_ctx_train           : params.n_ctx;
    cparams.rope_freq_base   = params.rope_freq_base  == 0.0f ? hparams.rope_freq_base_train  : params.rope_freq_base;
    cparams.rope_freq_scale  = params.rope_freq_scale == 0.0f ? hparams.rope_freq_scale_train : params.rope_freq_scale;

    cparams.n_ctx_orig_yarn  = params.yarn_orig_ctx    != 0 ? params.yarn_orig_ctx    :
                               hparams.n_ctx_orig_yarn != 0 ? hparams.n_ctx_orig_yarn :
                                                              hparams.n_ctx_train;

    cparams.cb_eval           = params.cb_eval;
    cparams.cb_eval_user_data = params.cb_eval_user_data;

    // Initialize backend samplers here so they are part of the sampling graph
    // before the reserve passes run later in this function. This avoids a later
    // re-reserve when graph nodes change.
    if (params.samplers != nullptr && params.n_samplers > 0) {
        for (size_t i = 0; i < params.n_samplers; ++i) {
            const auto & config = params.samplers[i];

            if (llama_sampler_chain_get(config.sampler, -1) == nullptr) {
                throw std::runtime_error("the backend samplers must be of type llama_sampler_chain");
            }

            if (set_sampler(config.seq_id, config.sampler)) {
                const int n_samplers = llama_sampler_chain_n(config.sampler);

                LLAMA_LOG_INFO("%s: setting backend sampler for seq_id %d (n = %d)\n", __func__, config.seq_id, n_samplers);
            }
        }
    }

    auto rope_scaling_type = params.rope_scaling_type;
    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED) {
        rope_scaling_type = hparams.rope_scaling_type_train;
    }

    if (rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_NONE) {
        cparams.rope_freq_scale = 1.0f; // never scale if scaling type is none
    }

    if (cparams.yarn_ext_factor < 0.0f) { // negative indicates 'not set'
        cparams.yarn_ext_factor = rope_scaling_type == LLAMA_ROPE_SCALING_TYPE_YARN ? 1.0f : 0.0f;
    }

    if (cparams.yarn_ext_factor != 0) {
        static auto get_mscale = [](float scale, float mscale) {
            return scale <= 1.0f ? 1.0f : (0.1f * mscale * logf(scale) + 1.0f);
        };

        const float factor = 1.0f / cparams.rope_freq_scale;

        // ref: https://github.com/huggingface/transformers/blob/6d00f6b0a5679c36510f203e4226e36f517c3032/src/transformers/modeling_rope_utils.py#L336-L348
        if (hparams.rope_yarn_log_mul != 0.0f) {
            // note: here we assume `mscale == 1.0f`
            // TODO: start reading the actual value of mscale and handle the case where it is not 1.0f
                  float mscale          = 1.0f;
            const float mscale_all_dims = hparams.rope_yarn_log_mul;

            // [TAG_DEEPSEEK2_YARN_LOG_MUL_FIX]
            // special-case DEEPSEEK v2:
            // https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/blob/main/config.json#L42-L43
            if (model.arch == LLM_ARCH_DEEPSEEK2 && mscale_all_dims != 1.0f) {
                mscale = mscale_all_dims;
            }

            cparams.yarn_attn_factor = get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dims);

            LLAMA_LOG_WARN("%s: setting new yarn_attn_factor = %.4f (mscale == %.1f, mscale_all_dim = %.1f)\n",
                    __func__, cparams.yarn_attn_factor, mscale, mscale_all_dims);
        } else {
            cparams.yarn_attn_factor = get_mscale(factor, 1.0f);
        }

        // when YARN is applied with yarn_ext_factor != 0.0f, we need to cancel this factor:
        // https://github.com/ggml-org/llama.cpp/blob/a81a569577cc38b32558958b048228150be63eae/ggml/src/ggml-cpu/ops.cpp#L5541-L5544
        //
        // ref: https://github.com/ggml-org/llama.cpp/discussions/7416
        //      https://github.com/ggml-org/llama.cpp/pull/17945
        cparams.yarn_attn_factor *= 1.0f / (1.0f + 0.1f * logf(factor));
    }

    cparams.yarn_attn_factor *= hparams.rope_attn_factor;

    if (cparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
        if (hparams.pooling_type == LLAMA_POOLING_TYPE_UNSPECIFIED) {
            cparams.pooling_type = LLAMA_POOLING_TYPE_NONE;
        } else {
            cparams.pooling_type = hparams.pooling_type;
        }
    }

    if (params.attention_type == LLAMA_ATTENTION_TYPE_UNSPECIFIED) {
        cparams.causal_attn = hparams.causal_attn;
    } else {
        cparams.causal_attn = params.attention_type == LLAMA_ATTENTION_TYPE_CAUSAL;
    }

    cparams.flash_attn = params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cparams.auto_fa    = params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO;

    cparams.fused_gdn_ar = true;
    cparams.fused_gdn_ch = true;
    cparams.auto_fgdn    = true;

    // with causal attention, the batch size is limited by the context size
    cparams.n_batch = cparams.causal_attn ? std::min(cparams.n_ctx, params.n_batch) : params.n_batch;

    cparams.n_ubatch = std::min(cparams.n_batch, params.n_ubatch == 0 ? params.n_batch : params.n_ubatch);

    // The DFlash drafter decodes EXACTLY one block of `dflash_block_size`
    // tokens per call, with bidirectional intra-block attention (paper §4.1).
    // n_ubatch MUST equal block_size: a smaller ubatch would split the block
    // across multiple llama_decode internal ubatches, each of which only
    // sees its own subset as queries, breaking the bidirectional intra-block
    // attention the draft relies on (acceptance collapses to ~0%). A larger
    // ubatch is harmless but wastes worst-case compute buffer. We force it
    // here regardless of what the user passed via --ubatch-size, because the
    // standalone speculative-dflash example shares params.n_ubatch with the
    // target context (where the user may legitimately want a different
    // value, e.g. -ub 1 for matching against llama-cli).
    if (model.arch == LLM_ARCH_DFLASH) {
        const uint32_t bs = std::max<uint32_t>(1, model.hparams.dflash_block_size);
        if (cparams.n_ubatch != bs) {
            LLAMA_LOG_INFO("%s: DFlash draft: forcing n_ubatch %u -> %u (= dflash_block_size); "
                           "smaller ubatch breaks bidirectional intra-block attention\n",
                           __func__, cparams.n_ubatch, bs);
            cparams.n_ubatch = bs;
        }
        if (cparams.n_batch < bs) {
            cparams.n_batch = bs;
        }
    }

    cparams.op_offload = params.op_offload;
    cparams.kv_unified = params.kv_unified;

    // DFlash-specific cparams (ignored for non-DFlash drafts).
    cparams.dflash_max_ctx     = params.dflash_max_ctx;
    cparams.dflash_topk        = params.dflash_topk == 0 ? 1 : params.dflash_topk;
    cparams.dflash_emit_logits = params.dflash_emit_logits;

    cparams.dflash_inline_encoder        = params.dflash_inline_encoder;
    cparams.dflash_inline_n_embd_dft     = params.dflash_inline_n_embd_dft;
    cparams.dflash_inline_n_head_kv_dft  = params.dflash_inline_n_head_kv_dft;
    cparams.dflash_inline_n_embd_head_dft = params.dflash_inline_n_embd_head_dft;
    cparams.dflash_inline_n_target_layers = params.dflash_inline_n_target_layers;
    cparams.dflash_gdn_history            = params.dflash_gdn_history;
    cparams.dflash_gdn_history_f16        = params.dflash_gdn_history_f16;

    if (cparams.dflash_inline_encoder) {
        // sizing cparams remain optional. When all four are zero, the
        // target's graph builder uses target's own hparams for the inline
        // encoder shapes. This is correct for model families where
        // the DFlash draft was trained to share the target's per-head
        // dimensions (Qwen3-N with Qwen3-DFlash). Models that don't share
        // (Qwen3.5 etc.) must supply the sizing fields explicitly.
        if (model.arch == LLM_ARCH_DFLASH) {
            throw std::runtime_error(
                "dflash_inline_encoder is target-only; do not set it on the "
                "DFlash draft context");
        }
    }

    if (cparams.dflash_gdn_history) {
        if (model.arch == LLM_ARCH_DFLASH) {
            throw std::runtime_error(
                "dflash_gdn_history is target-only; do not set it on the "
                "DFlash draft context");
        }
    }
    if (cparams.dflash_gdn_history_f16 && !cparams.dflash_gdn_history) {
        // Silently clear: f16 has no meaning without the persistent
        // buffer to type. This keeps the public API tolerant of stale
        // flags being passed in (e.g. server reusing a params struct).
        cparams.dflash_gdn_history_f16 = false;
    }

    // initialized later
    cparams.pipeline_parallel = false;

    {
        const char * LLAMA_GRAPH_REUSE_DISABLE = getenv("LLAMA_GRAPH_REUSE_DISABLE");
        graph_reuse_disable = LLAMA_GRAPH_REUSE_DISABLE ? (atoi(LLAMA_GRAPH_REUSE_DISABLE) != 0) : graph_reuse_disable;

        if (graph_reuse_disable) {
            LLAMA_LOG_WARN("%s: graph reuse disabled\n", __func__);
        }
    }

    // ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    cparams.n_ctx = GGML_PAD(cparams.n_ctx, 256);

    if (cparams.kv_unified) {
        cparams.n_ctx_seq = cparams.n_ctx;
    } else {
        cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
        cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);

        if (cparams.n_ctx_seq == 0) {
            throw std::runtime_error("n_ctx_seq == 0");
        }

        if (cparams.n_ctx != cparams.n_ctx_seq * cparams.n_seq_max) {
            cparams.n_ctx =  cparams.n_ctx_seq * cparams.n_seq_max;
            LLAMA_LOG_WARN("%s: n_ctx is not divisible by n_seq_max - rounding down to %u\n", __func__, cparams.n_ctx);
        }
    }

    LLAMA_LOG_INFO("%s: n_seq_max     = %u\n",   __func__, cparams.n_seq_max);
    LLAMA_LOG_INFO("%s: n_ctx         = %u\n",   __func__, cparams.n_ctx);
    LLAMA_LOG_INFO("%s: n_ctx_seq     = %u\n",   __func__, cparams.n_ctx_seq);
    LLAMA_LOG_INFO("%s: n_batch       = %u\n",   __func__, cparams.n_batch);
    LLAMA_LOG_INFO("%s: n_ubatch      = %u\n",   __func__, cparams.n_ubatch);
    LLAMA_LOG_INFO("%s: causal_attn   = %d\n",   __func__, cparams.causal_attn);
    LLAMA_LOG_INFO("%s: flash_attn    = %s\n",   __func__, llama_flash_attn_type_name(params.flash_attn_type));
    LLAMA_LOG_INFO("%s: kv_unified    = %s\n",   __func__, cparams.kv_unified ? "true" : "false");
    LLAMA_LOG_INFO("%s: freq_base     = %.1f\n", __func__, cparams.rope_freq_base);
    LLAMA_LOG_INFO("%s: freq_scale    = %g\n",   __func__, cparams.rope_freq_scale);

    if (cparams.n_ctx_seq < hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) < n_ctx_train (%u) -- the full capacity of the model will not be utilized\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (cparams.n_ctx_seq > hparams.n_ctx_train) {
        LLAMA_LOG_WARN("%s: n_ctx_seq (%u) > n_ctx_train (%u) -- possible training context overflow\n",
                __func__, cparams.n_ctx_seq, hparams.n_ctx_train);
    }

    if (!hparams.vocab_only) {
        // GPU backends
        for (const auto & dev : model.devices) {
            ggml_backend_t backend = ggml_backend_dev_init(dev.dev, nullptr);
            if (backend == nullptr) {
                throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev.dev)));
            }
            backends.emplace_back(backend);
        }

        // add ACCEL backends (such as BLAS)
        for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
            ggml_backend_dev_t dev = ggml_backend_dev_get(i);
            if (ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_ACCEL) {
                ggml_backend_t backend = ggml_backend_dev_init(dev, nullptr);
                if (backend == nullptr) {
                    throw std::runtime_error(format("failed to initialize %s backend", ggml_backend_dev_name(dev)));
                }
                backends.emplace_back(backend);
            }
        }

        // add CPU backend
        backend_cpu = ggml_backend_init_by_type(GGML_BACKEND_DEVICE_TYPE_CPU, nullptr);
        if (backend_cpu == nullptr) {
            throw std::runtime_error("failed to initialize CPU backend");
        }
        backends.emplace_back(backend_cpu);

        // create a list of the set_n_threads functions in the backends
        for (auto & backend : backends) {
            ggml_backend_dev_t dev = ggml_backend_get_device(backend.get());
            ggml_backend_reg_t reg = dev ? ggml_backend_dev_backend_reg(dev) : nullptr;
            if (reg) {
                auto ggml_backend_set_n_threads_fn = (ggml_backend_set_n_threads_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
                if (ggml_backend_set_n_threads_fn) {
                    set_n_threads_fns.emplace_back(backend.get(), ggml_backend_set_n_threads_fn);
                }
            }
        }

        llama_set_abort_callback(this, params.abort_callback, params.abort_callback_data);

        // graph outputs buffer
        {
            if (output_reserve(params.n_seq_max) < params.n_seq_max) {
                throw std::runtime_error("failed to reserve initial output buffer");
            }

            LLAMA_LOG_INFO("%s: %10s  output buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buffer_name    (buf_output.get()),
                    ggml_backend_buffer_get_size(buf_output.get()) / 1024.0 / 1024.0);
        }
    }

    // init the memory module
    if (!hparams.vocab_only) {
        llama_memory_params params_mem = {
            /*.type_k   =*/ params.type_k,
            /*.type_v   =*/ params.type_v,
            /*.swa_full =*/ params.swa_full,
        };

        memory.reset(model.create_memory(params_mem, cparams));
    }

    // init backends
    if (!hparams.vocab_only) {
        LLAMA_LOG_DEBUG("%s: enumerating backends\n", __func__);

        backend_buft.clear();
        backend_ptrs.clear();
        backend_buf_exp_size.clear();

        for (auto & backend : backends) {
            auto * buft = ggml_backend_get_default_buffer_type(backend.get());
            auto backend_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));

            if (backend_type == GGML_BACKEND_DEVICE_TYPE_CPU && !model.devices.empty()) {
                // use the host buffer of the first device CPU for faster transfer of the intermediate state
                const auto & dev = model.devices[0];
                auto * host_buft = ggml_backend_dev_host_buffer_type(dev.dev);
                if (host_buft) {
                    buft = host_buft;
                }
            }

            backend_buft.push_back(buft);
            backend_ptrs.push_back(backend.get());
            backend_buf_exp_size.push_back(0);
        }

        LLAMA_LOG_DEBUG("%s: backend_ptrs.size() = %zu\n", __func__, backend_ptrs.size());

        // TODO: move these checks to ggml_backend_sched
        // enabling pipeline parallelism in the scheduler increases memory usage, so it is only done when necessary
        bool pipeline_parallel =
            model.n_devices() > 1 &&
            model.n_gpu_layers() > model.hparams.n_layer &&
            model.split_mode() == LLAMA_SPLIT_MODE_LAYER &&
            cparams.offload_kqv &&
            !model.has_tensor_overrides();

        // pipeline parallelism requires support for async compute and events in all devices
        if (pipeline_parallel) {
            for (auto & backend : backends) {
                auto dev_type = ggml_backend_dev_type(ggml_backend_get_device(backend.get()));
                if (dev_type == GGML_BACKEND_DEVICE_TYPE_CPU) {
                    // ignore CPU backend
                    // TODO: should we ignore ACCEL types too?
                    continue;
                }
                auto * dev = ggml_backend_get_device(backend.get());
                ggml_backend_dev_props props;
                ggml_backend_dev_get_props(dev, &props);
                if (!props.caps.async || !props.caps.events) {
                    // device does not support async compute or events
                    pipeline_parallel = false;
                    break;
                }
            }
        }

        cparams.pipeline_parallel = pipeline_parallel;

        if (cparams.pipeline_parallel) {
            LLAMA_LOG_INFO("%s: pipeline parallelism enabled\n", __func__);
        }

        sched_reserve();

        if (!cparams.flash_attn) {
            if (ggml_is_quantized(params.type_v)) {
                throw std::runtime_error("quantized V cache was requested, but this requires Flash Attention");
            }
        }

        // DFlash encoder warmup. The first call to `dflash_extend` for any
        // given `n_new` shape compiles backend pipelines (cold cost
        // ~100-300 ms on Vulkan, paid lazily on the first user
        // generation). The per-`n_keep_pad` slot cache in
        // `gf_res_dflash_encode_slots` retains a built graph per width;
        // this loop populates every valid index in [1, dflash_block_size]
        // so steady-state chain mode (where n_keep_pad = 1 + accept_count
        // varies per iter) never rebuilds the encoder.
        //
        // Each warmup call pushes `bs` zero rows then `dflash_reset_ctx_kv()`
        // restores ctx_filled/ctx_pos_base to 0; the side-store data
        // bytes remain zero but are not visible to subsequent extends
        // (n_ctx is reset as well).
        if (model.arch == LLM_ARCH_DFLASH
                && dflash.n_features > 0
                && !dflash.ctx_K.empty()) {
            const uint32_t bs = std::max<uint32_t>(1, model.hparams.dflash_block_size);
            const int64_t  nf = dflash.n_features;
            std::vector<float> zero_buf((size_t) bs * (size_t) nf, 0.0f);

            const int64_t t_warm_start = ggml_time_us();
            int n_warmed = 0;
            int n_failed = 0;
            // Warm in descending order so n_new=bs (largest) is built
            // first; subsequent smaller n_new builds reuse the same
            // backend buffer pool layout where possible.
            for (int64_t n = (int64_t) bs; n >= 1; --n) {
                const int rc = dflash_extend(zero_buf.data(), n, /*pos_start=*/ 0);
                dflash_reset_ctx_kv();
                if (rc == 0) ++n_warmed; else ++n_failed;
            }

            const int64_t t_warm_us = ggml_time_us() - t_warm_start;
            if (n_failed > 0) {
                LLAMA_LOG_WARN("%s: DFlash encoder warmup partially failed (warmed=%d failed=%d of %u) — first generation may incur cold pipeline-compile cost\n",
                               __func__, n_warmed, n_failed, bs);
            } else {
                LLAMA_LOG_INFO("%s: DFlash encoder warmup: pre-compiled pipelines for n_new=[1..%u] in %.2f ms\n",
                               __func__, bs, t_warm_us / 1000.0);
            }
        }
    }

    // Initialize the full vocabulary token ids for backend samplers.
    {
        const int n_vocab = model.vocab.n_tokens();

        sampling.token_ids_full_vocab.resize(n_vocab);
        for (int i = 0; i < n_vocab; ++i) {
            sampling.token_ids_full_vocab[i] = i;
        }
    }
}

llama_context::~llama_context() {
    if (!model.hparams.no_alloc) {
        for (size_t i = 0; i < backend_ptrs.size(); ++i) {
            ggml_backend_t             backend = backend_ptrs[i];
            ggml_backend_buffer_type_t buft    = backend_buft[i];

            const size_t size_exp = backend_buf_exp_size[i];
            const size_t size_act = ggml_backend_sched_get_buffer_size(sched.get(), backend);
            if (size_exp == size_act) {
                LLAMA_LOG_DEBUG("%s: %10s compute buffer size is %8.4f MiB, matches expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            } else {
                LLAMA_LOG_WARN("%s: %10s compute buffer size of %8.4f MiB, does not match expectation of %8.4f MiB\n",
                    __func__, ggml_backend_buft_name(buft), size_act / (1024.0*1024.0), size_exp / (1024.0*1024.0));
            }
        }
    }
    ggml_opt_free(opt_ctx);
}

void llama_context::sched_reserve() {
    if (!sched_need_reserve) {
        return;
    }

    sched_need_reserve = false;

    LLAMA_LOG_INFO("%s: reserving ...\n", __func__);

    synchronize();

    const int64_t t_start_us = ggml_time_us();

    const uint32_t n_seqs = cparams.n_seq_max;
    const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

    const size_t max_nodes = this->graph_max_nodes(n_tokens);

    LLAMA_LOG_DEBUG("%s: max_nodes = %zu\n", __func__, max_nodes);

    gf_res_prev.reset(new llm_graph_result(max_nodes));
    gf_res_reserve.reset(new llm_graph_result(max_nodes));

    sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, cparams.pipeline_parallel, cparams.op_offload));

    llama_memory_context_ptr mctx;
    if (memory) {
        LLAMA_LOG_DEBUG("%s: reserving full memory module\n", __func__);
        mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory module");
        }
    }

    // Seed worst-case shapes for the DFlash drafter graph BEFORE
    // graph_reserve runs. Idempotent — only runs the first time
    // sched_reserve is called (when ctx_K is empty).
    if (model.arch == LLM_ARCH_DFLASH && dflash.ctx_K.empty() && model.dflash_fc != nullptr) {
        // Skip DFlash side-store init for synthetic test-llama-archs GGUFs
        // that lack dflash_fc; real DFlash GGUFs always have it.
        dflash.n_features = model.dflash_fc->ne[0];

        // dflash_max_ctx semantics: -1 = auto-scale, 0 = uncapped, >0 = explicit.
        int64_t ctx_capacity_seed;
        if (cparams.dflash_max_ctx == -1) {
            const int64_t scaled = (int64_t) cparams.n_ctx_seq / 4;
            ctx_capacity_seed = std::min<int64_t>(std::max<int64_t>(scaled, 512), 1024);
        } else if (cparams.dflash_max_ctx > 0) {
            ctx_capacity_seed = (int64_t) cparams.dflash_max_ctx;
        } else {
            ctx_capacity_seed = (int64_t) cparams.n_ctx_seq;
        }
        ctx_capacity_seed = std::min<int64_t>(ctx_capacity_seed, (int64_t) cparams.n_ctx_seq);
        dflash.n_ctx      = ctx_capacity_seed;

        // ---------- DFlash K/V side store (paper §4.1 reuse) ----------
        {
            const uint32_t n_layer        = model.hparams.n_layer;
            const int64_t  n_embd_k_gqa   = model.hparams.n_embd_k_gqa();
            const int64_t  n_embd_v_gqa   = model.hparams.n_embd_v_gqa();
            // Use F16 by default for the side store (matches kv-cache default).
            const ggml_type type_k        = GGML_TYPE_F16;
            const ggml_type type_v        = GGML_TYPE_F16;
            int64_t ctx_capacity = ctx_capacity_seed;

            struct ggml_backend_buft_comparator {
                bool operator()(const ggml_backend_buffer_type_t & lhs,
                                const ggml_backend_buffer_type_t & rhs) const {
                    return strcmp(ggml_backend_buft_name(lhs),
                                  ggml_backend_buft_name(rhs)) < 0;
                }
            };
            std::map<ggml_backend_buffer_type_t, ggml_context_ptr,
                     ggml_backend_buft_comparator> ctx_map;

            auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
                auto it = ctx_map.find(buft);
                if (it != ctx_map.end()) return it->second.get();
                ggml_init_params p = {
                    /*.mem_size   =*/ size_t(2u * n_layer * ggml_tensor_overhead()),
                    /*.mem_buffer =*/ NULL,
                    /*.no_alloc   =*/ true,
                };
                ggml_context * c = ggml_init(p);
                ctx_map.emplace(buft, c);
                return c;
            };

            dflash.ctx_K.resize(n_layer, nullptr);
            dflash.ctx_V.resize(n_layer, nullptr);
            dflash.ctx_capacity = ctx_capacity;
            dflash.ctx_filled   = 0;
            dflash.ctx_pos_base = 0;

            for (uint32_t il = 0; il < n_layer; ++il) {
                ggml_backend_dev_t        dev  = model.dev_layer(il);
                ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
                ggml_context * c              = ctx_for_buft(buft);
                GGML_ASSERT(c != nullptr);

                ggml_tensor * k = ggml_new_tensor_2d(c, type_k, n_embd_k_gqa, ctx_capacity);
                ggml_tensor * v = ggml_new_tensor_2d(c, type_v, n_embd_v_gqa, ctx_capacity);
                ggml_format_name(k, "dflash_ctx_K_l%u", il);
                ggml_format_name(v, "dflash_ctx_V_l%u", il);

                dflash.ctx_K[il] = k;
                dflash.ctx_V[il] = v;
            }

            size_t total_bytes = 0;
            for (auto & [buft, c] : ctx_map) {
                ggml_backend_buffer_t buf =
                    ggml_backend_alloc_ctx_tensors_from_buft(c.get(), buft);
                if (!buf) {
                    throw std::runtime_error(
                        "failed to allocate buffer for DFlash K/V side store");
                }
                LLAMA_LOG_INFO("%s: %10s DFlash K/V side store size = %8.2f MiB\n",
                        __func__, ggml_backend_buffer_name(buf),
                        ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
                total_bytes += ggml_backend_buffer_get_size(buf);
                ggml_backend_buffer_clear(buf, 0);
                dflash_kv_ctxs_bufs.emplace_back(std::move(c),
                    ggml_backend_buffer_ptr(buf));
            }
            LLAMA_LOG_INFO("%s: DFlash K/V side store: total %.2f MiB across %d layers, capacity %u\n",
                    __func__, total_bytes/1024.0/1024.0,
                    (int) n_layer, (uint32_t) ctx_capacity);
        }

        // Allocate dedicated scheduler(s) for the encoder graph. One slot
        // per `n_keep_pad` ∈ [1, dflash_block_size]; index 0 is unused so
        // the slot index can be the n_keep_pad value directly. Each slot
        // gets its own scheduler because compute-buffer sizing is shape-
        // dependent and `ggml_backend_sched_alloc_graph` is one-shot per
        // scheduler. Lazily filled by ensure_dflash_encode_slot() and
        // pre-warmed for the full range below in the same lifecycle pass.
        const int max_nodes_enc = std::max<int>(8192, (int) model.hparams.n_layer * 32);
        const uint32_t bs_enc = std::max<uint32_t>(1, model.hparams.dflash_block_size);
        sched_dflash_encode_slots.resize((size_t) bs_enc + 1);
        gf_res_dflash_encode_slots.resize((size_t) bs_enc + 1);
        for (uint32_t i = 1; i <= bs_enc; ++i) {
            sched_dflash_encode_slots[i].reset(ggml_backend_sched_new(
                backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                max_nodes_enc, /*parallel=*/ false, cparams.op_offload));
        }
    }

    // DFlash GDN history buffers: allocate one persistent state-history
    // buffer per recurrent (GDN)
    // layer on the TARGET context. Sized for the worst-case chain-verify
    // batch (gdn_history_max_tokens). One device buffer per backing
    // device (mirrors the side-store allocation pattern above).
    //
    // The graph builder will emit ggml_cpy(state_history_view ->
    // dflash.gdn_history[il]) at the end of each GDN layer in chain mode;
    // the spec driver consumes those tensors via the fixup graph.
    if (cparams.dflash_gdn_history
            && dflash.gdn_history.empty()
            && model.arch != LLM_ARCH_DFLASH) {
        // Chain mode cap = 1 + dflash_block_size_default = 17. The
        // tree-verify worst case is `id_last + tree.budget` tokens. The cap
        // is sized to fit both modes; tree-mode is gated on
        // cparams.n_seq_max > 1, in which case we size for a tree budget
        // around 22 with a small headroom.
        const bool tree_mode_alloc = (cparams.n_seq_max > 1);
        const int64_t DFLASH_GDN_HISTORY_MAX_TOKENS = tree_mode_alloc
            ? 32   // typical tree budget (~22) + root, rounded up
            : 17;  // chain: id_last + 16-token draft block

        // Per-seq dimension of the persistent buffer. Tree mode here
        // means the single-seq DFS-flattened layout: all tree tokens go
        // through seq 0 with attention separation by tree_mask and GDN
        // branching by
        // parent_ids. The recurrent ubatch's n_seqs is therefore 1 at
        // runtime even though cparams.n_seq_max may be much larger
        // (the spec driver bumps n_parallel = budget + 1 so the KV
        // cache reserves enough cells for the legacy multi-seq
        // alt-accept path — that path is dormant when this flag is on).
        //
        // Keeping n_seqs_max = 1 here keeps the persistent buffer small;
        // a per-seq buffer would multiply the per-layer footprint by the
        // tree budget and would not fit alongside weights + KV cache on
        // 32 GiB devices.
        const int64_t n_seqs_max = 1;

        const uint32_t n_layer = model.hparams.n_layer;

        // GDN per-head dims. For qwen3.5 these come from ssm_*:
        //   S_v == S_k == head_v_dim == head_k_dim == ssm_d_state
        //   H_v == ssm_dt_rank
        // Same hparams the graph builder uses in qwen35.cpp.
        const int64_t S_v = model.hparams.ssm_d_state;
        const int64_t H_v = model.hparams.ssm_dt_rank;

        if (S_v <= 0 || H_v <= 0) {
            throw std::runtime_error(
                "dflash_gdn_history: model lacks GDN dims "
                "(ssm_d_state / ssm_dt_rank). Only Qwen3.5-family targets "
                "are supported.");
        }

        struct ggml_backend_buft_comparator {
            bool operator()(const ggml_backend_buffer_type_t & lhs,
                            const ggml_backend_buffer_type_t & rhs) const {
                return strcmp(ggml_backend_buft_name(lhs),
                              ggml_backend_buft_name(rhs)) < 0;
            }
        };
        std::map<ggml_backend_buffer_type_t, ggml_context_ptr,
                 ggml_backend_buft_comparator> ctx_map;
        auto ctx_for_buft = [&](ggml_backend_buffer_type_t buft) -> ggml_context * {
            auto it = ctx_map.find(buft);
            if (it != ctx_map.end()) return it->second.get();
            ggml_init_params p = {
                // 2 tensors per recurrent layer (gdn_history + conv_history);
                // pad generously to absorb any future per-layer additions.
                /*.mem_size   =*/ size_t(4u * n_layer * ggml_tensor_overhead()),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };
            ggml_context * c = ggml_init(p);
            ctx_map.emplace(buft, c);
            return c;
        };

        dflash.gdn_history.assign(n_layer, nullptr);
        dflash.conv_history.assign(n_layer, nullptr);
        dflash.gdn_history_max_tokens = DFLASH_GDN_HISTORY_MAX_TOKENS;
        dflash.gdn_history_n_seqs_max = n_seqs_max;

        // Pick the persistent-buffer dtype. Tree mode mandates F16 on
        // the 27B target (the F32 buffer × n_seqs_max doesn't fit
        // alongside weights + KV cache on 32 GiB). Auto-promote when
        // the user enabled tree mode without explicitly opting into
        // F16; chain mode stays F32 unless explicitly requested.
        if (tree_mode_alloc && !cparams.dflash_gdn_history_f16) {
            LLAMA_LOG_INFO(
                "%s: dflash_gdn_history: auto-promoting persistent "
                "buffer to F16 (tree mode; n_seqs_max=%lld). Pass "
                "--dflash-gdn-history-f16 to make this explicit.\n",
                __func__, (long long) n_seqs_max);
            cparams.dflash_gdn_history_f16 = true;
        }
        const ggml_type gdn_history_type = cparams.dflash_gdn_history_f16
            ? GGML_TYPE_F16 : GGML_TYPE_F32;

        // Conv state companion (per-layer ssm_conv state history).
        // Layout: [conv_kernel_size - 1 + max_tokens, conv_channels, n_seqs].
        // Conv kernel reads (conv_kernel_size - 1) previous-token
        // representations + the new tokens; we persist the full
        // concat(old_conv_state, qkv_mixed) so the conv-state fixup can
        // pick the right 3 rows on partial acceptance.
        const int64_t conv_kernel_size = model.hparams.ssm_d_conv;
        const int64_t conv_channels    = model.hparams.ssm_d_inner
            + 2 * model.hparams.ssm_n_group * model.hparams.ssm_d_state;
        GGML_ASSERT(conv_kernel_size > 0);
        GGML_ASSERT(conv_channels    > 0);
        const int64_t conv_history_rows =
            (conv_kernel_size - 1) + DFLASH_GDN_HISTORY_MAX_TOKENS;

        uint32_t n_gdn_layers = 0;
        for (uint32_t il = 0; il < n_layer; ++il) {
            if (!model.hparams.is_recurrent(il)) continue;

            ggml_backend_dev_t dev = model.dev_layer(il);
            ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
            ggml_context * c = ctx_for_buft(buft);
            GGML_ASSERT(c != nullptr);

            // Layout: [S_v, S_v, H_v, max_tokens * n_seqs]. Ggml supports
            // only 4-D; the 5-D semantic ([S_v, S_v, H_v, max_tokens, n_seqs])
            // is recovered by treating the last dim as
            // (max_tokens * n_seqs). Chain mode keeps n_seqs == 1, so
            // ne[3] == max_tokens. Tree mode widens to
            // ne[3] == max_tokens * n_seqs_max.
            //
            // dtype is F32 in chain mode and F16 in tree mode (smaller
            // persistent footprint). The GDN kernel selects InterT at
            // runtime from this tensor's type.
            ggml_tensor * t = ggml_new_tensor_4d(
                c, gdn_history_type,
                S_v, S_v, H_v,
                DFLASH_GDN_HISTORY_MAX_TOKENS * n_seqs_max);
            ggml_format_name(t, "dflash_gdn_history_l%u", il);
            dflash.gdn_history[il] = t;

            // Conv history (full conv_input snapshot per decode).
            ggml_tensor * c_hist = ggml_new_tensor_3d(
                c, GGML_TYPE_F32,
                conv_history_rows, conv_channels, n_seqs_max);
            ggml_format_name(c_hist, "dflash_conv_history_l%u", il);
            dflash.conv_history[il] = c_hist;

            ++n_gdn_layers;
        }

        size_t total_bytes = 0;
        for (auto & [buft, c] : ctx_map) {
            ggml_backend_buffer_t buf =
                ggml_backend_alloc_ctx_tensors_from_buft(c.get(), buft);
            if (!buf) {
                throw std::runtime_error(
                    "failed to allocate buffer for DFlash GDN history");
            }
            LLAMA_LOG_INFO("%s: %10s DFlash GDN history size = %8.2f MiB\n",
                    __func__, ggml_backend_buffer_name(buf),
                    ggml_backend_buffer_get_size(buf)/1024.0/1024.0);
            total_bytes += ggml_backend_buffer_get_size(buf);
            ggml_backend_buffer_clear(buf, 0);
            dflash_gdn_history_ctxs_bufs.emplace_back(std::move(c),
                ggml_backend_buffer_ptr(buf));
        }
        LLAMA_LOG_INFO(
            "%s: DFlash GDN history: total %.2f MiB across %u GDN layers, "
            "max_n_tokens = %d, n_seqs_max = %lld, dtype = %s "
            "(S_v=%lld, H_v=%lld)\n",
            __func__,
            total_bytes/1024.0/1024.0,
            n_gdn_layers,
            (int) DFLASH_GDN_HISTORY_MAX_TOKENS,
            (long long) n_seqs_max,
            cparams.dflash_gdn_history_f16 ? "F16" : "F32",
            (long long) S_v, (long long) H_v);

        // Memory-budget guard. Refuse to enable history+tree when the
        // projected gdn_history allocation would exceed available VRAM
        // on the primary device. The check is approximate (uses ggml's
        // reported free) and skipped when n_devices > 1 (we can't
        // tell where the model wants to land).
        if (tree_mode_alloc && model.devices.size() == 1) {
            ggml_backend_dev_t dev = model.devices[0].dev;
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(dev, &props);
            const size_t free_after = props.memory_free;
            // Coarse threshold: the model + KV cache is typically the
            // bulk; if we now have <1 GiB free after this allocation
            // we're likely about to OOM mid-decode. Log a warning
            // rather than aborting — the user may have headroom we
            // can't see (host-pinned, multi-GPU). The actual abort
            // surfaces from cudaMalloc later if the budget really is
            // tight.
            if (free_after < (size_t) 1024 * 1024 * 1024) {
                LLAMA_LOG_WARN(
                    "%s: DFlash GDN history (tree mode): only %.2f MiB "
                    "free on %s after persistent buffer allocation. "
                    "Consider lowering --parallel (current n_seq_max=%u) "
                    "or keeping --dflash-gdn-history-f16 set if you "
                    "passed --no-dflash-gdn-history-f16 to disable it.\n",
                    __func__, free_after / 1024.0 / 1024.0,
                    ggml_backend_dev_name(dev),
                    cparams.n_seq_max);
            }
        }
    }

    // avoid reserving graphs with zero outputs - assume one output per sequence
    const int n_outputs = n_seqs;

    LLAMA_LOG_DEBUG("%s: worst-case: n_tokens = %d, n_seqs = %d, n_outputs = %d\n", __func__, n_tokens, n_seqs, n_outputs);

    // resolve automatic Flash Attention use
    if (cparams.auto_fa) {
        auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
        if (!gf) {
            throw std::runtime_error("failed to reserve graph for Flash Attention check");
        }

        const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FATTN) + 1;
        bool fa_device_mismatch = false;
        for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
            ggml_tensor * n = ggml_graph_node(gf, i);
            if (n->op != GGML_OP_FLASH_ATTN_EXT) {
                continue;
            }
            ggml_backend_dev_t device_fa = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

            // TODO: instead of the tensor names, use a map to keep track of which (FA) tensors belong to which layer
            GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FATTN "-", prefix_len) == 0);
            const int il = std::stoi(n->name + prefix_len);
            ggml_backend_dev_t device_kv = model.dev_layer(il);
            if (device_fa != device_kv) {
                LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the Flash Attention tensor "
                        "is assigned to device %s (usually due to missing support)\n",
                        __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_fa));
                // FIXME: fa_device_mismatch logic is wrong for --no-kv-offload, but this is broken anyways
                fa_device_mismatch = true;
                break;
            }
        }

        if (fa_device_mismatch) {
            cparams.flash_attn = false;
            LLAMA_LOG_WARN("%s: Flash Attention was auto, set to disabled\n", __func__);
        } else {
            cparams.flash_attn = true;
            LLAMA_LOG_INFO("%s: Flash Attention was auto, set to enabled\n", __func__);
        }

        cparams.auto_fa = false;
    }

    if (cparams.auto_fgdn) {
        LLAMA_LOG_INFO("%s: resolving fused Gated Delta Net support:\n", __func__);

        if (cparams.fused_gdn_ar) {
            auto * gf = graph_reserve(1, n_seqs, n_outputs, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (autoregressive)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_AR) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_AR "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ar = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (autoregressive) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (autoregressive) enabled\n", __func__);
            }
        }

        if (cparams.fused_gdn_ch) {
            // more than one token in the batch per sequence in order to take the chunked path
            // note: n_outputs must match n_tokens for embedding models with mean/rank pooling,
            // because build_pooling creates inp_mean with shape [n_tokens, n_seqs] and multiplies
            // it with t_embd which is reduced to [n_outputs, ...] via out_ids. if n_outputs != n_tokens,
            // the ggml_mul_mat assertion fails. this matches the pp reservation below (line ~553).
            const uint32_t n_tokens_ch = 16*n_seqs;
            auto * gf = graph_reserve(n_tokens_ch, n_seqs, n_tokens_ch, mctx.get(), true);
            if (!gf) {
                throw std::runtime_error("failed to reserve graph for fused Gated Delta Net check (chunked)");
            }

            const size_t prefix_len = strlen(LLAMA_TENSOR_NAME_FGDN_CH) + 1;
            bool gdn_device_mismatch = false;
            for (int i = 0; i < ggml_graph_n_nodes(gf); i++) {
                ggml_tensor * n = ggml_graph_node(gf, i);
                if (n->op != GGML_OP_GATED_DELTA_NET) {
                    continue;
                }
                ggml_backend_dev_t device_gdn = ggml_backend_get_device(ggml_backend_sched_get_tensor_backend(sched.get(), n));

                GGML_ASSERT(strncmp(n->name, LLAMA_TENSOR_NAME_FGDN_CH "-", prefix_len) == 0);
                const int il = std::stoi(n->name + prefix_len);
                ggml_backend_dev_t device_kv = model.dev_layer(il);
                if (device_gdn != device_kv) {
                    LLAMA_LOG_WARN("%s: layer %d is assigned to device %s but the fused Gated Delta Net tensor "
                            "is assigned to device %s (usually due to missing support)\n",
                            __func__, il, ggml_backend_dev_name(device_kv), ggml_backend_dev_name(device_gdn));
                    gdn_device_mismatch = true;
                    break;
                }
            }

            if (gdn_device_mismatch) {
                cparams.fused_gdn_ch = false;
                LLAMA_LOG_WARN("%s: fused Gated Delta Net (chunked) not supported, set to disabled\n", __func__);
            } else {
                LLAMA_LOG_INFO("%s: fused Gated Delta Net (chunked) enabled\n", __func__);
            }
        }

        cparams.auto_fgdn = false;
    }

    // reserve worst-case graph
    int n_splits_pp = -1;
    int n_nodes_pp  = -1;

    int n_splits_tg = -1;
    int n_nodes_tg  = -1;

    // reserve pp (prompt processing) graph first so that buffers are only allocated once
    {
        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get(),
                model.hparams.no_alloc, model.hparams.no_alloc ? backend_buf_exp_size.data() : nullptr);
        if (!gf) {
            if (cparams.pipeline_parallel) {
                LLAMA_LOG_WARN("%s: compute buffer allocation failed, retrying without pipeline parallelism\n", __func__);
                cparams.pipeline_parallel = false;
                sched.reset(ggml_backend_sched_new(backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(), max_nodes, false, cparams.op_offload));
                gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
            }
            if (!gf) {
                throw std::runtime_error("failed to allocate compute pp buffers");
            }
        }

        n_splits_pp = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_pp  = ggml_graph_n_nodes(gf);
    }

    // reserve with tg (token generation) graph to get the number of splits and nodes
    {
        auto * gf = graph_reserve(n_seqs, n_seqs, n_seqs, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute tg buffers");
        }

        n_splits_tg = ggml_backend_sched_get_n_splits(sched.get());
        n_nodes_tg  = ggml_graph_n_nodes(gf);
    }

    // reserve again with pp graph to avoid ggml-alloc reallocations during inference
    {
        // TODO: not sure if the following graph would be worst case for multi-stream KV caches:
        //
        // auto * gf = graph_reserve(n_tokens, 1, n_tokens, mctx.get());
        //
        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get(), model.hparams.no_alloc);
        if (!gf) {
            throw std::runtime_error("failed to allocate compute pp buffers");
        }
    }

    for (size_t i = 0; i < backend_ptrs.size(); ++i) {
        ggml_backend_t             backend = backend_ptrs[i];
        ggml_backend_buffer_type_t buft    = backend_buft[i];
        if (!model.hparams.no_alloc) {
            backend_buf_exp_size[i] = ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
        if (backend_buf_exp_size[i] > 1) {
            LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", __func__,
                    ggml_backend_buft_name(buft),
                    backend_buf_exp_size[i] / 1024.0 / 1024.0);
        }
    }

    if (n_nodes_pp == n_nodes_tg) {
        LLAMA_LOG_INFO("%s: graph nodes  = %d\n", __func__, n_nodes_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph nodes  = %d (with bs=%d), %d (with bs=1)\n", __func__, n_nodes_pp, n_tokens, n_nodes_tg);
    }

    if (n_splits_pp == n_splits_tg) {
        LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);
    } else {
        LLAMA_LOG_INFO("%s: graph splits = %d (with bs=%d), %d (with bs=1)\n", __func__, n_splits_pp, n_tokens, n_splits_tg);
    }

    const int64_t t_end_us = ggml_time_us();

    LLAMA_LOG_INFO("%s: reserve took %.2f ms, sched copies = %d\n",
            __func__, (t_end_us - t_start_us)/1000.0, ggml_backend_sched_get_n_copies(sched.get()));
}

void llama_context::synchronize() {
    if (!sched) {
        return;
    }

    ggml_backend_sched_synchronize(sched.get());

    // FIXME: if multiple single tokens are evaluated without a synchronization,
    // the stats will be added to the prompt evaluation stats
    // this should only happen when using batch size 1 to evaluate a batch

    // add the evaluation to the stats
    if (n_queued_tokens == 1) {
        if (!cparams.no_perf) {
            t_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_eval++;
    } else if (n_queued_tokens > 1) {
        if (!cparams.no_perf) {
            t_p_eval_us += ggml_time_us() - t_compute_start_us;
        }
        n_p_eval += n_queued_tokens;
    }

    // get a more accurate load time, upon first eval
    if (n_queued_tokens > 0 && !has_evaluated_once) {
        t_load_us = ggml_time_us() - t_start_us;
        has_evaluated_once = true;
    }

    n_queued_tokens = 0;
    t_compute_start_us = 0;
}

const llama_model & llama_context::get_model() const {
    return model;
}

const llama_cparams & llama_context::get_cparams() const {
    return cparams;
}

ggml_backend_sched_t llama_context::get_sched() const {
    return sched.get();
}

uint32_t llama_context::n_ctx() const {
    return cparams.n_ctx;
}

uint32_t llama_context::n_ctx_seq() const {
    return cparams.n_ctx_seq;
}

uint32_t llama_context::n_batch() const {
    return cparams.n_batch;
}

uint32_t llama_context::n_ubatch() const {
    return cparams.n_ubatch;
}

uint32_t llama_context::n_seq_max() const {
    return cparams.n_seq_max;
}

uint32_t llama_context::n_threads() const {
    return cparams.n_threads;
}

uint32_t llama_context::n_threads_batch() const {
    return cparams.n_threads_batch;
}

llama_memory_t llama_context::get_memory() const {
    return memory.get();
}

bool llama_context::memory_update(bool optimize) {
    if (!memory) {
        return false;
    }

    {
        const auto mctx = memory->init_update(this, optimize);
        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                    // noop
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    // no updates need to be performed
                    return false;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: failed to prepare memory update\n", __func__);
                    return false;
                }
        }

        // reset the previous graph result to make sure that it won't be reused
        // TODO: change the mctx->apply() to return information if a graph reserve is needed
        //       reset the graph result only if the memory module did reset the scheduler
        gf_res_prev->reset();

        if (!mctx->apply()) {
            LLAMA_LOG_ERROR("%s: failed to apply memory update\n", __func__);
        }
    }

    // if the memory module did any computation, we have to reserve a new worst-case graph
    {
        const auto mctx = memory->init_full();
        if (!mctx) {
            throw std::runtime_error("failed to initialize memory context");
        }

        const uint32_t n_seqs = cparams.n_seq_max;
        const uint32_t n_tokens = std::min(cparams.n_ctx, cparams.n_ubatch);

        auto * gf = graph_reserve(n_tokens, n_seqs, n_tokens, mctx.get());
        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to reserve graph after the memory update\n", __func__);
        }
    }

    return true;
}

enum llama_pooling_type llama_context::pooling_type() const {
    return cparams.pooling_type;
}

float * llama_context::get_logits() {
    output_reorder();

    return logits.data;
}

int64_t llama_context::output_resolve_row(int32_t i) const {
    int64_t j = -1;

    // support negative indices (last output row)
    if (i < 0) {
        j = n_outputs + i;
        if (j < 0) {
            throw std::runtime_error(format("negative index out of range [0, %d)", n_outputs));
        }
    } else if ((size_t) i >= output_ids.size()) {
        throw std::runtime_error(format("out of range [0, %zu)", output_ids.size()));
    } else {
        // use output_ids to translate the batch token index into a row number
        // that holds this token's data.
        j = output_ids[i];
    }

    if (j < 0) {
        // the batch token was not configured to output anything
        throw std::runtime_error(format("batch.logits[%d] != true", i));
    }

    if (j >= n_outputs) {
        throw std::runtime_error(format("corrupt output buffer (j=%" PRId64 ", n_outputs=%d)", j, n_outputs));
    }

    return j;
}

float * llama_context::get_logits_ith(int32_t i) {
    output_reorder();

    try {
        if (logits.data == nullptr) {
            throw std::runtime_error("no logits");
        }

        const int64_t j = output_resolve_row(i);
        return logits.data + j*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid logits id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings() {
    output_reorder();

    return embd.data;
}

llama_token * llama_context::get_sampled_tokens()  const{
    return sampling.sampled.data;
}

float * llama_context::get_embeddings_ith(int32_t i) {
    output_reorder();

    try {
        if (embd.data == nullptr) {
            throw std::runtime_error("no embeddings");
        }

        const int64_t j = output_resolve_row(i);
        const uint32_t n_embd_out = model.hparams.n_embd_out();
        return embd.data + j*n_embd_out;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid embeddings id %d, reason: %s\n", __func__, i, err.what());
#ifndef NDEBUG
        GGML_ABORT("fatal error");
#else
        return nullptr;
#endif
    }
}

float * llama_context::get_embeddings_seq(llama_seq_id seq_id) {
    auto it = embd_seq.find(seq_id);
    if (it == embd_seq.end()) {
        return nullptr;
    }

    return it->second.data();
}

llama_token llama_context::get_sampled_token_ith(int32_t idx) {
    output_reorder();

    if (!sampling.sampled.has_data()) {
        return LLAMA_TOKEN_NULL;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        GGML_ASSERT(row < (int64_t) sampling.sampled.size);
        return sampling.sampled.data[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled token id %d, reason: %s\n", __func__, idx, err.what());
        return LLAMA_TOKEN_NULL;
    }
}

float * llama_context::get_sampled_probs_ith(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size() || sampling.probs_count[row] == 0) {
            return nullptr;
        }
        return sampling.probs.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

float * llama_context::get_sampled_logits_ith(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return nullptr;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size() || sampling.logits_count[row] == 0) {
            return nullptr;
        }
        return sampling.logits.data + row*model.vocab.n_tokens();
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits id %d, reason: %s\n", __func__, idx, err.what());
        return nullptr;
    }
}

const llama_token * llama_context::get_sampled_candidates_ith(int32_t idx) {
    output_reorder();

    try {
        const int64_t row = output_resolve_row(idx);
        if (sampling.candidates.has_data() &&
            (size_t) row < sampling.candidates_count.size() &&
            sampling.candidates_count[row] > 0) {
            return sampling.candidates.data + row*model.vocab.n_tokens();
        }
    } catch (const std::exception & err) {
        // fallback to full vocab list
        GGML_UNUSED(err);
    }

    return sampling.token_ids_full_vocab.data();
}

size_t llama_context::get_sampled_candidates_count(int32_t idx) {
    output_reorder();

    if (!sampling.candidates.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.candidates_count.size()) {
            return 0;
        }
        return sampling.candidates_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled candidates count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_logits_count(int32_t idx) {
    output_reorder();

    if (!sampling.logits.has_data()) {
        return model.vocab.n_tokens();
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.logits_count.size()) {
            return 0;
        }
        return sampling.logits_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled logits count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}

size_t llama_context::get_sampled_probs_count(int32_t idx) {
    output_reorder();

    if (!sampling.probs.has_data()) {
        return 0;
    }

    try {
        const int64_t row = output_resolve_row(idx);
        if ((size_t) row >= sampling.probs_count.size()) {
            return 0;
        }
        return sampling.probs_count[row];
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: invalid backend sampled probs count id %d, reason: %s\n", __func__, idx, err.what());
        return 0;
    }
}


// DFlash speculative-decoding API

void llama_context::set_dflash_capture(const int32_t * layer_ids,
                                       size_t          n_layer_ids,
                                       int64_t         n_embd_target) {
    dflash.capture_layer_ids.clear();
    dflash.captured_features.clear();
    dflash.captured_n_outputs   = 0;
    dflash.capture_n_embd       = n_embd_target;
    dflash.last_packed_captures = nullptr;

    if (layer_ids == nullptr || n_layer_ids == 0) {
        return;
    }
    dflash.capture_layer_ids.assign(layer_ids, layer_ids + n_layer_ids);

    // Changing capture_layer_ids changes the graph topology — invalidate cache.
    if (gf_res_prev) {
        gf_res_prev->reset();
    }
}

const float * llama_context::get_dflash_captured_features(int64_t * n_outputs_out) const {
    if (n_outputs_out) {
        *n_outputs_out = dflash.captured_n_outputs;
    }
    if (dflash.captured_features.empty()) {
        return nullptr;
    }
    return dflash.captured_features.data();
}

int32_t llama_context::dflash_force_host_readback() {
    if (dflash.last_packed_captures == nullptr) {
        return -1;
    }
    if (dflash.capture_n_embd <= 0 || dflash.capture_layer_ids.empty()) {
        return -1;
    }
    const int64_t n_outputs        = dflash.captured_n_outputs;
    if (n_outputs <= 0) {
        return -1;
    }
    const int64_t n_layers         = (int64_t) dflash.capture_layer_ids.size();
    const int64_t n_embd_per_layer = dflash.capture_n_embd;
    const int64_t n_features       = n_layers * n_embd_per_layer;

    dflash.captured_features.assign((size_t) n_outputs * (size_t) n_features, 0.0f);

    ggml_backend_t backend_c = ggml_backend_sched_get_tensor_backend(
        sched.get(), dflash.last_packed_captures);
    ggml_backend_tensor_get_async(backend_c, dflash.last_packed_captures,
        dflash.captured_features.data(), 0,
        (size_t) n_outputs * (size_t) n_features * sizeof(float));
    ggml_backend_sched_synchronize(sched.get());
    return 0;
}

const int32_t * llama_context::get_dflash_draft_topk(int64_t * n_outputs_out, uint32_t * topk_out) const {
    if (n_outputs_out) {
        *n_outputs_out = dflash.draft_topk_n_outputs;
    }
    if (topk_out) {
        *topk_out = dflash.draft_topk_K;
    }
    if (dflash.draft_topk.empty()) {
        return nullptr;
    }
    return dflash.draft_topk.data();
}

void llama_context::dflash_finalize_draft_topk() {
    const uint32_t K = dflash.draft_topk_K;
    if (K < 2) {
        return;
    }
    if (dflash.draft_topk.empty() || dflash.draft_topk_argmax.empty()) {
        return;
    }

    const int64_t n = dflash.draft_topk_n_outputs;
    GGML_ASSERT((size_t) n * K == dflash.draft_topk.size());
    GGML_ASSERT((size_t) n     == dflash.draft_topk_argmax.size());

    for (int64_t i = 0; i < n; ++i) {
        int32_t * slots  = dflash.draft_topk.data() + (size_t) i * K;
        const int32_t am = dflash.draft_topk_argmax[i];

        for (uint32_t k = 0; k < K; ++k) {
            if (slots[k] == am) {
                if (k != 0) {
                    std::swap(slots[0], slots[k]);
                }
                goto next_pos;
            }
        }
    next_pos: ;
    }
}

const llama_dflash * llama_context::get_dflash() const {
    return &dflash;
}

void llama_context::dflash_reset_ctx_kv() {
    dflash.ctx_filled   = 0;
    dflash.ctx_pos_base = 0;
    if (gf_res_prev) {
        gf_res_prev->reset();
    }
}

bool llama_context::dflash_slide_left(int64_t n_drop) {
    GGML_ASSERT(n_drop > 0);
    GGML_ASSERT(n_drop <= dflash.ctx_filled);

    const int64_t n_keep = dflash.ctx_filled - n_drop;

    if (n_keep > 0) {
        std::vector<uint8_t> scratch;

        for (size_t il = 0; il < dflash.ctx_K.size(); ++il) {
            ggml_tensor * K = dflash.ctx_K[il];
            ggml_tensor * V = dflash.ctx_V[il];
            GGML_ASSERT(K != nullptr && V != nullptr);

            const size_t bytes_K = (size_t) n_keep * K->nb[1];
            const size_t bytes_V = (size_t) n_keep * V->nb[1];
            const size_t off_K   = (size_t) n_drop * K->nb[1];
            const size_t off_V   = (size_t) n_drop * V->nb[1];

            if (scratch.size() < bytes_K) scratch.resize(bytes_K);
            ggml_backend_tensor_get(K, scratch.data(), off_K, bytes_K);
            ggml_backend_tensor_set(K, scratch.data(), 0,    bytes_K);

            if (scratch.size() < bytes_V) scratch.resize(bytes_V);
            ggml_backend_tensor_get(V, scratch.data(), off_V, bytes_V);
            ggml_backend_tensor_set(V, scratch.data(), 0,    bytes_V);
        }
    }

    dflash.ctx_filled    = n_keep;
    dflash.ctx_pos_base += n_drop;

    return true;
}

int32_t llama_context::dflash_extend(const float * target_hidden_new,
                                     int64_t       n_new,
                                     int64_t       pos_start) {
    if (model.arch != LLM_ARCH_DFLASH) {
        LLAMA_LOG_ERROR("%s: model is not LLM_ARCH_DFLASH (arch=%d)\n",
                __func__, (int) model.arch);
        return -1;
    }
    if (n_new <= 0) {
        return 0;
    }
    if (target_hidden_new == nullptr) {
        LLAMA_LOG_ERROR("%s: target_hidden_new is null\n", __func__);
        return -1;
    }
    if (dflash.ctx_K.empty() || dflash.ctx_V.empty()) {
        LLAMA_LOG_ERROR("%s: DFlash K/V side store not allocated\n", __func__);
        return -1;
    }
    if (n_new > dflash.ctx_capacity) {
        LLAMA_LOG_ERROR("%s: n_new=%lld exceeds side store capacity=%lld\n",
                __func__, (long long) n_new, (long long) dflash.ctx_capacity);
        return -1;
    }
    if (dflash.ctx_filled + n_new > dflash.ctx_capacity) {
        const int64_t n_drop = dflash.ctx_filled + n_new - dflash.ctx_capacity;
        if (!dflash_slide_left(n_drop)) {
            return -1;
        }
    }
    if (dflash.n_features <= 0) {
        LLAMA_LOG_ERROR("%s: dflash.n_features not initialised\n", __func__);
        return -1;
    }

    const int64_t write_offset = dflash.ctx_filled;

    // Slot lookup: per-shape cache for n_new in [1, dflash_block_size]
    // (= the chain-mode steady-state range), plus a single fallback slot
    // for n_new > dflash_block_size (used by full-prompt prefill, which
    // is one-shot per prompt and doesn't benefit from caching).
    bool                  cache_hit = false;
    ggml_backend_sched_t  sched_slot = nullptr;
    llm_graph_result    * res        = nullptr;
    if ((size_t) n_new < sched_dflash_encode_slots.size()
            && sched_dflash_encode_slots[(size_t) n_new]) {
        sched_slot = sched_dflash_encode_slots[(size_t) n_new].get();
        if (!gf_res_dflash_encode_slots[(size_t) n_new]) {
            gf_res_dflash_encode_slots[(size_t) n_new].reset(
                new llm_graph_result(graph_max_nodes(cparams.n_ubatch)));
        }
        res       = gf_res_dflash_encode_slots[(size_t) n_new].get();
        cache_hit = (res->get_gf() != nullptr) && (ggml_graph_n_nodes(res->get_gf()) > 0);
    } else {
        // Fallback path: lazy-init the single shared slot used for any
        // n_new outside [1, block_size]. Same single-slot semantics as
        // before the multi-slot cache was added.
        if (!sched_dflash_encode_fallback) {
            sched_dflash_encode_fallback.reset(ggml_backend_sched_new(
                backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                graph_max_nodes(cparams.n_ubatch), /*parallel=*/ false, cparams.op_offload));
        }
        if (!gf_res_dflash_encode_fallback) {
            gf_res_dflash_encode_fallback.reset(
                new llm_graph_result(graph_max_nodes(cparams.n_ubatch)));
        }
        sched_slot = sched_dflash_encode_fallback.get();
        res        = gf_res_dflash_encode_fallback.get();
        cache_hit  = (gf_res_dflash_encode_fallback_n_new == n_new);
    }

    ggml_cgraph * gf = nullptr;

    if (cache_hit) {
        gf = res->get_gf();
        GGML_ASSERT(gf != nullptr && ggml_graph_n_nodes(gf) > 0);
    } else {
        res->reset();

        llama_ubatch ub_stub = {};
        ub_stub.n_tokens     = (uint32_t) n_new;
        ub_stub.n_seq_tokens = (uint32_t) n_new;
        ub_stub.n_seqs       = 1;
        ub_stub.n_seqs_unq   = 1;

        auto gparams = graph_params(res, ub_stub, /*mctx=*/nullptr,
                                    LLM_GRAPH_TYPE_DEFAULT);
        // Encoder graph runs on the per-size slot (or fallback) selected
        // above; rebind cb so the per-layer norm pin lands on the actual
        // scheduler this graph will execute on.
        gparams.cb = graph_get_cb(sched_slot);

        ggml_backend_sched_reset(sched_slot);
        ggml_backend_sched_set_eval_callback(sched_slot,
            cparams.cb_eval, cparams.cb_eval_user_data);

        // Build the encoder graph directly (bypasses model.build_graph dispatch).
        llama_model_dflash::encode_graph encode(model, gparams);
        gf = res->get_gf();

        if (!gf || ggml_graph_n_nodes(gf) == 0) {
            LLAMA_LOG_ERROR("%s: encoder graph build produced no nodes\n", __func__);
            return -1;
        }

        if (!ggml_backend_sched_alloc_graph(sched_slot, gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate encoder graph\n", __func__);
            return -2;
        }

        // Track the cache key for the fallback (single-slot) path. Per-shape
        // slots don't need a key — once their gf is built they always hit.
        if (sched_slot == sched_dflash_encode_fallback.get()) {
            gf_res_dflash_encode_fallback_n_new = n_new;
        }
    }

    // ----- set inputs on the allocated graph -----
    ggml_tensor * t_target_hidden_new =
        ggml_get_tensor(res->get_ctx(), "dflash_enc_target_hidden_new");
    ggml_tensor * t_pos_new =
        ggml_get_tensor(res->get_ctx(), "dflash_enc_pos_new");
    ggml_tensor * t_pos_idx =
        ggml_get_tensor(res->get_ctx(), "dflash_enc_pos_idx");
    GGML_ASSERT(t_target_hidden_new != nullptr);
    GGML_ASSERT(t_pos_new           != nullptr);
    GGML_ASSERT(t_pos_idx           != nullptr);

    {
        const size_t bytes = (size_t) n_new * (size_t) dflash.n_features * sizeof(float);
        GGML_ASSERT((size_t) ggml_nbytes(t_target_hidden_new) >= bytes);
        ggml_backend_tensor_set(t_target_hidden_new, target_hidden_new, 0, bytes);
    }

    {
        std::vector<int32_t> pos_buf((size_t) n_new);
        for (int64_t i = 0; i < n_new; ++i) {
            pos_buf[(size_t) i] = (int32_t) (pos_start + i);
        }
        ggml_backend_tensor_set(t_pos_new, pos_buf.data(),
                                0, n_new * sizeof(int32_t));
    }

    {
        std::vector<int64_t> idx_buf((size_t) n_new);
        for (int64_t i = 0; i < n_new; ++i) {
            idx_buf[(size_t) i] = write_offset + i;
        }
        ggml_backend_tensor_set(t_pos_idx, idx_buf.data(),
                                0, n_new * sizeof(int64_t));
    }

    {
        const auto status = graph_compute(sched_slot, gf, /*batched=*/n_new > 1);
        if (status != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: encoder graph compute failed (%d)\n",
                    __func__, (int) status);
            return -3;
        }
    }

    dflash.ctx_filled += n_new;
    dflash.n_ctx = dflash.ctx_filled;

    if (gf_res_prev) {
        gf_res_prev->reset();
    }

    return 0;
}

int32_t llama_context::dflash_extend_from_tensor(ggml_tensor * src_captures,
                                                 int64_t       src_row_offset,
                                                 int64_t       n_keep,
                                                 int64_t       pos_start) {
    const int64_t n_keep_real = n_keep;
    int64_t       n_keep_pad  = n_keep;

    if (model.arch != LLM_ARCH_DFLASH) {
        LLAMA_LOG_ERROR("%s: model is not LLM_ARCH_DFLASH (arch=%d)\n",
                __func__, (int) model.arch);
        return -1;
    }
    if (n_keep <= 0) {
        return 0;
    }
    if (src_captures == nullptr) {
        LLAMA_LOG_ERROR("%s: src_captures is null\n", __func__);
        return -1;
    }
    if (src_row_offset < 0 || src_row_offset + n_keep > src_captures->ne[1]) {
        LLAMA_LOG_ERROR("%s: src_row_offset=%lld n_keep=%lld out of bounds for src ne[1]=%lld\n",
                __func__, (long long) src_row_offset, (long long) n_keep,
                (long long) src_captures->ne[1]);
        return -1;
    }
    if (dflash.ctx_K.empty() || dflash.ctx_V.empty()) {
        LLAMA_LOG_ERROR("%s: DFlash K/V side store not allocated\n", __func__);
        return -1;
    }
    // Padding writes up to (n_keep_pad - n_keep_real) extra rows into slots
    // above ctx_filled. Treat the padded width as the capacity-consuming size
    // so we slide_left enough to keep all writes (real + padded) in-bounds.
    if (n_keep_pad > dflash.ctx_capacity) {
        n_keep_pad = n_keep_real; // pathological capacity; fall back to unpadded
    }
    if (n_keep_real > dflash.ctx_capacity) {
        LLAMA_LOG_ERROR("%s: n_keep=%lld exceeds side store capacity=%lld\n",
                __func__, (long long) n_keep_real, (long long) dflash.ctx_capacity);
        return -1;
    }
    if (dflash.ctx_filled + n_keep_pad > dflash.ctx_capacity) {
        const int64_t n_drop = dflash.ctx_filled + n_keep_pad - dflash.ctx_capacity;
        if (!dflash_slide_left(n_drop)) {
            return -1;
        }
    }
    if (dflash.n_features <= 0) {
        LLAMA_LOG_ERROR("%s: dflash.n_features not initialised\n", __func__);
        return -1;
    }
    if (src_captures->ne[0] != dflash.n_features) {
        LLAMA_LOG_ERROR("%s: src_captures ne[0]=%lld != dflash.n_features=%lld\n",
                __func__, (long long) src_captures->ne[0], (long long) dflash.n_features);
        return -1;
    }

    const int64_t write_offset = dflash.ctx_filled;

    // Slot lookup: per-shape cache for n_keep_pad in [1, dflash_block_size]
    // (chain-mode steady state), single fallback slot for larger widths
    // (full-prompt prefill, one-shot per prompt). Same shape as the
    // dflash_extend() variant above; both share the slot vectors.
    bool                  cache_hit  = false;
    ggml_backend_sched_t  sched_slot = nullptr;
    llm_graph_result    * res        = nullptr;
    if ((size_t) n_keep_pad < sched_dflash_encode_slots.size()
            && sched_dflash_encode_slots[(size_t) n_keep_pad]) {
        sched_slot = sched_dflash_encode_slots[(size_t) n_keep_pad].get();
        if (!gf_res_dflash_encode_slots[(size_t) n_keep_pad]) {
            gf_res_dflash_encode_slots[(size_t) n_keep_pad].reset(
                new llm_graph_result(graph_max_nodes(cparams.n_ubatch)));
        }
        res       = gf_res_dflash_encode_slots[(size_t) n_keep_pad].get();
        cache_hit = (res->get_gf() != nullptr) && (ggml_graph_n_nodes(res->get_gf()) > 0);
    } else {
        if (!sched_dflash_encode_fallback) {
            sched_dflash_encode_fallback.reset(ggml_backend_sched_new(
                backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
                graph_max_nodes(cparams.n_ubatch), /*parallel=*/ false, cparams.op_offload));
        }
        if (!gf_res_dflash_encode_fallback) {
            gf_res_dflash_encode_fallback.reset(
                new llm_graph_result(graph_max_nodes(cparams.n_ubatch)));
        }
        sched_slot = sched_dflash_encode_fallback.get();
        res        = gf_res_dflash_encode_fallback.get();
        cache_hit  = (gf_res_dflash_encode_fallback_n_new == n_keep_pad);
    }

    ggml_cgraph * gf = nullptr;

    if (cache_hit) {
        gf = res->get_gf();
        GGML_ASSERT(gf != nullptr && ggml_graph_n_nodes(gf) > 0);
    } else {
        res->reset();

        llama_ubatch ub_stub = {};
        ub_stub.n_tokens     = (uint32_t) n_keep_pad;
        ub_stub.n_seq_tokens = (uint32_t) n_keep_pad;
        ub_stub.n_seqs       = 1;
        ub_stub.n_seqs_unq   = 1;

        auto gparams = graph_params(res, ub_stub, /*mctx=*/nullptr,
                                    LLM_GRAPH_TYPE_DEFAULT);
        // Encoder graph runs on the per-size slot (or fallback) selected
        // above; rebind cb so the per-layer norm pin lands on the actual
        // scheduler this graph will execute on.
        gparams.cb = graph_get_cb(sched_slot);

        ggml_backend_sched_reset(sched_slot);
        ggml_backend_sched_set_eval_callback(sched_slot,
            cparams.cb_eval, cparams.cb_eval_user_data);

        llama_model_dflash::encode_graph encode(model, gparams);
        gf = res->get_gf();

        if (!gf || ggml_graph_n_nodes(gf) == 0) {
            LLAMA_LOG_ERROR("%s: encoder graph build produced no nodes\n", __func__);
            return -1;
        }

        if (!ggml_backend_sched_alloc_graph(sched_slot, gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate encoder graph\n", __func__);
            return -2;
        }

        if (sched_slot == sched_dflash_encode_fallback.get()) {
            gf_res_dflash_encode_fallback_n_new = n_keep_pad;
        }
    }

    ggml_tensor * t_target_hidden_new =
        ggml_get_tensor(res->get_ctx(), "dflash_enc_target_hidden_new");
    ggml_tensor * t_pos_new =
        ggml_get_tensor(res->get_ctx(), "dflash_enc_pos_new");
    ggml_tensor * t_pos_idx =
        ggml_get_tensor(res->get_ctx(), "dflash_enc_pos_idx");
    GGML_ASSERT(t_target_hidden_new != nullptr);
    GGML_ASSERT(t_pos_new           != nullptr);
    GGML_ASSERT(t_pos_idx           != nullptr);

    // ----- D2D copy from src_captures slice into t_target_hidden_new -----
    //
    // Build a transient view of the requested slice of src_captures. The
    // view inherits view_src->buffer in ggml_backend_tensor_copy, so the
    // copy path picks src's buffer correctly regardless of whether src
    // lives in the same or a different scheduler than dst.
    //
    // Only the first n_keep_real rows carry valid features. Padded rows
    // (n_keep_real..n_keep_pad) are left with whatever t_target_hidden_new
    // already contains from a previous compute — they are processed by the
    // encoder but their outputs are scattered to slots above the post-call
    // ctx_filled boundary (via pos_idx below) and are never read.
    {
        struct ggml_init_params vp = {
            /*mem_size=*/   ggml_tensor_overhead() * 3,
            /*mem_buffer=*/ nullptr,
            /*no_alloc=*/   true,
        };
        struct ggml_context * view_ctx = ggml_init(vp);
        GGML_ASSERT(view_ctx != nullptr);

        ggml_tensor * src_view = ggml_view_2d(view_ctx, src_captures,
            /*ne0=*/    src_captures->ne[0],
            /*ne1=*/    n_keep_real,
            /*nb1=*/    src_captures->nb[1],
            /*offset=*/ (size_t) src_row_offset * src_captures->nb[1]);
        GGML_ASSERT(src_view != nullptr);

        // ggml_backend_tensor_copy() reads src->buffer directly (does NOT
        // follow view_src->buffer), so manually inherit the buffer pointer
        // here. Without this the copy crashes on the buffer-is-host check.
        src_view->buffer = src_captures->buffer;

        // When padding (n_keep_pad > n_keep_real), t_target_hidden_new is
        // wider than the slice we're copying in. The backend copy asserts
        // identical layouts, so build a matching sub-view of the dst that
        // covers only the first n_keep_real columns.
        ggml_tensor * dst_view = t_target_hidden_new;
        if (n_keep_pad != n_keep_real) {
            dst_view = ggml_view_2d(view_ctx, t_target_hidden_new,
                /*ne0=*/    t_target_hidden_new->ne[0],
                /*ne1=*/    n_keep_real,
                /*nb1=*/    t_target_hidden_new->nb[1],
                /*offset=*/ 0);
            GGML_ASSERT(dst_view != nullptr);
            dst_view->buffer = t_target_hidden_new->buffer;
        }

        const size_t bytes = (size_t) n_keep_real * (size_t) dflash.n_features * sizeof(float);
        GGML_ASSERT((size_t) ggml_nbytes(dst_view) >= bytes);
        GGML_ASSERT((size_t) ggml_nbytes(src_view) >= bytes);

        // Per ggml semantics this is a true D2D within a single device
        // backend (e.g. Vulkan/CUDA cpy_tensor); only falls back to a host
        // bounce if buffers are on different physical devices. Cross-ctx
        // ordering is fine here because llama_decode() synchronizes its
        // scheduler before returning, so src_captures's contents are
        // settled by the time we enter this function.
        ggml_backend_tensor_copy(src_view, dst_view);

        ggml_free(view_ctx);
    }

    {
        std::vector<int32_t> pos_buf((size_t) n_keep_pad);
        for (int64_t i = 0; i < n_keep_real; ++i) {
            pos_buf[(size_t) i] = (int32_t) (pos_start + i);
        }
        // Padded rows: any continuation; positions don't affect correctness
        // because their outputs land in slots above ctx_filled.
        for (int64_t i = n_keep_real; i < n_keep_pad; ++i) {
            pos_buf[(size_t) i] = (int32_t) (pos_start + i);
        }
        ggml_backend_tensor_set(t_pos_new, pos_buf.data(),
                                0, n_keep_pad * sizeof(int32_t));
    }

    {
        std::vector<int64_t> idx_buf((size_t) n_keep_pad);
        for (int64_t i = 0; i < n_keep_real; ++i) {
            idx_buf[(size_t) i] = write_offset + i;
        }
        // Padded rows: scatter to slots above the post-call ctx_filled
        // boundary. These slots get overwritten by the next extend's real
        // rows before the draft can read them.
        for (int64_t i = n_keep_real; i < n_keep_pad; ++i) {
            idx_buf[(size_t) i] = write_offset + i;
        }
        ggml_backend_tensor_set(t_pos_idx, idx_buf.data(),
                                0, n_keep_pad * sizeof(int64_t));
    }

    {
        const auto status = graph_compute(sched_slot, gf, /*batched=*/n_keep_pad > 1);
        if (status != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: encoder graph compute failed (%d)\n",
                    __func__, (int) status);
            return -3;
        }
    }

    dflash.ctx_filled += n_keep_real;
    dflash.n_ctx = dflash.ctx_filled;

    if (gf_res_prev) {
        gf_res_prev->reset();
    }

    return 0;
}

// inline encoder on target context's scheduler
//
// Same encoder graph contents as dflash_extend_from_tensor, executed on
// the TARGET context's sched_dflash_inline_encode instead of the DRAFT's
// sched_dflash_encode_slots[]. Called as
//   target_ctx->dflash_inline_encode_from_ctx(draft_ctx, ...)
// with src_captures being target_ctx->dflash.last_packed_captures (i.e.
// a tensor in target's own buffer — the captures read is now local, no
// cross-context D2D). The graph still references the draft's encoder
// weights (dflash_fc, dflash_hidden_norm, per-layer wk/wv/k_norm) and
// the draft's side-store ctx_K/ctx_V tensors; ggml-backend dispatches
// each op onto whichever ggml_backend_t owns the buffer of the tensor
// it touches. This requires the standard llama.cpp init where both
// contexts share singleton CUDA backends per device.
int32_t llama_context::dflash_inline_encode_from_ctx(llama_context * draft_ctx,
                                                     ggml_tensor *   src_captures,
                                                     int64_t         src_row_offset,
                                                     int64_t         n_keep,
                                                     int64_t         pos_start) {
    if (draft_ctx == nullptr) {
        LLAMA_LOG_ERROR("%s: draft_ctx is null\n", __func__);
        return -1;
    }
    if (draft_ctx->model.arch != LLM_ARCH_DFLASH) {
        LLAMA_LOG_ERROR("%s: draft_ctx is not LLM_ARCH_DFLASH (arch=%d)\n",
                __func__, (int) draft_ctx->model.arch);
        return -1;
    }
    if (n_keep <= 0) {
        return 0;
    }
    if (src_captures == nullptr) {
        LLAMA_LOG_ERROR("%s: src_captures is null\n", __func__);
        return -1;
    }
    if (src_row_offset < 0 || src_row_offset + n_keep > src_captures->ne[1]) {
        LLAMA_LOG_ERROR("%s: src_row_offset=%lld n_keep=%lld out of bounds for src ne[1]=%lld\n",
                __func__, (long long) src_row_offset, (long long) n_keep,
                (long long) src_captures->ne[1]);
        return -1;
    }

    // All the side-store sizing lives on the draft context.
    llama_dflash & ddflash = draft_ctx->dflash;
    if (ddflash.ctx_K.empty() || ddflash.ctx_V.empty()) {
        LLAMA_LOG_ERROR("%s: DFlash K/V side store not allocated on draft\n", __func__);
        return -1;
    }
    if (n_keep > ddflash.ctx_capacity) {
        LLAMA_LOG_ERROR("%s: n_keep=%lld exceeds side store capacity=%lld\n",
                __func__, (long long) n_keep, (long long) ddflash.ctx_capacity);
        return -1;
    }
    if (ddflash.ctx_filled + n_keep > ddflash.ctx_capacity) {
        const int64_t n_drop = ddflash.ctx_filled + n_keep - ddflash.ctx_capacity;
        if (!draft_ctx->dflash_slide_left(n_drop)) {
            return -1;
        }
    }
    if (ddflash.n_features <= 0) {
        LLAMA_LOG_ERROR("%s: ddflash.n_features not initialised\n", __func__);
        return -1;
    }
    if (src_captures->ne[0] != ddflash.n_features) {
        LLAMA_LOG_ERROR("%s: src_captures ne[0]=%lld != ddflash.n_features=%lld\n",
                __func__, (long long) src_captures->ne[0], (long long) ddflash.n_features);
        return -1;
    }

    const int64_t write_offset = ddflash.ctx_filled;

    // Lazy-init target-side encoder scheduler + result holder using THIS
    // context's backend pointers (target's). Graph build still references
    // draft's tensors; the sched dispatches ops to the right backend per
    // tensor.
    if (!gf_res_dflash_inline_encode) {
        gf_res_dflash_inline_encode.reset(
            new llm_graph_result(graph_max_nodes(draft_ctx->cparams.n_ubatch)));
    }
    if (!sched_dflash_inline_encode) {
        sched_dflash_inline_encode.reset(ggml_backend_sched_new(
            backend_ptrs.data(), backend_buft.data(), backend_ptrs.size(),
            graph_max_nodes(draft_ctx->cparams.n_ubatch),
            /*parallel=*/ false, cparams.op_offload));
    }
    auto * res = gf_res_dflash_inline_encode.get();

    const bool cache_hit = (gf_res_dflash_inline_encode_n_new == n_keep);
    ggml_cgraph * gf = nullptr;

    if (cache_hit) {
        gf = res->get_gf();
        GGML_ASSERT(gf != nullptr && ggml_graph_n_nodes(gf) > 0);
    } else {
        res->reset();

        llama_ubatch ub_stub = {};
        ub_stub.n_tokens     = (uint32_t) n_keep;
        ub_stub.n_seq_tokens = (uint32_t) n_keep;
        ub_stub.n_seqs       = 1;
        ub_stub.n_seqs_unq   = 1;

        // Hand-roll llama_graph_params with draft's arch/hparams/cparams/
        // dflash, target's sched + backend_cpu. The encoder's encode_graph
        // constructor uses arch/hparams/cparams for op shapes, dflash for
        // ctx_K/V destinations, and references its model arg directly for
        // weights.
        llm_graph_params gparams = {
            /*.arch        =*/ draft_ctx->model.arch,
            /*.hparams     =*/ draft_ctx->model.hparams,
            /*.cparams     =*/ draft_ctx->cparams,
            /*.ubatch      =*/ ub_stub,
            /*.gtype       =*/ LLM_GRAPH_TYPE_DEFAULT,
            /*.sched       =*/ sched_dflash_inline_encode.get(),
            /*.backend_cpu =*/ backend_cpu,
            /*.cvec        =*/ draft_ctx->cvec.get(),
            /*.loras       =*/ draft_ctx->loras.get(),
            /*.mctx        =*/ nullptr,
            /*.cross       =*/ &draft_ctx->cross,
            /*.dflash      =*/ &ddflash,
            /*.tree_mask   =*/ nullptr,
            /*.samplers    =*/ draft_ctx->sampling.samplers,
            /*.n_outputs   =*/ (uint32_t) n_keep,
            /*.cb          =*/ draft_ctx->graph_get_cb(sched_dflash_inline_encode.get()),
            /*.res         =*/ res,
        };

        ggml_backend_sched_reset(sched_dflash_inline_encode.get());
        ggml_backend_sched_set_eval_callback(sched_dflash_inline_encode.get(),
            cparams.cb_eval, cparams.cb_eval_user_data);

        llama_model_dflash::encode_graph encode(draft_ctx->model, gparams);
        gf = res->get_gf();

        if (!gf || ggml_graph_n_nodes(gf) == 0) {
            LLAMA_LOG_ERROR("%s: encoder graph build produced no nodes\n", __func__);
            return -1;
        }

        if (!ggml_backend_sched_alloc_graph(sched_dflash_inline_encode.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate encoder graph on target sched\n",
                            __func__);
            return -2;
        }

        gf_res_dflash_inline_encode_n_new = n_keep;
    }

    ggml_tensor * t_target_hidden_new =
        ggml_get_tensor(res->get_ctx(), "dflash_enc_target_hidden_new");
    ggml_tensor * t_pos_new =
        ggml_get_tensor(res->get_ctx(), "dflash_enc_pos_new");
    ggml_tensor * t_pos_idx =
        ggml_get_tensor(res->get_ctx(), "dflash_enc_pos_idx");
    GGML_ASSERT(t_target_hidden_new != nullptr);
    GGML_ASSERT(t_pos_new           != nullptr);
    GGML_ASSERT(t_pos_idx           != nullptr);

    // src_captures lives in TARGET's buffer (this context); t_target_hidden_new
    // is allocated by THIS sched in target's compute buffer. Same context, so
    // the copy is intra-context same-device. No cross-context D2D involved.
    {
        struct ggml_init_params vp = {
            /*mem_size=*/   ggml_tensor_overhead() * 2,
            /*mem_buffer=*/ nullptr,
            /*no_alloc=*/   true,
        };
        struct ggml_context * view_ctx = ggml_init(vp);
        GGML_ASSERT(view_ctx != nullptr);

        ggml_tensor * src_view = ggml_view_2d(view_ctx, src_captures,
            /*ne0=*/    src_captures->ne[0],
            /*ne1=*/    n_keep,
            /*nb1=*/    src_captures->nb[1],
            /*offset=*/ (size_t) src_row_offset * src_captures->nb[1]);
        GGML_ASSERT(src_view != nullptr);
        src_view->buffer = src_captures->buffer;

        const size_t bytes = (size_t) n_keep * (size_t) ddflash.n_features * sizeof(float);
        GGML_ASSERT((size_t) ggml_nbytes(t_target_hidden_new) >= bytes);
        GGML_ASSERT((size_t) ggml_nbytes(src_view)            >= bytes);

        ggml_backend_tensor_copy(src_view, t_target_hidden_new);
        ggml_free(view_ctx);
    }

    {
        std::vector<int32_t> pos_buf((size_t) n_keep);
        for (int64_t i = 0; i < n_keep; ++i) {
            pos_buf[(size_t) i] = (int32_t) (pos_start + i);
        }
        ggml_backend_tensor_set(t_pos_new, pos_buf.data(),
                                0, n_keep * sizeof(int32_t));
    }

    {
        std::vector<int64_t> idx_buf((size_t) n_keep);
        for (int64_t i = 0; i < n_keep; ++i) {
            idx_buf[(size_t) i] = write_offset + i;
        }
        ggml_backend_tensor_set(t_pos_idx, idx_buf.data(),
                                0, n_keep * sizeof(int64_t));
    }

    {
        const auto status = graph_compute(sched_dflash_inline_encode.get(), gf,
                                          /*batched=*/n_keep > 1);
        if (status != GGML_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: encoder graph compute failed (%d)\n",
                    __func__, (int) status);
            return -3;
        }
    }

    // Update draft's side-store bookkeeping. ctx_filled / n_ctx live on
    // draft because the side-store tensors do.
    ddflash.ctx_filled += n_keep;
    ddflash.n_ctx = ddflash.ctx_filled;

    return 0;
}

void llama_context::set_tree_mask(const uint8_t * visibility, int n_tree_tokens) {
    GGML_ASSERT(visibility != nullptr);
    GGML_ASSERT(n_tree_tokens > 0);

    // No gf_res_prev->reset() here. The graph-reuse machinery
    // (llm_graph_result::can_reuse + llm_graph_params::allow_reuse) handles
    // tree-mask transitions correctly:
    //   - Same-shape verify-after-verify: gparams.tree_mask pointer is
    //     stable (== &this->tree_mask), ubatch dims match, and the
    //     attn-kv input class's set_input re-applies the visibility
    //     matrix on every decode — no rebuild needed.
    //   - chain → tree or tree → chain: gparams.tree_mask flips
    //     between nullptr and &tree_mask, allow_reuse's pointer
    //     comparison forces a rebuild on its own.
    //   - tree-N → tree-M (different n_tree_tokens): forces a rebuild
    //     via the n_tokens check in allow_reuse / can_reuse_kq_mask.
    //
    // Callers wanting eager invalidation can call gf_res_prev->reset()
    // directly.
    tree_mask.active        = true;
    tree_mask.n_tree_tokens = n_tree_tokens;
    const size_t n2         = (size_t) n_tree_tokens * (size_t) n_tree_tokens;
    tree_mask.visibility.assign(visibility, visibility + n2);

    // tree-mode: propagate the tree-active state to the underlying KV cache
    // so apply_ubatch's contiguity-invariant purge is bypassed for the next
    // verify decode. The keep_positions_range
    // post-accept call restores monotonicity.
    if (memory) {
        memory->set_tree_mode_active(true);
    }
}

void llama_context::clear_tree_mask() {
    if (!tree_mask.active && tree_mask.visibility.empty()) {
        return;
    }

    // same rationale as set_tree_mask: no eager gf_res_prev->reset().
    // After clear_tree_mask, gparams.tree_mask becomes nullptr;
    // allow_reuse's pointer comparison on the next
    // decode rebuilds the graph if the next decode runs in chain
    // mode, and short-circuits to the previous tree-mode cache if the
    // next call is another set_tree_mask with the same shape.
    tree_mask.active        = false;
    tree_mask.n_tree_tokens = 0;
    tree_mask.visibility.clear();

    if (memory) {
        memory->set_tree_mode_active(false);
    }
}

void llama_context::attach_threadpool(
           ggml_threadpool_t threadpool,
           ggml_threadpool_t threadpool_batch) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = threadpool;
    this->threadpool_batch = threadpool_batch ? threadpool_batch : threadpool;
}

void llama_context::detach_threadpool() {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->threadpool       = nullptr;
    this->threadpool_batch = nullptr;
}

void llama_context::set_n_threads(int32_t n_threads, int32_t n_threads_batch) {
    LLAMA_LOG_DEBUG("%s: n_threads = %d, n_threads_batch = %d\n", __func__, n_threads, n_threads_batch);

    cparams.n_threads       = n_threads;
    cparams.n_threads_batch = n_threads_batch;
}

void llama_context::set_abort_callback(bool (*abort_callback)(void * data), void * abort_callback_data) {
    LLAMA_LOG_DEBUG("%s: call\n", __func__);

    this->abort_callback      = abort_callback;
    this->abort_callback_data = abort_callback_data;

    for (auto & backend : backends) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend.get()));
        if (reg) {
            auto * set_abort_callback_fn = (ggml_backend_set_abort_callback_t) ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_abort_callback");
            if (set_abort_callback_fn) {
                set_abort_callback_fn(backend.get(), this->abort_callback, this->abort_callback_data);
            }
        }
    }
}

void llama_context::set_embeddings(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    cparams.embeddings = value;

    // TODO: not sure yet if we want to reserve here
    //sched_need_reserve = true;
}

void llama_context::set_causal_attn(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.causal_attn == value) {
        return;
    }

    cparams.causal_attn = value;

    sched_need_reserve = true;
}

void llama_context::set_warmup(bool value) {
    LLAMA_LOG_DEBUG("%s: value = %d\n", __func__, value);

    if (cparams.warmup == value) {
        return;
    }

    cparams.warmup = value;

    // warmups are usually with small batches, so no need to reserve
    //sched_need_reserve = true;
}

bool llama_context::set_sampler(llama_seq_id seq_id, llama_sampler * sampler) {
    if (!sampler && sampling.samplers.count(seq_id) == 0) {
        return true;
    }

    LLAMA_LOG_DEBUG("%s: seq_id = %d, sampler = %p\n", __func__, (int) seq_id, (void *) sampler);

    const bool can_offload =
        sampler &&
        sampler->iface->backend_init &&
        sampler->iface->backend_apply &&
        llama_sampler_chain_n(sampler) > 0;

    if (sampler && can_offload) {
        auto * buft = ggml_backend_dev_buffer_type(model.dev_output());

        sampler->iface->backend_init(sampler, buft);

        sampling.samplers[seq_id] = sampler;

        sched_need_reserve = true;

        return true;
    }

    if (sampler && !can_offload) {
        LLAMA_LOG_WARN("%s: sampler '%s' for seq_id = %d, cannot be offloaded to the backend\n", __func__, llama_sampler_name(sampler), seq_id);

        if (sampling.samplers.count(seq_id) > 0) {
            sched_need_reserve = true;
        }

        sampling.samplers.erase(seq_id);

        return false;
    }

    sampling.samplers.erase(seq_id);

    sched_need_reserve = true;

    return true;
}

void llama_context::set_adapters_lora(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    if (adapters_lora_are_same(adapters, n_adapters, scales)) {
        return;
    }

    loras.reset(new llama_adapter_loras());

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] != 0.0f) {
            loras->insert({adapters[i], scales[i]});
        }
    }

    sched_need_reserve = true;
}

bool llama_context::adapters_lora_are_same(llama_adapter_lora ** adapters, size_t n_adapters, float * scales) {
    LLAMA_LOG_DEBUG("%s: adapters = %p\n", __func__, (void *) adapters);

    // Adapters with a zero scale are never added to `loras`, so also ignore them for the comparison.
    size_t n_non_zero = 0;

    for (size_t i = 0; i < n_adapters; i ++) {
        if (scales[i] == 0.0f) {
            continue;
        }
        n_non_zero++;

        auto it = loras->find(adapters[i]);

        if (it == loras->end() || it->second != scales[i]) {
            return false;
        }
    }

    if (n_non_zero != loras->size()) {
        return false;
    }

    return true;
}

bool llama_context::set_adapter_cvec(
            const float * data,
                 size_t   len,
                int32_t   n_embd,
                int32_t   il_start,
                int32_t   il_end) {
    LLAMA_LOG_DEBUG("%s: il_start = %d, il_end = %d\n", __func__, il_start, il_end);

    bool res = cvec->apply(model, data, len, n_embd, il_start, il_end);

    sched_need_reserve = true;

    return res;
}

llm_graph_result * llama_context::process_ubatch(const llama_ubatch & ubatch, llm_graph_type gtype, llama_memory_context_i * mctx, ggml_status & ret) {
    if (mctx && !mctx->apply()) {
        LLAMA_LOG_ERROR("%s: failed to apply memory context\n", __func__);
        ret = GGML_STATUS_FAILED;
        return nullptr;
    }

    auto * res = gf_res_prev.get();
    auto * gf  = res->get_gf();

    // the new graph parameters
    // in order to correctly reuse a graph, it's full topology has to be uniquely determined by these parameters
    const auto gparams = graph_params(res, ubatch, mctx, gtype);

    if (!graph_reuse_disable && res->can_reuse(gparams)) {
        //LLAMA_LOG_DEBUG("%s: reusing previous graph\n", __func__);

        // with pipeline parallelism, the previous graph_compute_async may still be running
        // on the GPU. we must synchronize before set_inputs to avoid overwriting input tensors
        // that the previous compute is still reading.
        if (cparams.pipeline_parallel) {
            ggml_backend_sched_synchronize(sched.get());
        }

        n_reused++;
    } else {
        res->reset();

        ggml_backend_sched_reset(sched.get());
        ggml_backend_sched_set_eval_callback(sched.get(), cparams.cb_eval, cparams.cb_eval_user_data);

        //const auto t_start_us = ggml_time_us();

        gf = model.build_graph(gparams);

        //LLAMA_LOG_INFO("graph build time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);

        if (!gf) {
            LLAMA_LOG_ERROR("%s: failed to initialize graph\n", __func__);
            ret = GGML_STATUS_FAILED;
            return nullptr;
        }

        if (!ggml_backend_sched_alloc_graph(sched.get(), gf)) {
            LLAMA_LOG_ERROR("%s: failed to allocate graph\n", __func__);
            ret = GGML_STATUS_ALLOC_FAILED;
            return nullptr;
        }
    }

    // set the input data for the input tensors
    {
        //const auto t_start_us = ggml_time_us();

        // FIXME this call causes a crash if any model inputs were not used in the graph and were therefore not allocated
        res->set_inputs(&ubatch);

        //LLAMA_LOG_INFO("graph set inputs time: %.3f ms\n", (ggml_time_us() - t_start_us)/1000.0);
    }

    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: failed to compute graph, compute status: %d\n", __func__, status);
        ret = status;
        return nullptr;
    }

    ret = GGML_STATUS_SUCCESS;

    return res;
}

int llama_context::encode(const llama_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & hparams = model.hparams;

    const int64_t n_embd  = hparams.n_embd_inp();
    const int64_t n_vocab = model.vocab.n_tokens();

    // note: during encode, we always pass the full sequence starting from pos = 0
    if (!balloc->init(batch_inp, model.vocab, nullptr, n_embd, cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens = balloc->get_n_tokens();

    // [TAG_NO_CACHE_PAD]
    // TODO: add new split mode where we pad the input sequences so that ubatch.equal_seqs == true
    const llama_ubatch ubatch = balloc->split_simple(n_tokens);

    // micro-batching is not possible for non-causal encoding, so we process the batch in a single shot
    GGML_ASSERT(cparams.n_ubatch >= n_tokens && "encoder requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();

    sched_reserve();

    n_queued_tokens += n_tokens;

    // reserve output buffer
    if (output_reserve(n_tokens) < n_tokens) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %u outputs\n", __func__, n_tokens);
        return -2;
    };

    for (uint32_t i = 0; i < n_tokens; ++i) {
        output_ids[i] = i;
    }

    n_outputs = n_tokens;

    const auto causal_attn_org = cparams.causal_attn;

    // always use non-causal attention for encoder graphs
    // TODO: this is a tmp solution until we have a proper way to support enc-dec models
    //       ref: https://github.com/ggml-org/llama.cpp/pull/12181#issuecomment-2730451223
    cparams.causal_attn = false;

    ggml_status status;
    const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_ENCODER, nullptr, status);

    cparams.causal_attn = causal_attn_org;

    if (!res) {
        switch (status) {
            case GGML_STATUS_ABORTED:      return  2;
            case GGML_STATUS_ALLOC_FAILED: return -2;
            case GGML_STATUS_FAILED:       return -3;
            case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
        }
    }

    auto * t_logits = res->get_logits();
    auto * t_embd = res->get_embd_pooled() ? res->get_embd_pooled() : res->get_embd();

    // extract logits
    if (logits.data && t_logits) {
        ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
        GGML_ASSERT(backend_res != nullptr);
        GGML_ASSERT(logits.data != nullptr);

        ggml_backend_tensor_get_async(backend_res, t_logits, logits.data, 0, n_tokens*n_vocab*sizeof(float));
    }

    // extract embeddings
    if (embd.data && t_embd) {
        ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
        GGML_ASSERT(backend_embd != nullptr);

        switch (cparams.pooling_type) {
            case LLAMA_POOLING_TYPE_NONE:
                {
                    // extract token embeddings
                    GGML_ASSERT(embd.data != nullptr);
                    const uint32_t n_embd_out = hparams.n_embd_out();

                    GGML_ASSERT(n_tokens*n_embd_out <= (int64_t) embd.size);
                    ggml_backend_tensor_get_async(backend_embd, t_embd, embd.data, 0, n_tokens*n_embd_out*sizeof(float));
                } break;
            case LLAMA_POOLING_TYPE_MEAN:
            case LLAMA_POOLING_TYPE_CLS:
            case LLAMA_POOLING_TYPE_LAST:
                {
                    // extract sequence embeddings
                    auto & embd_seq_out = embd_seq;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        embd_seq_out[seq_id].resize(n_embd_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_RANK:
                {
                    // extract the rerank score - n_cls_out floats per sequence
                    auto & embd_seq_out = embd_seq;

                    const uint32_t n_cls_out = hparams.n_cls_out;

                    for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                        const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                        const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                        embd_seq_out[seq_id].resize(n_cls_out);
                        ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                    }
                } break;
            case LLAMA_POOLING_TYPE_UNSPECIFIED:
                {
                    GGML_ABORT("unknown pooling type");
                }
        }
    }

    // TODO: hacky solution
    if (model.arch == LLM_ARCH_T5 && t_embd) {
        //cross.t_embd = t_embd;

        synchronize();

        cross.n_embd = t_embd->ne[0];
        cross.n_enc  = t_embd->ne[1];
        cross.v_embd.resize(cross.n_embd*cross.n_enc);
        memcpy(cross.v_embd.data(), embd.data, ggml_nbytes(t_embd));

        const auto & batch = balloc->get_batch();

        // remember the sequence ids used during the encoding - needed for cross attention later
        cross.seq_ids_enc.resize(n_tokens);
        for (uint32_t i = 0; i < n_tokens; i++) {
            cross.seq_ids_enc[i].clear();

            for (int s = 0; s < batch.n_seq_id[i]; s++) {
                const llama_seq_id seq_id = batch.seq_id[i][s];

                cross.seq_ids_enc[i].insert(seq_id);
            }
        }
    }

    return 0;
}

static std::map<llama_seq_id, uint32_t> build_seq_to_output_row(const llama_ubatch & ubatch, uint32_t row_offset) {
    std::map<llama_seq_id, uint32_t> seq_to_row;
    // how many output tokens we have seen so far for this ubatch.
    uint32_t local = 0;
    for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
        // skip tokens that are not output.
        if (!ubatch.output[i]) {
            continue;
        }

        const llama_seq_id seq_id = ubatch.seq_id[i][0];
        // row_offset is the number of output tokens before this ubatch.
        seq_to_row[seq_id] = row_offset + local;
        ++local;
    }
    return seq_to_row;
}

static void copy_tensor_async_ints(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & sampled,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!sampled.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < sampled.size);

        GGML_ASSERT(ggml_is_contiguous(tensor) && "sampled tokens tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        ggml_backend_tensor_get_async(backend, tensor, sampled.data + row, 0, sizeof(sampled.data[row]));
    }
}

static void copy_tensor_async_floats(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<float> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "logits/probs tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        float * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of logits/probabilities that were written for this row.
        counts[row] = ggml_nelements(tensor);
    }
}

static void copy_tensor_async_candidates(
    const std::map<llama_seq_id, ggml_tensor*> & tensor_map,
    const buffer_view<llama_token> & dst,
    size_t stride,
    std::vector<uint32_t> & counts,
    const std::map<llama_seq_id, uint32_t> & seq_to_row,
    ggml_backend_sched_t sched) {
    if (!dst.has_data()) {
        return;
    }

    for (const auto & [seq_id, tensor] : tensor_map) {
        auto it = seq_to_row.find(seq_id);
        if (it == seq_to_row.end()) {
            continue;
        }

        const uint32_t row = it->second;
        GGML_ASSERT(row < counts.size());

        GGML_ASSERT(ggml_is_contiguous(tensor) && "candidates tensor must be contiguous for async copy");

        ggml_backend_t backend = ggml_backend_sched_get_tensor_backend(sched, tensor);
        llama_token * row_ptr = dst.data + (size_t) row * stride;
        ggml_backend_tensor_get_async(backend, tensor, row_ptr, 0, ggml_nbytes(tensor));

        // Update the actual number of candidates that were written.
        counts[row] = ggml_nelements(tensor);
    }
}

static bool needs_raw_logits(const llama_ubatch & ubatch, const std::map<llama_seq_id, llama_sampler *> & samplers) {
    for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
        if (!ubatch.output[i]) {
            continue;
        }

        // Check if the output token has at least one sequence without a backend sampler.
        for (int32_t j = 0; j < ubatch.n_seq_id[i]; ++j) {
            llama_seq_id seq_id = ubatch.seq_id[i][j];
            if (samplers.find(seq_id) == samplers.end()) {
                return true;
            }
        }
    }
    return false; // all sequences use backend sampling
}

int llama_context::decode(const llama_batch & batch_inp) {
    GGML_ASSERT((!batch_inp.token && batch_inp.embd) || (batch_inp.token && !batch_inp.embd)); // NOLINT

    if (!memory) {
        LLAMA_LOG_DEBUG("%s: cannot decode batches with this context (calling encode() instead)\n", __func__);
        return encode(batch_inp);
    }

    if (batch_inp.n_tokens == 0) {
        LLAMA_LOG_ERROR("%s: n_tokens == 0\n", __func__);
        return -1;
    }

    const auto & vocab   = model.vocab;
    const auto & hparams = model.hparams;

    const int64_t n_vocab = vocab.n_tokens();
    const int64_t n_embd  = hparams.n_embd_inp();

    // when computing embeddings, all tokens are output
    const bool output_all   = cparams.embeddings;
    const bool has_samplers = !sampling.samplers.empty();

    const uint32_t n_seq_max = cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max;

    // TODO: avoid this workaround in the future
    if (has_samplers && batch_inp.logits) {
        std::vector<int32_t> seq_output_count(n_seq_max, 0);

        for (int32_t i = 0; i < batch_inp.n_tokens; ++i) {
            if (batch_inp.logits[i] == 0) {
                continue;
            }

            const int ns = batch_inp.n_seq_id ? batch_inp.n_seq_id[i] : 1;

            for (int32_t s = 0; s < ns; ++s) {
                const llama_seq_id seq_id = batch_inp.seq_id ? batch_inp.seq_id[i][s] : 0;

                seq_output_count[seq_id]++;
                if (seq_output_count[seq_id] > 1) {
                    LLAMA_LOG_ERROR("%s: backend sampling requires at most one output token per sequence (seq_id %d had %d)\n",
                            __func__, seq_id, seq_output_count[seq_id]);
                    return -1;
                }
            }
        }
    }

    if (!balloc->init(batch_inp, vocab, memory.get(), n_embd, n_seq_max, output_all)) {
        LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
        return -1;
    }

    const uint32_t n_tokens_all  = balloc->get_n_tokens();
    const uint32_t n_outputs_all = balloc->get_n_outputs();

    if (output_all) {
        // require that all tokens are output
        if (n_outputs_all != n_tokens_all) {
            LLAMA_LOG_ERROR("%s: pooled embedding requires that all tokens are output (n_outputs_all = %d, n_tokens_all = %d)\n",
                    __func__, n_outputs_all, n_tokens_all);
            return -1;
        }
    }

    GGML_ASSERT(n_tokens_all <= cparams.n_batch);

    GGML_ASSERT((cparams.causal_attn || cparams.n_ubatch >= n_tokens_all) && "non-causal attention requires n_ubatch >= n_tokens");

    if (t_compute_start_us == 0) {
        t_compute_start_us = ggml_time_us();
    }
    n_queued_tokens += n_tokens_all;

    // TODO: this clear of the buffer can easily be forgotten - need something better
    embd_seq.clear();
    output_swaps.clear();

    sched_reserve();

    bool did_optimize = false;

    // handle any pending shifts/copies
    memory_update(false);

    llama_memory_context_ptr mctx;

    while (true) {
        mctx = memory->init_batch(*balloc, cparams.n_ubatch, output_all);
        if (!mctx) {
            return -2;
        }

        switch (mctx->get_status()) {
            case LLAMA_MEMORY_STATUS_SUCCESS:
                {
                } break;
            case LLAMA_MEMORY_STATUS_NO_UPDATE:
                {
                    LLAMA_LOG_ERROR("%s: unexpected memory context status: %d\n", __func__, mctx->get_status());

                    return -2;
                }
            case LLAMA_MEMORY_STATUS_FAILED_PREPARE:
                {
                    if (!did_optimize) {
                        did_optimize = true;

                        if (memory_update(true)) {
                            LLAMA_LOG_DEBUG("%s: retrying batch size %d after cache optimization\n", __func__, balloc->get_n_tokens());

                            continue;
                        }
                    }

                    LLAMA_LOG_WARN("%s: failed to find a memory slot for batch of size %d\n", __func__, balloc->get_n_tokens());

                    return 1;
                }
            case LLAMA_MEMORY_STATUS_FAILED_COMPUTE:
                {
                    LLAMA_LOG_ERROR("%s: compute failed while preparing batch of size %d\n", __func__, balloc->get_n_tokens());

                    return -2;
                }
        }

        break;
    }

    // reserve output buffer
    if (output_reserve(n_outputs_all) < n_outputs_all) {
        LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
        return -2;
    };

    int64_t n_outputs_prev = 0;

    do {
        const auto & ubatch = mctx->get_ubatch();

        // count the outputs in this ubatch
        {
            int32_t n_outputs_new = 0;

            if (n_outputs_all == n_tokens_all) {
                n_outputs_new = ubatch.n_tokens;
            } else {
                for (uint32_t i = 0; i < ubatch.n_tokens; i++) {
                    n_outputs_new += (int32_t) (ubatch.output[i] != 0);
                }
            }

            // needs to happen before the graph is built
            n_outputs = n_outputs_new;
        }

        ggml_status status;
        const auto * res = process_ubatch(ubatch, LLM_GRAPH_TYPE_DECODER, mctx.get(), status);

        if (!res) {
            // the last ubatch failed or was aborted -> remove all positions of that ubatch from the memory module
            llama_pos pos_min[LLAMA_MAX_SEQ];
            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                pos_min[s] = std::numeric_limits<llama_pos>::max();
            }

            for (uint32_t i = 0; i < ubatch.n_tokens; ++i) {
                const auto & seq_id = ubatch.seq_id[i][0];

                pos_min[seq_id] = std::min(pos_min[seq_id], ubatch.pos[i]);
            }

            for (int s = 0; s < LLAMA_MAX_SEQ; ++s) {
                if (pos_min[s] == std::numeric_limits<llama_pos>::max()) {
                    continue;
                }

                LLAMA_LOG_WARN("%s: removing memory module entries for seq_id = %d, pos = [%d, +inf)\n", __func__, s, pos_min[s]);

                memory->seq_rm(s, pos_min[s], -1);
            }

            switch (status) {
                case GGML_STATUS_ABORTED:      return  2;
                case GGML_STATUS_ALLOC_FAILED: return -2;
                case GGML_STATUS_FAILED:       return -3;
                case GGML_STATUS_SUCCESS:      GGML_ABORT("should not happen");
            }
        }

        // plot the computation graph in dot format (for debugging purposes)
        //if (n_past%100 == 0) {
        //    ggml_graph_dump_dot(gf, NULL, "llama.dot");
        //}

        auto * t_logits = res->get_logits();
        auto * t_embd   = cparams.embeddings ? res->get_embd() : nullptr;

        if (t_embd && res->get_embd_pooled()) {
            t_embd = res->get_embd_pooled();
        }

        // extract logits
        if (logits.data && t_logits && n_outputs > 0 && needs_raw_logits(ubatch, sampling.samplers)) {
            ggml_backend_t backend_res = ggml_backend_sched_get_tensor_backend(sched.get(), t_logits);
            GGML_ASSERT(backend_res != nullptr);
            GGML_ASSERT(logits.data != nullptr);

            float * logits_out = logits.data + n_outputs_prev*n_vocab;

            if (n_outputs) {
                GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                GGML_ASSERT((n_outputs_prev + n_outputs)*n_vocab <= (int64_t) logits.size);
                ggml_backend_tensor_get_async(backend_res, t_logits, logits_out, 0, n_outputs*n_vocab*sizeof(float));
            }
        }

        // DFlash: read back in-graph top-K candidate token IDs (replaces the
        // bs * n_vocab * 4 byte float-logits transfer per draft block).
        if (model.arch == LLM_ARCH_DFLASH && res->t_dflash_topk && n_outputs > 0) {
            const uint32_t K = cparams.dflash_topk == 0 ? 1 : cparams.dflash_topk;
            // Pre-size at the start of the (first) ubatch.
            if (n_outputs_prev == 0) {
                dflash.draft_topk.assign((size_t) n_outputs_all * K, 0);
                dflash.draft_topk_n_outputs = n_outputs_all;
                dflash.draft_topk_K         = K;
                if (K >= 2) {
                    dflash.draft_topk_argmax.assign((size_t) n_outputs_all, 0);
                } else {
                    dflash.draft_topk_argmax.clear();
                }
            }
            ggml_backend_t backend_t = ggml_backend_sched_get_tensor_backend(sched.get(), res->t_dflash_topk);
            int32_t * dst = dflash.draft_topk.data() + (size_t) n_outputs_prev * K;
            ggml_backend_tensor_get_async(backend_t, res->t_dflash_topk, dst, 0,
                                          n_outputs * K * sizeof(int32_t));
            if (K >= 2 && res->t_dflash_topk_argmax) {
                ggml_backend_t backend_a = ggml_backend_sched_get_tensor_backend(sched.get(), res->t_dflash_topk_argmax);
                int32_t * dst_a = dflash.draft_topk_argmax.data() + n_outputs_prev;
                ggml_backend_tensor_get_async(backend_a, res->t_dflash_topk_argmax, dst_a, 0,
                                              n_outputs * sizeof(int32_t));
            }
        }

        // DFlash target capture: read back per-layer hidden states.
        //
        // fast path: when build_dflash_capture finalized a packed tensor
        // (shape [n_layers*n_embd_per_layer, n_outputs] in the same
        // row-per-token, all-layers-side-by-side layout the encoder graph
        // consumes), do a single D2H straight into dflash.captured_features
        // — no per-layer staging, no host transpose.
        //
        // Slow path: fall back to per-layer staging + host transpose when
        // the packed tensor isn't available (older graph result, or only a
        // subset of captures arrived).
        if (!res->t_dflash_captures.empty() && n_outputs > 0 && dflash.capture_n_embd > 0) {
            const int n_layers = (int) res->t_dflash_captures.size();
            const int64_t n_embd_per_layer = dflash.capture_n_embd;
            const int64_t n_features       = (int64_t) n_layers * n_embd_per_layer;

            if (res->t_dflash_captures_packed != nullptr) {
                // multi-ubatch decode: t_dflash_captures_packed only carries
                // this ubatch's rows (each ubatch ran a fresh graph), so the
                // device fast path can't reach the earlier ubatches' captures.
                // Force host accumulation (overriding skip_host_readback) and
                // null the device pointer so consumers route through the host
                // buffer instead of reading a partial tensor.
                const bool multi_ubatch_decode = (n_outputs_all > n_outputs) || (n_outputs_prev > 0);

                dflash.last_packed_captures = multi_ubatch_decode
                    ? nullptr
                    : res->t_dflash_captures_packed;

                const bool populate_host_buf = !dflash.skip_host_readback || multi_ubatch_decode;

                if (populate_host_buf) {
                    if (n_outputs_prev == 0) {
                        dflash.captured_features.assign(
                            (size_t) n_outputs_all * (size_t) n_features, 0.0f);
                        dflash.captured_n_outputs = n_outputs_all;
                    }
                    ggml_backend_t backend_c = ggml_backend_sched_get_tensor_backend(
                        sched.get(), res->t_dflash_captures_packed);
                    float * dst = dflash.captured_features.data()
                                + (size_t) n_outputs_prev * (size_t) n_features;
                    ggml_backend_tensor_get_async(backend_c, res->t_dflash_captures_packed,
                        dst, 0, (size_t) n_outputs * (size_t) n_features * sizeof(float));
                } else {
                    // Still record n_outputs so consumers can ask "how many
                    // captures are pending?" without inspecting tensors.
                    if (n_outputs_prev == 0) {
                        dflash.captured_features.clear();
                        dflash.captured_n_outputs = n_outputs_all;
                    }
                }
            } else {
                // Fallback path: build_dflash_capture didn't finalize a
                // packed tensor this graph. This shouldn't normally happen
                // — all expected captures arrive during a full model loop
                // — and the host-side per-layer staging + transpose path
                // is significantly slower. Log once if hit so we can spot
                // a graph-construction regression in the field.
                static bool warned_fallback = false;
                if (!warned_fallback) {
                    LLAMA_LOG_WARN("dflash capture fallback path hit (no packed tensor); "
                                   "consider this a perf regression. n_captures=%d expected=%zu\n",
                                   n_layers, dflash.capture_layer_ids.size());
                    warned_fallback = true;
                }
                // Per-layer staging: [n_layers] each sized n_outputs_all * n_embd_per_layer.
                if (n_outputs_prev == 0) {
                    dflash.capture_staging.assign(n_layers, std::vector<float>{});
                    for (int il = 0; il < n_layers; ++il) {
                        dflash.capture_staging[il].assign(
                            (size_t) n_outputs_all * (size_t) n_embd_per_layer, 0.0f);
                    }
                    dflash.captured_n_outputs = n_outputs_all;
                }
                for (int il = 0; il < n_layers; ++il) {
                    ggml_tensor * t = res->t_dflash_captures[il];
                    if (!t) continue;
                    ggml_backend_t backend_c = ggml_backend_sched_get_tensor_backend(sched.get(), t);
                    float * dst = dflash.capture_staging[il].data() + (size_t) n_outputs_prev * (size_t) n_embd_per_layer;
                    ggml_backend_tensor_get_async(backend_c, t, dst, 0,
                                                  (size_t) n_outputs * (size_t) n_embd_per_layer * sizeof(float));
                }
            }
        }

        // extract embeddings
        if (embd.data && t_embd && n_outputs > 0) {
            ggml_backend_t backend_embd = ggml_backend_sched_get_tensor_backend(sched.get(), t_embd);
            GGML_ASSERT(backend_embd != nullptr);

            switch (cparams.pooling_type) {
                case LLAMA_POOLING_TYPE_NONE:
                    {
                        // extract token embeddings
                        GGML_ASSERT(embd.data != nullptr);
                        const uint32_t n_embd_out = hparams.n_embd_out();
                        float * embd_out = embd.data + n_outputs_prev*n_embd_out;

                        if (n_outputs) {
                            GGML_ASSERT( n_outputs_prev + n_outputs <= n_outputs_all);
                            GGML_ASSERT((n_outputs_prev + n_outputs)*n_embd_out <= (int64_t) embd.size);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_out, 0, n_outputs*n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_MEAN:
                case LLAMA_POOLING_TYPE_CLS:
                case LLAMA_POOLING_TYPE_LAST:
                    {
                        // extract sequence embeddings (cleared before processing each batch)
                        auto & embd_seq_out = embd_seq;

                        // use n_embd_out (not n_embd_inp) - the pooled embedding has the model's
                        // output dimension, which differs from input dimension for deepstack models (e.g. qwen3vl)
                        const uint32_t n_embd_out = hparams.n_embd_out();

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_embd_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_embd_out*seq_idx)*sizeof(float), n_embd_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_RANK:
                    {
                        // extract the rerank score - n_cls_out floats per sequence
                        auto & embd_seq_out = embd_seq;

                        const uint32_t n_cls_out = hparams.n_cls_out;

                        for (uint32_t s = 0; s < ubatch.n_seqs_unq; ++s) {
                            const llama_seq_id seq_id  = ubatch.seq_id_unq[s];
                            const int32_t      seq_idx = ubatch.seq_idx[seq_id];

                            embd_seq_out[seq_id].resize(n_cls_out);
                            ggml_backend_tensor_get_async(backend_embd, t_embd, embd_seq_out[seq_id].data(), (n_cls_out*seq_idx)*sizeof(float), n_cls_out*sizeof(float));
                        }
                    } break;
                case LLAMA_POOLING_TYPE_UNSPECIFIED:
                    {
                        GGML_ABORT("unknown pooling type");
                    }
            }
        }

        // Copy backend sampling output if this ubatch produced any sampling tensors.
        if (has_samplers && (!res->t_sampled.empty() || !res->t_sampled_probs.empty() || !res->t_sampled_logits.empty())) {
            const auto seq_to_output_row = build_seq_to_output_row(ubatch, n_outputs_prev);
            const auto stride = n_vocab;

            // async copy the sampling data from the backend to the host
            copy_tensor_async_ints(res->t_sampled, sampling.sampled, seq_to_output_row, sched.get());

            copy_tensor_async_floats    (res->t_sampled_logits, sampling.logits,     stride, sampling.logits_count,     seq_to_output_row, sched.get());
            copy_tensor_async_floats    (res->t_sampled_probs,  sampling.probs,      stride, sampling.probs_count,      seq_to_output_row, sched.get());
            copy_tensor_async_candidates(res->t_candidates,     sampling.candidates, stride, sampling.candidates_count, seq_to_output_row, sched.get());
        }

        n_outputs_prev += n_outputs;
    } while (mctx->next());

    // DFlash: drain the per-layer staging D2Hs queued in the loop above
    // and host-transpose them into the row-per-token, all-layers-side-by-side
    // `captured_features` layout the encoder graph in dflash_extend() expects.
    //
    // skipped when build_dflash_capture finalized a packed tensor: in that
    // case the D2H above already wrote straight into dflash.captured_features
    // in the right layout, no host transpose needed.
    // We still sync the scheduler so the D2H is observed before the consumer
    // reads captured_features.
    if (!dflash.capture_layer_ids.empty()
            && dflash.capture_staging.empty()
            && !dflash.captured_features.empty()
            && n_outputs_all > 0) {
        ggml_backend_sched_synchronize(sched.get());
    } else if (!dflash.capture_layer_ids.empty()
            && dflash.skip_host_readback
            && dflash.last_packed_captures != nullptr
            && n_outputs_all > 0) {
        // skip-host-readback mode: we didn't enqueue a D2H but we still need
        // to synchronize so the packed tensor's compute settles before a
        // downstream consumer (llama_dflash_extend_from_ctx) reads it via
        // a cross-context D2D copy. Without this sync, target compute can
        // race the copy and the draft side store gets stale features.
        ggml_backend_sched_synchronize(sched.get());
    } else if (!dflash.capture_layer_ids.empty()
            && !dflash.capture_staging.empty()
            && n_outputs_all > 0) {
        ggml_backend_sched_synchronize(sched.get());

        const int64_t n_embd_target = dflash.capture_n_embd;
        const int64_t n_layers_cap  = (int64_t) dflash.capture_layer_ids.size();
        const int64_t n_features    = n_layers_cap * n_embd_target;

        dflash.captured_features.assign((size_t) n_outputs_all * (size_t) n_features, 0.0f);

        for (int64_t li = 0; li < n_layers_cap; ++li) {
            if ((size_t) li >= dflash.capture_staging.size()) break;
            const auto & layer_buf = dflash.capture_staging[(size_t) li];
            for (int64_t i = 0; i < n_outputs_all; ++i) {
                std::memcpy(
                    dflash.captured_features.data() + (size_t) i * (size_t) n_features
                                                    + (size_t) li * (size_t) n_embd_target,
                    layer_buf.data()                + (size_t) i * (size_t) n_embd_target,
                    (size_t) n_embd_target * sizeof(float));
            }
        }
    }

    // set to total number of outputs in the batch, for use in llama_get_logits_ith
    n_outputs = n_outputs_all;

    // set output mappings
    if (n_outputs > 0) {
        bool sorted_output = true;

        auto & out_ids = balloc->get_out_ids();

        GGML_ASSERT(out_ids.size() == (size_t) n_outputs);

        for (int64_t i = 0; i < n_outputs; ++i) {
            int64_t out_id = out_ids[i];
            output_ids[out_id] = i;
            if (out_id != i) {
                sorted_output = false;
            }
        }

        // make the outputs have the same order they had in the user-provided batch
        // note: this is mostly relevant for recurrent models atm
        if (!sorted_output && n_outputs > 1) {
            GGML_ASSERT((size_t) n_outputs == out_ids.size());

            // TODO: is there something more efficient which also minimizes swaps?
            // selection sort, to minimize swaps (from https://en.wikipedia.org/wiki/Selection_sort)
            for (uint32_t i = 0; i < n_outputs - 1; ++i) {
                uint32_t j_min = i;
                for (uint32_t j = i + 1; j < n_outputs; ++j) {
                    if (out_ids[j] < out_ids[j_min]) {
                        j_min = j;
                    }
                }
                if (j_min == i) {
                    continue;
                }
                std::swap(out_ids[i], out_ids[j_min]);

                // remember the swaps and apply them lazily upon logits/embeddings access
                output_swaps.push_back({ i, j_min });
            }

            std::fill(output_ids.begin(), output_ids.end(), -1);

            for (uint32_t i = 0; i < n_outputs; ++i) {
                output_ids[out_ids[i]] = i;
            }
        }
    }

    // wait for the computation to finish (automatically done when obtaining the model output)
    //synchronize();

    return 0;
}

//
// output
//

uint32_t llama_context::output_reserve(int32_t n_outputs) {
    const auto & hparams = model.hparams;
    const auto & vocab   = model.vocab;

    const int64_t n_outputs_max = std::max<int64_t>(n_outputs, n_seq_max());

    const auto n_batch    = cparams.n_batch;
    const auto n_vocab    = vocab.n_tokens();
    const auto n_embd_out = hparams.n_embd_out();

    bool has_logits = true;
    bool has_embd   = cparams.embeddings;

    // TODO: hacky enc-dec support
    if (model.arch == LLM_ARCH_T5) {
        has_logits = true;
        has_embd   = true;
    }


    size_t backend_float_count = 0;
    size_t backend_token_count = 0;

    logits.size = has_logits ? n_vocab*n_outputs_max : 0;
    embd.size   = has_embd ? n_embd_out*n_outputs_max : 0;

    // Allocate backend sampling output buffers if there are backend samplers configured.
    const bool has_sampling = !sampling.samplers.empty();
    if (has_sampling) {
        backend_float_count = 2 * n_vocab * n_outputs_max;      // logits + probs
        backend_token_count = (1 + n_vocab) * n_outputs_max;    // sampled + candidates
    }

    if (output_ids.empty()) {
        // init, never resized afterwards
        output_ids.resize(n_batch);
    }

    const size_t prev_size = buf_output ? ggml_backend_buffer_get_size(buf_output.get()) : 0;
    const size_t new_size  =
        (logits.size + embd.size + backend_float_count) * sizeof(float) +
        (                          backend_token_count) * sizeof(llama_token);

    // alloc only when more than the current capacity is required
    // TODO: also consider shrinking the buffer
    if (!buf_output || prev_size < new_size) {
        if (buf_output) {
#ifndef NDEBUG
            // This doesn't happen often, but may be annoying in some cases (like the HellaSwag benchmark)
            LLAMA_LOG_DEBUG("%s: reallocating output buffer from size %.02f MiB to %.02f MiB\n", __func__, prev_size / 1024.0 / 1024.0, new_size / 1024.0 / 1024.0);
#endif
            synchronize();

            // TODO: not needed?
            buf_output = nullptr;
            logits.data = nullptr;
            embd.data = nullptr;
        }

        auto * buft = ggml_backend_cpu_buffer_type();
        // try to use the host buffer of the device where the output tensor is allocated for faster transfer to system memory
        auto * output_dev = model.dev_output();
        auto * output_dev_host_buft = output_dev ? ggml_backend_dev_host_buffer_type(output_dev) : nullptr;
        if (output_dev_host_buft) {
            buft = output_dev_host_buft;
        }
        buf_output.reset(ggml_backend_buft_alloc_buffer(buft, new_size));
        if (buf_output == nullptr) {
            LLAMA_LOG_ERROR("%s: failed to allocate output buffer of size %.2f MiB\n", __func__, new_size / (1024.0 * 1024.0));
            return 0;
        }
        ggml_backend_buffer_clear(buf_output.get(), 0);
    }

    float * output_base = (float *) ggml_backend_buffer_get_base(buf_output.get());

    size_t offset = 0;
    uint8_t * base = (uint8_t *) output_base;

    logits = has_logits ? buffer_view<float>{output_base, logits.size} : buffer_view<float>{nullptr, 0};
    offset += logits.size * sizeof(float);

    embd = has_embd ? buffer_view<float>{(float *) (base + offset), embd.size} : buffer_view<float>{nullptr, 0};
    offset += embd.size * sizeof(float);

    if (has_sampling) {
        sampling.logits = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.logits.size * sizeof(float);

        sampling.probs = {(float *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.probs.size * sizeof(float);

        sampling.sampled = {(llama_token *) (base + offset), (size_t)n_outputs_max};
        offset += sampling.sampled.size * sizeof(llama_token);

        sampling.candidates = {(llama_token *) (base + offset), (size_t)(n_vocab*n_outputs_max)};
        offset += sampling.candidates.size * sizeof(llama_token);

        // The count vectors keep track of the actual number of logits/probs/candidates
        // copied from the backend for each output row.

        sampling.logits_count.resize(n_outputs_max);
        sampling.probs_count.resize(n_outputs_max);
        sampling.candidates_count.resize(n_outputs_max);

        std::fill(sampling.logits_count.begin(),     sampling.logits_count.end(),     0);
        std::fill(sampling.probs_count.begin(),      sampling.probs_count.end(),      0);
        std::fill(sampling.candidates_count.begin(), sampling.candidates_count.end(), 0);

        std::fill_n(sampling.sampled.data, sampling.sampled.size, LLAMA_TOKEN_NULL);
    } else {
        sampling.logits     = {nullptr, 0};
        sampling.probs      = {nullptr, 0};
        sampling.sampled    = {nullptr, 0};
        sampling.candidates = {nullptr, 0};

        sampling.logits_count.clear();
        sampling.probs_count.clear();
        sampling.candidates_count.clear();
    }

    // set all ids as invalid (negative)
    std::fill(output_ids.begin(), output_ids.end(), -1);

    this->n_outputs = 0;

    return n_outputs_max;
}

void llama_context::output_reorder() {
    const uint64_t n_vocab = model.vocab.n_tokens();
    const uint64_t n_embd  = model.hparams.n_embd;

    for (size_t s = 0; s < output_swaps.size(); ++s) {
        const uint64_t i0 = output_swaps[s].i0;
        const uint64_t i1 = output_swaps[s].i1;

        if (logits.size > 0) {
            for (uint64_t k = 0; k < n_vocab; k++) {
                std::swap(logits.data[i0*n_vocab + k], logits.data[i1*n_vocab + k]);
            }
        }

        if (embd.size > 0) {
            for (uint64_t k = 0; k < n_embd; k++) {
                std::swap(embd.data[i0*n_embd + k], embd.data[i1*n_embd + k]);
            }
        }

        if (!sampling.samplers.empty()) {
            assert(sampling.logits.size > 0);
            assert(sampling.probs.size > 0);
            assert(sampling.candidates.size > 0);
            assert(sampling.sampled.size > 0);
            assert(sampling.logits_count.size() > 0);
            assert(sampling.probs_count.size() > 0);
            assert(sampling.candidates_count.size() > 0);

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.logits.data[i0*n_vocab + k], sampling.logits.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.probs.data[i0*n_vocab + k], sampling.probs.data[i1*n_vocab + k]);
            }

            for (uint64_t k = 0; k < n_vocab; ++k) {
                std::swap(sampling.candidates.data[i0*n_vocab + k], sampling.candidates.data[i1*n_vocab + k]);
            }

            std::swap(sampling.sampled.data[i0],     sampling.sampled.data[i1]);
            std::swap(sampling.logits_count[i0],     sampling.logits_count[i1]);
            std::swap(sampling.probs_count[i0],      sampling.probs_count[i1]);
            std::swap(sampling.candidates_count[i0], sampling.candidates_count[i1]);
        }
    }

    output_swaps.clear();
}

//
// graph
//

uint32_t llama_context::graph_max_nodes(uint32_t n_tokens) const {
    if (model.arch == LLM_ARCH_QWEN3NEXT || model.arch == LLM_ARCH_KIMI_LINEAR || model.arch == LLM_ARCH_QWEN35 || model.arch == LLM_ARCH_QWEN35MOE) {
        return std::max<uint32_t>(n_tokens * 40, 32u * model.n_tensors());
    }
    uint32_t res = std::max<uint32_t>(1024u, 8u*model.n_tensors());
    for (const auto & lora : model.loras) {
        res += lora->get_n_nodes();
    }
    return res;
}

llm_graph_result * llama_context::get_gf_res_reserve() const {
    return static_cast<llm_graph_result *>(gf_res_reserve.get());
}

ggml_cgraph * llama_context::graph_reserve(
        uint32_t n_tokens, uint32_t n_seqs, uint32_t n_outputs, const llama_memory_context_i * mctx, bool split_only, size_t * sizes) {
    LLAMA_LOG_DEBUG("%s: reserving a graph for ubatch with n_tokens = %4u, n_seqs = %2u, n_outputs = %4u\n", __func__, n_tokens, n_seqs, n_outputs);
    GGML_ASSERT(n_outputs >= 1);

    if (n_tokens % n_seqs != 0) {
        n_tokens = ((n_tokens + (n_seqs - 1)) / n_seqs) * n_seqs; // round to next multiple of n_seqs
        n_outputs = std::max(n_outputs, n_tokens);

        LLAMA_LOG_DEBUG("%s: making n_tokens a multiple of n_seqs - n_tokens = %u, n_seqs = %u, n_outputs = %u\n", __func__, n_tokens, n_seqs, n_outputs);
    }

    ggml_backend_sched_reset(sched.get());

    // when the scheduler is reset, we cannot reuse the old graph, so we reset the previous graph result to prevent that
    gf_res_prev->reset();

    // store the n_outputs as it is, and restore it afterwards
    // TODO: not sure if needed, might simplify in the future by removing this
    const auto save_n_outputs = this->n_outputs;

    this->n_outputs = n_outputs;

    llama_batch_allocr balloc(model.hparams.n_pos_per_embd());
    llama_ubatch ubatch = balloc.ubatch_reserve(n_tokens/n_seqs, n_seqs);

    // set one output token per sequence in order to activate all backend samplers
    std::vector<llama_seq_id> seq_ids(n_seqs);
    for (uint32_t i = 0; i < n_seqs; ++i) {
        seq_ids[i] = i;
        ubatch.n_seq_id[i] = 1;
        ubatch.seq_id[i] = &seq_ids[i];
        ubatch.output[i] = true;
    }

    auto * res = gf_res_reserve.get();

    const auto gparams = graph_params(res, ubatch, mctx, LLM_GRAPH_TYPE_DEFAULT);

    res->reset();

    auto * gf = model.build_graph(gparams);

    this->n_outputs = save_n_outputs;

    // initialize scheduler with the specified graph
    if (split_only) {
        if (sizes) {
            ggml_backend_sched_reserve_size(sched.get(), gf, sizes);
        } else {
            ggml_backend_sched_split_graph(sched.get(), gf);
        }
    } else if (!ggml_backend_sched_reserve(sched.get(), gf)) {
        GGML_ASSERT(!sizes);
        LLAMA_LOG_ERROR("%s: failed to allocate compute buffers\n", __func__);
        return nullptr;
    }

    return gf;
}

llm_graph_params llama_context::graph_params(
                        llm_graph_result * res,
                      const llama_ubatch & ubatch,
            const llama_memory_context_i * mctx,
                          llm_graph_type   gtype) const {
    return {
        /*.arch        =*/ model.arch,
        /*.hparams     =*/ model.hparams,
        /*.cparams     =*/ cparams,
        /*.ubatch      =*/ ubatch,
        /*.gtype       =*/ gtype,
        /*.sched       =*/ sched.get(),
        /*.backend_cpu =*/ backend_cpu,
        /*.cvec        =*/ cvec.get(),
        /*.loras       =*/ loras.get(),
        /*.mctx        =*/ mctx,
        /*.cross       =*/ &cross,
        /*.dflash      =*/ &dflash,
        /*.tree_mask   =*/ tree_mask.active ? &tree_mask : nullptr,
        /*.samplers    =*/ sampling.samplers,
        /*.n_outputs   =*/ n_outputs,
        /*.cb          =*/ graph_get_cb(sched.get()),
        /*.res         =*/ res,
    };
}

ggml_status llama_context::graph_compute(
            ggml_cgraph * gf,
                   bool   batched) {
    return graph_compute(sched.get(), gf, batched);
}

ggml_status llama_context::graph_compute(
            ggml_backend_sched_t sched_use,
            ggml_cgraph * gf,
                   bool   batched) {
    int n_threads        = batched ? cparams.n_threads_batch : cparams.n_threads;
    ggml_threadpool_t tp = batched ? threadpool_batch        : threadpool;

    if (backend_cpu != nullptr) {
        auto * reg = ggml_backend_dev_backend_reg(ggml_backend_get_device(backend_cpu));
        auto * set_threadpool_fn = (decltype(ggml_backend_cpu_set_threadpool) *) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cpu_set_threadpool");
        if (set_threadpool_fn) {
            set_threadpool_fn(backend_cpu, tp);
        }
    }

    // set the number of threads for all the backends
    for (const auto & set_n_threads_fn : set_n_threads_fns) {
        set_n_threads_fn.second(set_n_threads_fn.first, n_threads);
    }

    auto status = ggml_backend_sched_graph_compute_async(sched_use, gf);
    if (status != GGML_STATUS_SUCCESS) {
        LLAMA_LOG_ERROR("%s: ggml_backend_sched_graph_compute_async failed with error %d\n", __func__, status);
    }

    // fprintf(stderr, "splits: %d\n", ggml_backend_sched_get_n_splits(sched));

    return status;
}

llm_graph_cb llama_context::graph_get_cb(ggml_backend_sched_t sched_for_cb) const {
    return [this, sched_for_cb](const llama_ubatch & ubatch, ggml_tensor * cur, const char * name, int il) {
        if (il >= 0) {
            ggml_format_name(cur, "%s-%d", name, il);
        } else {
            ggml_set_name(cur, name);
        }

        // norm may be automatically assigned to the backend of the previous layer, increasing data transfer between backends
        // FIXME: fix in ggml_backend_sched
        const bool full_offload = model.n_gpu_layers() > model.hparams.n_layer;
        if (ubatch.n_tokens < 32 || full_offload) {
            if (il != -1 && strcmp(name, "norm") == 0) {
                const auto & dev_layer = model.dev_layer(il);
                for (const auto & backend : backends) {
                    if (ggml_backend_get_device(backend.get()) == dev_layer) {
                        if (ggml_backend_supports_op(backend.get(), cur)) {
                            ggml_backend_sched_set_tensor_backend(sched_for_cb, cur, backend.get());
                        }
                    }
                }
            }
        }
    };
}

//
// state save/load
//

class llama_io_write_dummy : public llama_io_write_i {
public:
    llama_io_write_dummy(bool skip_tensors) : skip_tensors(skip_tensors) {}

    void write(const void * /* src */, size_t size) override {
        size_written += size;
    }

    void write_tensor(ggml_tensor * /* tensor */, size_t /* offset */, size_t size) override {
        if (skip_tensors) {
            return;
        }

        size_written += size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    const bool skip_tensors;

    size_t size_written = 0;
};

class llama_io_write_host : public llama_io_write_i {
public:
    llama_io_write_host(
            uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_write_host() {
        // TODO: add backend support to batch tensor_get? or some other way to speed this up
        for (const auto & winfo : winfos) {
            ggml_backend_tensor_get(winfo.tensor, winfo.ptr, winfo.offset, winfo.size);
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;
};

class llama_io_read_host : public llama_io_read_i {
public:
    llama_io_read_host(const uint8_t * p, size_t len) : ptr(p), buf_size(len) {}

    ~llama_io_read_host() {
        // flush the reads
        for (const auto & rinfo : rinfos) {
            ggml_backend_tensor_set(rinfo.tensor, rinfo.ptr, rinfo.offset, rinfo.size);
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }

        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});

        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;
};

class llama_io_write_file : public llama_io_write_i {
public:
    llama_io_write_file(llama_file * f) : file(f) {}

    void write(const void * src, size_t size) override {
        file->write_raw(src, size);
        size_written += size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        ggml_backend_tensor_get(tensor, temp_buffer.data(), offset, size);
        write(temp_buffer.data(), temp_buffer.size());
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    llama_file * file;
    size_t size_written = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_read_file : public llama_io_read_i {
public:
    llama_io_read_file(llama_file * f) : file(f) {}

    void read(void * dst, size_t size) override {
        file->read_raw(dst, size);
        size_read += size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        temp_buffer.resize(size);
        read(temp_buffer.data(), size);
        ggml_backend_tensor_set(tensor, temp_buffer.data(), offset, size);
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    llama_file * file;
    size_t size_read = 0;
    std::vector<uint8_t> temp_buffer;
};

class llama_io_write_device : public llama_io_write_i {
public:
    llama_io_write_device(uint8_t * p, size_t len, llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs)  {
    }

    ~llama_io_write_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += winfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            ggml_init_params params = {
                /*.mem_size   =*/ 2*mbuf.n_tensors*ggml_tensor_overhead(),
                /*.mem_buffer =*/ NULL,
                /*.no_alloc   =*/ true,
            };

            mbuf.ctx.reset(ggml_init(params));

            mbuf.org.reserve(mbuf.n_tensors);
            mbuf.cpy.reserve(mbuf.n_tensors);
        }

        for (const auto & winfo : winfos) {
            auto * buft = ggml_backend_buffer_get_type(winfo.tensor->buffer);

            const int64_t n = winfo.size/ggml_element_size(winfo.tensor);

            auto & mbuf = mbufs_new[buft];

            mbuf.org.push_back(ggml_view_1d      (mbuf.ctx.get(), winfo.tensor, n, winfo.offset));
            mbuf.cpy.push_back(ggml_new_tensor_1d(mbuf.ctx.get(), winfo.tensor->type, n));
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            auto & mbuf_cur = mbufs[buft];

            if (!mbuf_cur.buf || mbuf_cur.org.size() != mbuf.org.size() || mbuf_cur.total_size != mbuf.total_size) {
                mbuf_cur = std::move(mbuf);

                mbuf_cur.buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(mbuf_cur.ctx.get(), buft));

                LLAMA_LOG_INFO("%s: allocated '%s' buffer %.3f MiB\n", __func__, ggml_backend_buft_name(buft), mbuf.total_size/1024.0/1024.0);
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.org[i], mbuf_cur.cpy[i]);
            }
        }
    }

    void write(const void * src, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(ptr, src, size);
        ptr += size;
        size_written += size;
        buf_size -= size;
    }

    void write_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save the write for later during destruction
        winfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_written;
    }

private:
    uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_written = 0;

    struct write_info {
        ggml_tensor * tensor;
        uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<write_info> winfos;

    llama_memory_buffers & mbufs;
};

class llama_io_read_device : public llama_io_read_i {
public:
    llama_io_read_device(const uint8_t * p, size_t len, const llama_memory_buffers & mbufs) : ptr(p), buf_size(len), mbufs(mbufs) {
    }

    ~llama_io_read_device() {
        llama_memory_buffers mbufs_new;

        for (const auto & rinfo : rinfos) {
            auto * buft = ggml_backend_buffer_get_type(rinfo.tensor->buffer);

            mbufs_new[buft].n_tensors++;
            mbufs_new[buft].total_size += rinfo.size;
        }

        for (auto & [buft, mbuf] : mbufs_new) {
            const auto & mbuf_cur = mbufs.at(buft);

            if (!mbuf_cur.buf || mbuf_cur.n_tensors != mbuf.n_tensors || mbuf_cur.total_size != mbuf.total_size) {
                GGML_ABORT("%s: memory buffer mismatch\n", __func__);
            }

            for (size_t i = 0; i < mbuf_cur.org.size(); ++i) {
                ggml_backend_tensor_copy(mbuf_cur.cpy[i], mbuf_cur.org[i]);
            }
        }
    }

    void read(void * dst, size_t size) override {
        if (size > buf_size) {
            throw std::runtime_error("unexpectedly reached end of buffer");
        }
        memcpy(dst, ptr, size);
        ptr += size;
        size_read += size;
        buf_size -= size;
    }

    void read_tensor(ggml_tensor * tensor, size_t offset, size_t size) override {
        // save for later during destruction
        rinfos.push_back({tensor, ptr, size, offset});
    }

    size_t n_bytes() override {
        return size_read;
    }

private:
    const uint8_t * ptr;
    size_t buf_size = 0;
    size_t size_read = 0;

    struct read_info {
        ggml_tensor * tensor;
        const uint8_t * ptr;
        size_t size;
        size_t offset;
    };
    std::vector<read_info> rinfos;

    const llama_memory_buffers & mbufs;
};

size_t llama_context::state_get_size() {
    llama_io_write_dummy io(false);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_get_data(uint8_t * dst, size_t size) {
    llama_io_write_host io(dst, size);
    try {
        return state_write_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_set_data(const uint8_t * src, size_t size) {
    llama_io_read_host io(src, size);
    try {
        return state_read_data(io);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

static constexpr uint32_t io_magic = 0xaf143cd8;

size_t llama_context::state_seq_get_size(llama_seq_id seq_id, llama_state_seq_flags flags) {
    llama_io_write_dummy io(flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE);
    try {
        io.write(&io_magic, sizeof(io_magic));
        io.write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error getting state size: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_get_data(llama_seq_id seq_id, uint8_t * dst, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_write_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        io = std::make_unique<llama_io_write_device>(dst, size, mem_storage[seq_id]);
    } else {
        io = std::make_unique<llama_io_write_host>(dst, size);
    }

    try {
        io->write(&io_magic, sizeof(io_magic));
        io->write(&seq_id, sizeof(seq_id));

        return state_seq_write_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving state: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_context::state_seq_set_data(llama_seq_id seq_id, const uint8_t * src, size_t size, llama_state_seq_flags flags) {
    std::unique_ptr<llama_io_read_i> io;
    if (flags & LLAMA_STATE_SEQ_FLAGS_ON_DEVICE) {
        // create a temporary io to read the magic and the src seq_id
        io = std::make_unique<llama_io_read_host>(src, size);

        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));

        GGML_ASSERT(mem_storage.find(seq_id_read) != mem_storage.end());

        io = std::make_unique<llama_io_read_device>(src, size, mem_storage[seq_id_read]);
    } else {
        io = std::make_unique<llama_io_read_host>(src, size);
    }

    try {
        uint32_t magic_read;
        io->read(&magic_read, sizeof(magic_read));
        if (io_magic != magic_read) {
            throw std::runtime_error("wrong sequence state magic");
        }

        const bool need_seq_match = (flags & LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);

        llama_seq_id seq_id_read;
        io->read(&seq_id_read, sizeof(seq_id_read));
        if (need_seq_match && seq_id != seq_id_read) {
            throw std::runtime_error("wrong sequence id");
        }

        return state_seq_read_data(*io, seq_id, flags);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading state: %s\n", __func__, err.what());
        return 0;
    }
}

bool llama_context::state_load_file(const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // sanity checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_SESSION_MAGIC || version != LLAMA_SESSION_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
            return false;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in session file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return false;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t n_state_size_cur = file.size() - file.tell();

        llama_io_read_file io( &file);
        const size_t n_read = state_read_data(io);

        if (n_read != n_state_size_cur) {
            LLAMA_LOG_ERROR("%s: did not read all of the session file data! size %zu, got %zu\n", __func__, n_state_size_cur, n_read);
            return false;
        }
    }

    return true;
}

bool llama_context::state_save_file(const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_SESSION_MAGIC);
    file.write_u32(LLAMA_SESSION_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_write_data(io);

    return true;
}

size_t llama_context::state_seq_load_file(llama_seq_id seq_id, const char * filepath, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    llama_file file(filepath, "rb");

    // version checks
    {
        const uint32_t magic   = file.read_u32();
        const uint32_t version = file.read_u32();

        if (magic != LLAMA_STATE_SEQ_MAGIC || version != LLAMA_STATE_SEQ_VERSION) {
            LLAMA_LOG_ERROR("%s: unknown (magic, version) for sequence state file: %08x, %08x\n", __func__, magic, version);
            return 0;
        }
    }

    // load the prompt
    {
        const uint32_t n_token_count = file.read_u32();

        if (n_token_count > n_token_capacity) {
            LLAMA_LOG_ERROR("%s: token count in sequence state file exceeded capacity! %u > %zu\n", __func__, n_token_count, n_token_capacity);
            return 0;
        }

        file.read_raw(tokens_out, sizeof(llama_token) * n_token_count);
        *n_token_count_out = n_token_count;
    }

    // restore the context state
    {
        const size_t state_size = file.size() - file.tell();
        llama_io_read_file io(&file);
        const size_t nread = state_seq_read_data(io, seq_id, 0);
        if (!nread) {
            LLAMA_LOG_ERROR("%s: failed to restore sequence state\n", __func__);
            return 0;
        }
        GGML_ASSERT(nread <= state_size);
        GGML_ASSERT(nread + sizeof(uint32_t) * 3 + sizeof(llama_token) * *n_token_count_out == file.tell());
    }

    return file.tell();
}

size_t llama_context::state_seq_save_file(llama_seq_id seq_id, const char * filepath, const llama_token * tokens, size_t n_token_count) {
    llama_file file(filepath, "wb");

    file.write_u32(LLAMA_STATE_SEQ_MAGIC);
    file.write_u32(LLAMA_STATE_SEQ_VERSION);

    // save the prompt
    file.write_u32((uint32_t) n_token_count);
    file.write_raw(tokens, sizeof(llama_token) * n_token_count);

    // save the context state using stream saving
    llama_io_write_file io(&file);
    state_seq_write_data(io, seq_id, 0);

    const size_t res = file.tell();
    GGML_ASSERT(res == sizeof(uint32_t) * 3 + sizeof(llama_token) * n_token_count + io.n_bytes());

    return res;
}

size_t llama_context::state_write_data(llama_io_write_i & io) {
    LLAMA_LOG_DEBUG("%s: writing state\n", __func__);

    // write model info
    {
        LLAMA_LOG_DEBUG("%s: - writing model info\n", __func__);

        const std::string arch_str = llm_arch_name(model.arch);
        io.write_string(arch_str);
        // TODO: add more model-specific info which should prevent loading the session file if not identical
    }

    if (memory != nullptr) {
        LLAMA_LOG_DEBUG("%s: - writing memory module\n", __func__);
        memory->state_write(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_read_data(llama_io_read_i & io) {
    LLAMA_LOG_DEBUG("%s: reading state\n", __func__);

    // read model info
    {
        LLAMA_LOG_DEBUG("%s: - reading model info\n", __func__);

        const std::string cur_arch_str = llm_arch_name(model.arch);

        std::string arch_str;
        io.read_string(arch_str);
        if (cur_arch_str != arch_str) {
            throw std::runtime_error(format("wrong model arch: '%s' instead of '%s'", arch_str.c_str(), cur_arch_str.c_str()));
        }
        // TODO: add more info which needs to be identical but which is not verified otherwise
    }

    if (memory) {
        LLAMA_LOG_DEBUG("%s: - reading memory module\n", __func__);

        memory->state_read(io);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_write_data(llama_io_write_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_write(io, seq_id, flags);
    }

    return io.n_bytes();
}

size_t llama_context::state_seq_read_data(llama_io_read_i & io, llama_seq_id seq_id, llama_state_seq_flags flags) {
    GGML_UNUSED(seq_id);

    if (memory) {
        memory->state_read(io, seq_id, flags);
    }

    return io.n_bytes();
}

//
// perf
//

llama_perf_context_data llama_context::perf_get_data() const {
    llama_perf_context_data data = {};

    data.t_start_ms  = 1e-3 * t_start_us;
    data.t_load_ms   = 1e-3 * t_load_us;
    data.t_p_eval_ms = 1e-3 * t_p_eval_us;
    data.t_eval_ms   = 1e-3 * t_eval_us;
    data.n_p_eval    = std::max(1, n_p_eval);
    data.n_eval      = std::max(1, n_eval);
    data.n_reused    = std::max(0, n_reused);

    return data;
}

void llama_context::perf_reset() {
    t_start_us  = ggml_time_us();
    t_eval_us   = n_eval = 0;
    t_p_eval_us = n_p_eval = 0;
    n_reused    = 0;
}

llama_memory_breakdown llama_context::memory_breakdown() const {
    std::map<ggml_backend_buffer_type_t, llama_memory_breakdown_data> ret;
    for (const auto & [buft, size] : model.memory_breakdown()) {
        ret[buft].model += size;
    }
    if (memory) {
        for (const auto & [buft, size] : memory->memory_breakdown()) {
            ret[buft].context += size;
        }
    }
    if (model.hparams.no_alloc) {
        for (size_t i = 0; i < backends.size(); ++i) {
            ggml_backend_t             backend = backends[i].get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += backend_buf_exp_size[i];
        }
    } else {
        for (const auto & backend_ptr : backends) {
            ggml_backend_t             backend = backend_ptr.get();
            ggml_backend_buffer_type_t buft    = ggml_backend_sched_get_buffer_type(sched.get(), backend);
            ret[buft].compute += ggml_backend_sched_get_buffer_size(sched.get(), backend);
        }
    }
    return ret;
}

//
// training
//

static void llama_set_param(struct ggml_tensor * tensor, llama_opt_param_filter param_filter, void * userdata) {
    if (!tensor || tensor->type != GGML_TYPE_F32) {
        return;
    }
    if (!param_filter(tensor, userdata)) {
        return;
    }
    if (strcmp(tensor->name, "token_embd.weight") == 0) {
        return; // FIXME
    }
    if (strcmp(tensor->name, "rope_freqs.weight") == 0) {
        return; // FIXME
    }
    ggml_set_param(tensor);
}

void llama_context::opt_init(struct llama_model * model, struct llama_opt_params lopt_params) {
    GGML_ASSERT(!opt_ctx);
    model->hparams.n_ctx_train = lopt_params.n_ctx_train > 0 ? lopt_params.n_ctx_train : n_ctx();
    const uint32_t n_batch     = std::min(this->n_batch(),  model->hparams.n_ctx_train);
    const uint32_t n_ubatch    = std::min(this->n_ubatch(), n_batch);
    GGML_ASSERT(model->hparams.n_ctx_train % n_batch  == 0);
    GGML_ASSERT(n_batch                    % n_ubatch == 0);

    ggml_opt_params opt_params = ggml_opt_default_params(sched.get(), GGML_OPT_LOSS_TYPE_CROSS_ENTROPY);
    opt_params.opt_period      = n_batch / n_ubatch;
    opt_params.get_opt_pars    = lopt_params.get_opt_pars;
    opt_params.get_opt_pars_ud = lopt_params.get_opt_pars_ud;
    opt_params.optimizer       = lopt_params.optimizer_type;
    opt_ctx = ggml_opt_init(opt_params);

    llama_opt_param_filter param_filter = lopt_params.param_filter;
    void * param_filter_ud              = lopt_params.param_filter_ud;

  //llama_set_param(model->tok_embd,        param_filter, param_filter_ud); // FIXME
    llama_set_param(model->type_embd,       param_filter, param_filter_ud);
    llama_set_param(model->pos_embd,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm,        param_filter, param_filter_ud);
    llama_set_param(model->tok_norm_b,      param_filter, param_filter_ud);
    llama_set_param(model->output_norm,     param_filter, param_filter_ud);
    llama_set_param(model->output_norm_b,   param_filter, param_filter_ud);
    llama_set_param(model->output,          param_filter, param_filter_ud);
    llama_set_param(model->output_b,        param_filter, param_filter_ud);
    llama_set_param(model->output_norm_enc, param_filter, param_filter_ud);
    llama_set_param(model->cls,             param_filter, param_filter_ud);
    llama_set_param(model->cls_b,           param_filter, param_filter_ud);
    llama_set_param(model->cls_out,         param_filter, param_filter_ud);
    llama_set_param(model->cls_out_b,       param_filter, param_filter_ud);
    llama_set_param(model->cls_norm,        param_filter, param_filter_ud);

    for (struct llama_layer & layer : model->layers) {
        for (size_t i = 0; i < sizeof(layer)/sizeof(struct ggml_tensor *); ++i) {
            llama_set_param(reinterpret_cast<struct ggml_tensor **>(&layer)[i], param_filter, param_filter_ud);
        }
    }
}

void llama_context::opt_epoch_iter(
        ggml_opt_dataset_t               dataset,
        ggml_opt_result_t                result,
        const std::vector<llama_token> & tokens,
        const std::vector<llama_token> & labels_sparse,
        llama_batch                    & batch,
        ggml_opt_epoch_callback          callback,
        bool                             train,
        int64_t                          idata_in_loop,
        int64_t                          ndata_in_loop,
        int64_t                          t_loop_start) {
    GGML_ASSERT(opt_ctx);
    const uint32_t n_ctx    = llama_model_n_ctx_train(&model);
    const uint32_t n_batch  = std::min(this->n_batch(),  n_ctx);
    const uint32_t n_ubatch = std::min(this->n_ubatch(), n_batch);

    memory->clear(true);

    for (uint32_t pos_ctx = 0; pos_ctx < n_ctx; pos_ctx += n_batch) {
        batch.n_tokens = n_batch;
        for (uint32_t pos_batch = 0; pos_batch < n_batch; ++pos_batch) {
            batch.token   [pos_batch]    = tokens[pos_ctx + pos_batch];
            batch.pos     [pos_batch]    = pos_ctx + pos_batch;
            batch.n_seq_id[pos_batch]    = 1;
            batch.seq_id  [pos_batch][0] = 0;
            batch.logits  [pos_batch]    = true;
        }

        if (!balloc->init(batch, model.vocab, nullptr, model.hparams.n_embd_inp(), cparams.kv_unified ? LLAMA_MAX_SEQ : cparams.n_seq_max, true)) {
            LLAMA_LOG_ERROR("%s: failed to initialize batch\n", __func__);
            return;
        }

        const uint32_t n_tokens_all = balloc->get_n_tokens();

        n_queued_tokens += n_tokens_all;

        embd_seq.clear();

        uint32_t n_outputs_all = n_tokens_all;

        auto mctx = memory->init_batch(*balloc, cparams.n_ubatch, true);
        if (!mctx || mctx->get_status() != LLAMA_MEMORY_STATUS_SUCCESS) {
            LLAMA_LOG_ERROR("%s: could not initialize batch\n", __func__);
            break;
        }

        // reserve output buffer
        if (output_reserve(n_outputs_all) < n_outputs_all) {
            LLAMA_LOG_ERROR("%s: could not reserve space for batch with %d outputs\n", __func__, n_outputs_all);
            GGML_ABORT("TODO: handle this error");
        };

        uint32_t pos_batch = 0;
        do {
            const auto & ubatch = mctx->get_ubatch();

            n_outputs = ubatch.n_tokens;

            if (!mctx->apply()) {
                LLAMA_LOG_ERROR("%s: failed to update the memory context\n", __func__);
                break;
            }

            auto * res = gf_res_prev.get();

            const auto gparams = graph_params(res, ubatch, mctx.get(), LLM_GRAPH_TYPE_DEFAULT);

            res->reset();

            auto * gf = model.build_graph(gparams);

            struct ggml_context * ctx_compute_opt;
            {
                const size_t size_gf = ggml_graph_size(gf);
                const size_t size_meta = 4*size_gf*ggml_tensor_overhead() + 2*ggml_graph_overhead_custom(size_gf, /*grads = */ true);
                struct ggml_init_params params = {
                    /*.mem_size   =*/ size_meta,
                    /*.mem_buffer =*/ nullptr,
                    /*.no_alloc   =*/ true,
                };
                ctx_compute_opt = ggml_init(params);
            }
            ggml_opt_prepare_alloc(opt_ctx, ctx_compute_opt, gf, res->get_inp_tokens(), res->get_logits());
            ggml_opt_alloc(opt_ctx, train);

            res->set_inputs(&ubatch);
            {
                struct ggml_tensor * labels = ggml_opt_labels(opt_ctx);
                GGML_ASSERT(labels->ne[1] == n_ubatch);
                ggml_set_zero(labels);
                const float onef = 1.0f;
                for (uint32_t pos_ubatch = 0; pos_ubatch < n_ubatch; ++pos_ubatch) {
                    const uint32_t ilabel = pos_ctx + pos_batch + pos_ubatch;
                    GGML_ASSERT(labels_sparse[ilabel] < labels->ne[0]);
                    ggml_backend_tensor_set(labels, &onef, (pos_ubatch*labels->ne[0] + labels_sparse[ilabel])*sizeof(float), sizeof(float));
                }
            }
            ggml_opt_eval(opt_ctx, result);
            if (callback) {
                callback(train, opt_ctx, dataset, result, idata_in_loop + (pos_ctx + pos_batch)/n_ubatch + 1, ndata_in_loop, t_loop_start);
            }
            ggml_free(ctx_compute_opt);

            pos_batch += ubatch.n_tokens;
        } while (mctx->next());
    }
}

void llama_context::opt_epoch(
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    const uint32_t n_ctx    = this->n_ctx();
    const uint32_t n_batch  = std::min(cparams.n_batch,  n_ctx);
    const uint32_t n_ubatch = std::min(cparams.n_ubatch, n_batch);
    const  int64_t ndata    = ggml_opt_dataset_ndata(dataset);

    GGML_ASSERT(idata_split >= 0);
    GGML_ASSERT(idata_split <= ndata);

    const uint32_t ubatch_per_ctx = n_ctx / n_ubatch;

    struct llama_batch batch = llama_batch_init(n_batch, 0, 1);
    std::vector<llama_token>        tokens(n_ctx);
    std::vector<llama_token> labels_sparse(n_ctx);

    int64_t idata = 0;

    int64_t t_loop_start = ggml_time_us();
    int64_t ndata_in_loop = idata_split*ubatch_per_ctx;
    for (; idata < idata_split; ++idata) {
        constexpr bool train = true;
        const int64_t idata_in_loop = idata*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_train, tokens, labels_sparse, batch,
            callback_train, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    t_loop_start = ggml_time_us();
    ndata_in_loop = (ndata - idata_split)*ubatch_per_ctx;
    for (; idata < ndata; ++idata) {
        constexpr bool train = false;
        const int64_t idata_in_loop = (idata - idata_split)*ubatch_per_ctx;

        ggml_opt_dataset_get_batch_host(dataset, tokens.data(), n_ctx*sizeof(llama_token), labels_sparse.data(), idata);
        opt_epoch_iter(dataset, result_eval, tokens, labels_sparse, batch,
            callback_eval, train, idata_in_loop, ndata_in_loop, t_loop_start);
    }

    llama_batch_free(batch);
}

//
// interface implementation
//

llama_context_params llama_context_default_params() {
    llama_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_batch                     =*/ 2048,
        /*.n_ubatch                    =*/ 512,
        /*.n_seq_max                   =*/ 1,
        /*.n_threads                   =*/ GGML_DEFAULT_N_THREADS, // TODO: better default
        /*.n_threads_batch             =*/ GGML_DEFAULT_N_THREADS,
        /*.rope_scaling_type           =*/ LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED,
        /*.pooling_type                =*/ LLAMA_POOLING_TYPE_UNSPECIFIED,
        /*.attention_type              =*/ LLAMA_ATTENTION_TYPE_UNSPECIFIED,
        /*.flash_attn_type             =*/ LLAMA_FLASH_ATTN_TYPE_AUTO,
        /*.rope_freq_base              =*/ 0.0f,
        /*.rope_freq_scale             =*/ 0.0f,
        /*.yarn_ext_factor             =*/ -1.0f,
        /*.yarn_attn_factor            =*/ -1.0f,
        /*.yarn_beta_fast              =*/ -1.0f,
        /*.yarn_beta_slow              =*/ -1.0f,
        /*.yarn_orig_ctx               =*/ 0,
        /*.defrag_thold                =*/ -1.0f,
        /*.cb_eval                     =*/ nullptr,
        /*.cb_eval_user_data           =*/ nullptr,
        /*.type_k                      =*/ GGML_TYPE_F16,
        /*.type_v                      =*/ GGML_TYPE_F16,
        /*.abort_callback              =*/ nullptr,
        /*.abort_callback_data         =*/ nullptr,
        /*.embeddings                  =*/ false,
        /*.offload_kqv                 =*/ true,
        /*.no_perf                     =*/ true,
        /*.op_offload                  =*/ true,
        /*.swa_full                    =*/ true,
        /*.kv_unified                  =*/ false,
        /*.dflash_max_ctx              =*/ -1, // auto-scale
        /*.dflash_topk                 =*/ 1,  // chain mode
        /*.dflash_emit_logits          =*/ false, // best-first opt-in
        /*.dflash_inline_encoder       =*/ false, // phase 1: plumbing only
        /*.dflash_inline_n_embd_dft    =*/ 0,
        /*.dflash_inline_n_head_kv_dft =*/ 0,
        /*.dflash_inline_n_embd_head_dft  =*/ 0,
        /*.dflash_inline_n_target_layers =*/ 0,
        /*.dflash_gdn_history          =*/ false,
        /*.dflash_gdn_history_f16      =*/ false,
        /*.sampler                     =*/ nullptr,
        /*.n_sampler                   =*/ 0,
    };

    return result;
}

llama_context * llama_init_from_model(
                 llama_model * model,
        llama_context_params   params) {
    if (!model) {
        LLAMA_LOG_ERROR("%s: model cannot be NULL\n", __func__);
        return nullptr;
    }

    if (params.n_batch == 0 && params.n_ubatch == 0) {
        LLAMA_LOG_ERROR("%s: n_batch and n_ubatch cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.n_ctx == 0 && model->hparams.n_ctx_train == 0) {
        LLAMA_LOG_ERROR("%s: n_ctx and model->hparams.n_ctx_train cannot both be zero\n", __func__);
        return nullptr;
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && model->arch == LLM_ARCH_GROK) {
        LLAMA_LOG_WARN("%s: flash_attn is not compatible with Grok - forcing off\n", __func__);
        params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }

    if (model->split_mode() == LLAMA_SPLIT_MODE_TENSOR) {
        if (params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO) {
            LLAMA_LOG_INFO("%s: enabling flash_attn since it is required for SPLIT_MODE_TENSOR\n", __func__);
            params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        }
        if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_ENABLED) {
            LLAMA_LOG_ERROR("%s: SPLIT_MODE_TENSOR requires flash_attn to be enabled\n", __func__);
            return nullptr;
        }
        if (ggml_is_quantized(params.type_k) || ggml_is_quantized(params.type_v)) {
            LLAMA_LOG_ERROR("%s: simultaneous use of SPLIT_MODE_TENSOR and KV cache quantization not implemented\n", __func__);
            return nullptr;
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_k)) {
        const uint32_t blck_size = ggml_blck_size(params.type_k);
        for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
            if (model->hparams.n_embd_head_k(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: K cache type %s with block size %u does not divide n_embd_head_k=%u\n",
                    __func__, ggml_type_name(params.type_k), blck_size, model->hparams.n_embd_head_k(il));
                return nullptr;
            }
        }
    }

    if (params.flash_attn_type != LLAMA_FLASH_ATTN_TYPE_DISABLED && ggml_is_quantized(params.type_v)) {
        const uint32_t blck_size = ggml_blck_size(params.type_v);
        for (uint32_t il = 0; il < model->hparams.n_layer; ++il) {
            if (model->hparams.n_embd_head_v(il) % blck_size != 0) {
                LLAMA_LOG_ERROR("%s: V cache type %s with block size %u does not divide n_embd_head_v=%u\n",
                    __func__, ggml_type_name(params.type_v), blck_size, model->hparams.n_embd_head_v(il));
                return nullptr;
            }
        }
    }

    if (ggml_is_quantized(params.type_v) && params.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED) {
        LLAMA_LOG_ERROR("%s: V cache quantization requires flash_attn\n", __func__);
        return nullptr;
    }

    if (params.pooling_type != LLAMA_POOLING_TYPE_UNSPECIFIED &&
        params.pooling_type != model->hparams.pooling_type) {
        //user-specified pooling-type is different from the model default
        LLAMA_LOG_WARN("%s: model default pooling_type is [%d], but [%d] was specified\n", __func__,
                       model->hparams.pooling_type, params.pooling_type);
    }

    try {
        auto * ctx = new llama_context(*model, params);
        return ctx;
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: failed to initialize the context: %s\n", __func__, err.what());
    }

    return nullptr;
}

// deprecated
llama_context * llama_new_context_with_model(
                 llama_model * model,
        llama_context_params   params) {
    return llama_init_from_model(model, params);
}

void llama_free(llama_context * ctx) {
    delete ctx;
}

uint32_t llama_n_ctx(const llama_context * ctx) {
    return ctx->n_ctx();
}

uint32_t llama_n_ctx_seq(const llama_context * ctx) {
    return ctx->n_ctx_seq();
}

uint32_t llama_n_batch(const llama_context * ctx) {
    return ctx->n_batch();
}

uint32_t llama_n_ubatch(const llama_context * ctx) {
    return ctx->n_ubatch();
}

uint32_t llama_n_seq_max(const llama_context * ctx) {
    return ctx->n_seq_max();
}

const llama_model * llama_get_model(const llama_context * ctx) {
    return &ctx->get_model();
}

enum llama_pooling_type llama_pooling_type(const llama_context * ctx) {
    return ctx->pooling_type();
}

void llama_attach_threadpool(
            llama_context * ctx,
        ggml_threadpool_t   threadpool,
        ggml_threadpool_t   threadpool_batch) {
    ctx->attach_threadpool(threadpool, threadpool_batch);
}

void llama_detach_threadpool(llama_context * ctx) {
    ctx->detach_threadpool();
}

void llama_set_n_threads(llama_context * ctx, int32_t n_threads, int32_t n_threads_batch) {
    ctx->set_n_threads(n_threads, n_threads_batch);
}

int32_t llama_n_threads(llama_context * ctx) {
    return ctx->n_threads();
}

int32_t llama_n_threads_batch(llama_context * ctx) {
    return ctx->n_threads_batch();
}

void llama_set_abort_callback(llama_context * ctx, bool (*abort_callback)(void * data), void * abort_callback_data) {
    ctx->set_abort_callback(abort_callback, abort_callback_data);
}

void llama_set_embeddings(llama_context * ctx, bool embeddings) {
    ctx->set_embeddings(embeddings);
}

void llama_set_causal_attn(llama_context * ctx, bool causal_attn) {
    ctx->set_causal_attn(causal_attn);
}

void llama_set_warmup(llama_context * ctx, bool warmup) {
    ctx->set_warmup(warmup);
}

void llama_synchronize(llama_context * ctx) {
    ctx->synchronize();
}

float * llama_get_logits(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_logits();
}

float * llama_get_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    float * res = nullptr;

    res = ctx->get_sampled_logits_ith(i);

    if (!res) {
        res = ctx->get_logits_ith(i);
    }

    return res;
}

float * llama_get_embeddings(llama_context * ctx) {
    ctx->synchronize();

    return ctx->get_embeddings();
}

float * llama_get_embeddings_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_embeddings_ith(i);
}

float * llama_get_embeddings_seq(llama_context * ctx, llama_seq_id seq_id) {
    ctx->synchronize();

    return ctx->get_embeddings_seq(seq_id);
}

bool llama_set_sampler(llama_context * ctx, llama_seq_id seq_id, llama_sampler * smpl) {
    return ctx->set_sampler(seq_id, smpl);
}

llama_token llama_get_sampled_token_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_token_ith(i);
}

float * llama_get_sampled_probs_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_probs_ith(i);
}

float * llama_get_sampled_logits_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return ctx->get_sampled_logits_ith(i);
}

llama_token * llama_get_sampled_candidates_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return const_cast<llama_token *>(ctx->get_sampled_candidates_ith(i));
}

uint32_t llama_get_sampled_candidates_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_candidates_count(i));
}

uint32_t llama_get_sampled_logits_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_logits_count(i));
}

uint32_t llama_get_sampled_probs_count_ith(llama_context * ctx, int32_t i) {
    ctx->synchronize();

    return static_cast<uint32_t>(ctx->get_sampled_probs_count(i));
}

struct ggml_cgraph * llama_graph_reserve(
        struct llama_context * ctx,
        uint32_t n_tokens,
        uint32_t n_seqs,
        uint32_t n_outputs) {
    auto * memory = ctx->get_memory();
    llama_memory_context_ptr mctx;
    if (memory) {
        mctx = memory->init_full();
    }
    return ctx->graph_reserve(n_tokens, n_seqs, n_outputs, mctx.get());
}

// llama adapter API

int32_t llama_set_adapters_lora(
            llama_context * ctx,
            llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales) {
    if (adapters == nullptr || scales == nullptr) {
        GGML_ASSERT(n_adapters == 0 && "invalid llama_set_adapters_lora call");
    }

    ctx->set_adapters_lora(adapters, n_adapters, scales);

    return 0;
}

int32_t llama_set_adapter_cvec(
        llama_context * ctx,
          const float * data,
               size_t   len,
              int32_t   n_embd,
              int32_t   il_start,
              int32_t   il_end) {
    bool res = ctx->set_adapter_cvec(data, len, n_embd, il_start, il_end);

    return res ? 0 : -1;
}

//
// memory
//

llama_memory_t llama_get_memory(const struct llama_context * ctx) {
    return ctx->get_memory();
}

void llama_memory_clear(llama_memory_t mem, bool data) {
    if (!mem) {
        return;
    }

    mem->clear(data);
}

bool llama_memory_seq_rm(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    return mem->seq_rm(seq_id, p0, p1);
}

bool llama_dflash_memory_seq_rm_partial_tail_state_managed_externally(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return true;
    }

    // Recognize the two memory backends that have GDN/recurrent state
    // (and therefore know how to handle the partial-tail rewind). For
    // any other backend, fall back to the regular seq_rm — those
    // backends support partial-tail removal natively.
    if (auto * h = dynamic_cast<llama_memory_hybrid *>(mem)) {
        return h->seq_rm_partial_tail_state_managed_externally(seq_id, p0, p1);
    }
    if (auto * r = dynamic_cast<llama_memory_recurrent *>(mem)) {
        return r->seq_rm_partial_tail_state_managed_externally(seq_id, p0, p1);
    }
    return mem->seq_rm(seq_id, p0, p1);
}

bool llama_memory_keep_positions_range(
        llama_memory_t    mem,
        llama_seq_id      seq_id,
        const llama_pos * positions,
        int32_t           n_positions,
        llama_pos         p_min) {
    if (!mem) {
        return true;
    }
    return mem->keep_positions_range(seq_id, positions, n_positions, p_min);
}

bool llama_memory_keep_cells_dfs_ordinals_range(
        llama_memory_t  mem,
        llama_seq_id    seq_id,
        const int32_t * dfs_keep,
        int32_t         n_keep,
        llama_pos       p_min) {
    if (!mem) {
        return true;
    }
    return mem->keep_cells_dfs_ordinals_range(seq_id, dfs_keep, n_keep, p_min);
}

void llama_memory_seq_cp(
        llama_memory_t mem,
          llama_seq_id seq_id_src,
          llama_seq_id seq_id_dst,
             llama_pos p0,
             llama_pos p1) {
    if (!mem) {
        return;
    }

    mem->seq_cp(seq_id_src, seq_id_dst, p0, p1);
}

void llama_memory_seq_keep(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return;
    }

    mem->seq_keep(seq_id);
}

void llama_memory_seq_add(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
             llama_pos delta) {
    if (!mem) {
        return;
    }

    mem->seq_add(seq_id, p0, p1, delta);
}

void llama_memory_seq_div(
        llama_memory_t mem,
          llama_seq_id seq_id,
             llama_pos p0,
             llama_pos p1,
                   int d) {
    if (!mem) {
        return;
    }

    mem->seq_div(seq_id, p0, p1, d);
}

llama_pos llama_memory_seq_pos_min(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_min(seq_id);
}

llama_pos llama_memory_seq_pos_max(
        llama_memory_t mem,
          llama_seq_id seq_id) {
    if (!mem) {
        return -1;
    }

    return mem->seq_pos_max(seq_id);
}

bool llama_memory_can_shift(llama_memory_t mem) {
    if (!mem) {
        return false;
    }

    return mem->get_can_shift();
}

// llama state API

// deprecated
size_t llama_get_state_size(llama_context * ctx) {
    return llama_state_get_size(ctx);
}

// deprecated
size_t llama_copy_state_data(llama_context * ctx, uint8_t * dst) {
    return llama_state_get_data(ctx, dst, -1);
}

// deprecated
size_t llama_set_state_data(llama_context * ctx, const uint8_t * src) {
    return llama_state_set_data(ctx, src, -1);
}

// deprecated
bool llama_load_session_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    return llama_state_load_file(ctx, path_session, tokens_out, n_token_capacity, n_token_count_out);
}

// deprecated
bool llama_save_session_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    return llama_state_save_file(ctx, path_session, tokens, n_token_count);
}

// Returns the *actual* size of the state.
// Intended to be used when saving to state to a buffer.
size_t llama_state_get_size(llama_context * ctx) {
    return ctx->state_get_size();
}

size_t llama_state_get_data(llama_context * ctx, uint8_t * dst, size_t size) {
    ctx->synchronize();

    return ctx->state_get_data(dst, size);
}

// Sets the state reading from the specified source address
size_t llama_state_set_data(llama_context * ctx, const uint8_t * src, size_t size) {
    ctx->synchronize();

    return ctx->state_set_data(src, size);
}

bool llama_state_load_file(llama_context * ctx, const char * path_session, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_load_file(path_session, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading session file: %s\n", __func__, err.what());
        return false;
    }
}

bool llama_state_save_file(llama_context * ctx, const char * path_session, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_save_file(path_session, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving session file: %s\n", __func__, err.what());
        return false;
    }
}

size_t llama_state_seq_get_size(llama_context * ctx, llama_seq_id seq_id) {
    return llama_state_seq_get_size_ext(ctx, seq_id, 0);
}

size_t llama_state_seq_get_data(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_get_data_ext(ctx, dst, size, seq_id, 0);
}

size_t llama_state_seq_set_data(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id) {
    return llama_state_seq_set_data_ext(ctx, src, size, seq_id, 0);
}

size_t llama_state_seq_get_size_ext(llama_context * ctx, llama_seq_id seq_id, llama_state_seq_flags flags) {
    return ctx->state_seq_get_size(seq_id, flags);
}

size_t llama_state_seq_get_data_ext(llama_context * ctx, uint8_t * dst, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_get_data(seq_id, dst, size, flags);
}
size_t llama_state_seq_set_data_ext(llama_context * ctx, const uint8_t * src, size_t size, llama_seq_id seq_id, llama_state_seq_flags flags) {
    ctx->synchronize();

    return ctx->state_seq_set_data(seq_id, src, size, flags);
}

size_t llama_state_seq_save_file(llama_context * ctx, const char * filepath, llama_seq_id seq_id, const llama_token * tokens, size_t n_token_count) {
    ctx->synchronize();

    try {
        return ctx->state_seq_save_file(seq_id, filepath, tokens, n_token_count);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error saving sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

size_t llama_state_seq_load_file(llama_context * ctx, const char * filepath, llama_seq_id dest_seq_id, llama_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    ctx->synchronize();

    try {
        return ctx->state_seq_load_file(dest_seq_id, filepath, tokens_out, n_token_capacity, n_token_count_out);
    } catch (const std::exception & err) {
        LLAMA_LOG_ERROR("%s: error loading sequence state file: %s\n", __func__, err.what());
        return 0;
    }
}

///

int32_t llama_encode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->encode(batch);
    if (ret != 0) {
        LLAMA_LOG_ERROR("%s: failed to encode, ret = %d\n", __func__, ret);
    }

    return ret;
}

int32_t llama_decode(
        llama_context * ctx,
          llama_batch   batch) {
    const int ret = ctx->decode(batch);
    if (ret != 0 && ret != 1) {
        LLAMA_LOG_ERROR("%s: failed to decode, ret = %d\n", __func__, ret);
    }

    return ret;
}

// DFlash speculative-decoding C API

void llama_dflash_set_capture(struct llama_context * ctx,
                              const int32_t * layer_ids,
                              size_t          n_layer_ids,
                              int64_t         n_embd_target) {
    if (ctx == nullptr) return;
    ctx->set_dflash_capture(layer_ids, n_layer_ids, n_embd_target);
}

const float * llama_dflash_get_captured_features(struct llama_context * ctx,
                                                 int64_t * n_outputs_out) {
    if (ctx == nullptr) {
        if (n_outputs_out) *n_outputs_out = 0;
        return nullptr;
    }
    return ctx->get_dflash_captured_features(n_outputs_out);
}

const int32_t * llama_dflash_get_draft_topk(struct llama_context * ctx,
                                            int64_t * n_outputs_out,
                                            uint32_t * topk_out) {
    if (ctx == nullptr) {
        if (n_outputs_out) *n_outputs_out = 0;
        if (topk_out)      *topk_out      = 0;
        return nullptr;
    }
    ctx->synchronize();
    ctx->dflash_finalize_draft_topk();
    return ctx->get_dflash_draft_topk(n_outputs_out, topk_out);
}

int32_t llama_dflash_extend(llama_context * ctx,
                            const float * target_hidden_new,
                            int64_t n_new,
                            int64_t pos_start) {
    if (ctx == nullptr) return -1;
    return ctx->dflash_extend(target_hidden_new, n_new, pos_start);
}

int32_t llama_dflash_extend_from_ctx(llama_context * dst_ctx,
                                     llama_context * src_ctx,
                                     int64_t         src_row_offset,
                                     int64_t         n_keep,
                                     int64_t         pos_start) {
    if (dst_ctx == nullptr || src_ctx == nullptr) return -1;
    ggml_tensor * src_packed = src_ctx->get_dflash_last_packed_captures();
    if (src_packed == nullptr) {
        LLAMA_LOG_ERROR("%s: src_ctx has no packed captures (was decode() run with capture_layer_ids set?)\n",
                __func__);
        return -1;
    }
    return dst_ctx->dflash_extend_from_tensor(src_packed, src_row_offset, n_keep, pos_start);
}

bool llama_dflash_bind_inline_side_store(llama_context * ctx_tgt,
                                         llama_context * ctx_dft) {
    if (ctx_tgt == nullptr || ctx_dft == nullptr) {
        return false;
    }
    llama_dflash & ddflash = ctx_dft->get_dflash_mut();
    if (ddflash.ctx_K.empty() || ddflash.ctx_V.empty()) {
        LLAMA_LOG_WARN("%s: ctx_dft has no side store (not a DFlash draft?)\n", __func__);
        return false;
    }
    if (ddflash.ctx_K.size() != ddflash.ctx_V.size()) {
        return false;
    }
    llama_dflash & tdflash = ctx_tgt->get_dflash_mut();
    tdflash.inline_dst_K = ddflash.ctx_K; // vector copy of pointers
    tdflash.inline_dst_V = ddflash.ctx_V;
    return true;
}

void llama_dflash_set_inline_encode_state(llama_context * ctx_tgt,
                                          int64_t         write_offset,
                                          int64_t         pos_start) {
    if (ctx_tgt == nullptr) return;
    llama_dflash & tdflash = ctx_tgt->get_dflash_mut();
    tdflash.inline_write_offset = write_offset;
    tdflash.inline_pos_start    = pos_start;
}

void llama_dflash_inline_advance_ctx_filled(llama_context * ctx_dft,
                                            int64_t         n_keep) {
    if (ctx_dft == nullptr || n_keep <= 0) return;
    llama_dflash & ddflash = ctx_dft->get_dflash_mut();
    ddflash.ctx_filled += n_keep;
    ddflash.n_ctx       = ddflash.ctx_filled;
}

void llama_dflash_set_gdn_history_k_index(llama_context * ctx_tgt,
                                          int32_t         k_index) {
    if (ctx_tgt == nullptr) return;
    llama_dflash & tdflash = ctx_tgt->get_dflash_mut();
    tdflash.gdn_history_k_index = k_index;
    // Clear the per-seq vector so a subsequent chain-mode iteration
    // can't pick up a stale tree-mode k_indices array.
    tdflash.gdn_history_k_indices.clear();
}

void llama_dflash_set_gdn_history_k_index_per_seq(llama_context * ctx_tgt,
                                                  const int32_t * k_indices,
                                                  int32_t         n_seqs) {
    if (ctx_tgt == nullptr) return;
    llama_dflash & tdflash = ctx_tgt->get_dflash_mut();
    if (k_indices == nullptr || n_seqs <= 0) {
        tdflash.gdn_history_k_indices.clear();
        return;
    }
    tdflash.gdn_history_k_indices.assign(k_indices, k_indices + n_seqs);
    // Mirror the first entry onto the scalar so any chain-mode fixup
    // sites still in the graph (n_seqs == 1) pick up something sane.
    tdflash.gdn_history_k_index = k_indices[0];
}

void llama_dflash_set_gdn_history_parent_ids(llama_context * ctx_tgt,
                                             const int32_t * parent_ids,
                                             int32_t         n_tokens,
                                             int32_t         n_seqs) {
    if (ctx_tgt == nullptr) return;
    llama_dflash & tdflash = ctx_tgt->get_dflash_mut();
    if (parent_ids == nullptr || n_tokens <= 0 || n_seqs <= 0) {
        tdflash.gdn_history_parent_ids.clear();
        tdflash.gdn_history_parent_ids_n_tokens = 0;
        tdflash.gdn_history_parent_ids_n_seqs   = 0;
        return;
    }
    const size_t n = (size_t) n_tokens * (size_t) n_seqs;
    tdflash.gdn_history_parent_ids.assign(parent_ids, parent_ids + n);
    tdflash.gdn_history_parent_ids_n_tokens = n_tokens;
    tdflash.gdn_history_parent_ids_n_seqs   = n_seqs;
}

int32_t llama_dflash_inline_encode_from_ctx(llama_context * tgt_ctx,
                                            llama_context * dft_ctx,
                                            int64_t         src_row_offset,
                                            int64_t         n_keep,
                                            int64_t         pos_start) {
    if (tgt_ctx == nullptr || dft_ctx == nullptr) return -1;
    ggml_tensor * src_packed = tgt_ctx->get_dflash_last_packed_captures();
    if (src_packed == nullptr) {
        LLAMA_LOG_ERROR("%s: tgt_ctx has no packed captures (was decode() run with capture_layer_ids set?)\n",
                __func__);
        return -1;
    }
    return tgt_ctx->dflash_inline_encode_from_ctx(dft_ctx, src_packed,
                                                  src_row_offset, n_keep, pos_start);
}

void llama_dflash_set_skip_host_readback(llama_context * ctx, bool skip) {
    if (ctx == nullptr) return;
    ctx->set_dflash_skip_host_readback(skip);
}

int32_t llama_dflash_force_host_readback(llama_context * ctx) {
    if (ctx == nullptr) return -1;
    return ctx->dflash_force_host_readback();
}

void llama_dflash_reset_ctx_kv(llama_context * ctx) {
    if (ctx == nullptr) return;
    ctx->dflash_reset_ctx_kv();
}

void llama_set_tree_mask(struct llama_context * ctx,
                         const uint8_t * visibility,
                         int n_tree_tokens) {
    if (ctx == nullptr) return;
    ctx->set_tree_mask(visibility, n_tree_tokens);
}

void llama_clear_tree_mask(struct llama_context * ctx) {
    if (ctx == nullptr) return;
    ctx->clear_tree_mask();
}

//
// perf
//

llama_perf_context_data llama_perf_context(const llama_context * ctx) {
    llama_perf_context_data data = {};

    if (ctx == nullptr) {
        return data;
    }

    data = ctx->perf_get_data();

    return data;
}

void llama_perf_context_print(const llama_context * ctx) {
    const auto data = llama_perf_context(ctx);

    const double t_end_ms = 1e-3 * ggml_time_us();

    LLAMA_LOG_INFO("%s:        load time = %10.2f ms\n", __func__, data.t_load_ms);
    LLAMA_LOG_INFO("%s: prompt eval time = %10.2f ms / %5d tokens (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_p_eval_ms, data.n_p_eval, data.t_p_eval_ms / data.n_p_eval, 1e3 / data.t_p_eval_ms * data.n_p_eval);
    LLAMA_LOG_INFO("%s:        eval time = %10.2f ms / %5d runs   (%8.2f ms per token, %8.2f tokens per second)\n",
            __func__, data.t_eval_ms, data.n_eval, data.t_eval_ms / data.n_eval, 1e3 / data.t_eval_ms * data.n_eval);
    LLAMA_LOG_INFO("%s:       total time = %10.2f ms / %5d tokens\n", __func__, (t_end_ms - data.t_start_ms), (data.n_p_eval + data.n_eval));
    LLAMA_LOG_INFO("%s:    graphs reused = %10d\n", __func__, data.n_reused);
}

void llama_perf_context_reset(llama_context * ctx) {
    ctx->perf_reset();
}

//
// training
//

bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata) {
    GGML_UNUSED(tensor);
    GGML_UNUSED(userdata);
    return true;
}

void llama_opt_init(struct llama_context * ctx, struct llama_model * model, struct llama_opt_params lopt_params) {
    ctx->opt_init(model, lopt_params);
}

void llama_opt_epoch(
        struct llama_context    * ctx,
        ggml_opt_dataset_t        dataset,
        ggml_opt_result_t         result_train,
        ggml_opt_result_t         result_eval,
        int64_t                   idata_split,
        ggml_opt_epoch_callback   callback_train,
        ggml_opt_epoch_callback   callback_eval) {
    ctx->opt_epoch(
        dataset,
        result_train,
        result_eval,
        idata_split,
        callback_train,
        callback_eval);
}

//
// ext
//

llama_memory_breakdown llama_get_memory_breakdown(const struct llama_context * ctx) {
    return ctx->memory_breakdown();
}
