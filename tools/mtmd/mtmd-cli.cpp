#include "arg.h"
#include "debug.h"
#include "log.h"
#include "common.h"
#include "sampling.h"
#include "llama.h"
#include "ggml.h"
#include "console.h"
#include "chat.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <vector>
#include <chrono>
#include <cstdio>
#include <limits.h>
#include <cinttypes>
#include <clocale>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#define WIN32_LEAN_AND_MEAN
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <signal.h>
#endif

// volatile, because of signal being an interrupt
static volatile bool g_is_generating = false;
static volatile bool g_is_interrupted = false;

/**
 * Please note that this is NOT a production-ready stuff.
 * It is a playground for trying multimodal support in llama.cpp.
 * For contributors: please keep this code simple and easy to understand.
 */

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG(
        "Experimental CLI for multimodal\n\n"
        "Usage: %s [options] -m <model> --mmproj <mmproj> --image <image> --audio <audio> -p <prompt>\n\n"
        "  -m and --mmproj are required\n"
        "  -hf user/repo can replace both -m and --mmproj in most cases\n"
        "  --image, --audio and -p are optional, if NOT provided, the CLI will run in chat mode\n"
        "  to disable using GPU for mmproj model, add --no-mmproj-offload\n",
        argv[0]
    );
}

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
static void sigint_handler(int signo) {
    if (signo == SIGINT) {
        if (g_is_generating) {
            g_is_generating = false;
        } else {
            console::cleanup();
            if (g_is_interrupted) {
                _exit(1);
            }
            g_is_interrupted = true;
        }
    }
}
#endif

struct mtmd_cli_context {
    mtmd::context_ptr ctx_vision;
    common_init_result_ptr llama_init;

    llama_model       * model;
    llama_context     * lctx;
    // [QVAC-21372 item 2 — EXPERIMENTAL] two-context mode: a 2nd model loaded with
    // n_gpu_layers=0 (native CPU weights) does the prefill, then the KV cache is handed to
    // the GPU `lctx` for decode. Avoids item 1's broken CPU-reads-GPU-buffer path. Opt-in
    // via env QVAC_TWO_CTX=1. Costs 2x weight memory (two model loads).
    common_init_result_ptr llama_init_cpu;
    llama_context     * lctx_cpu   = nullptr;
    bool                two_ctx    = false;
    llama_token         first_token = -1; // first token sampled from the CPU prefill ctx
    const llama_vocab * vocab;
    common_sampler    * smpl;
    llama_batch         batch;
    int                 n_batch;

    mtmd::bitmaps bitmaps;

    // chat template
    common_chat_templates_ptr tmpls;
    std::vector<common_chat_msg> chat_history;
    bool use_jinja = false;
    // TODO: support for --system-prompt with /clear command

    // support for legacy templates (models not having EOT token)
    llama_tokens antiprompt_tokens;

    int n_threads    = 1;
    llama_pos n_past = 0;

    common_debug_cb_user_data cb_data;

    mtmd_cli_context(common_params & params) : llama_init(common_init_from_params(params)) {
        model = llama_init->model();
        lctx = llama_init->context();
        vocab = llama_model_get_vocab(model);
        smpl = common_sampler_init(model, params.sampling);
        n_threads = params.cpuparams.n_threads;
        batch = llama_batch_init(1, 0, 1); // batch for next token generation
        n_batch = params.n_batch;

        if (!model || !lctx) {
            exit(1);
        }

        if (!llama_model_chat_template(model, nullptr) && params.chat_template.empty()) {
            LOG_ERR("Model does not have chat template.\n");
            LOG_ERR("  For old llava models, you may need to use '--chat-template vicuna'\n");
            LOG_ERR("  For MobileVLM models, use '--chat-template deepseek'\n");
            LOG_ERR("  For Mistral Small 3.1, use '--chat-template mistral-v7'\n");
            exit(1);
        }

        tmpls = common_chat_templates_init(model, params.chat_template);
        use_jinja = params.use_jinja;
        chat_history.clear();
        LOG_INF("%s: chat template example:\n%s\n", __func__, common_chat_format_example(tmpls.get(), params.use_jinja, params.default_template_kwargs).c_str());

        init_vision_context(params);

        // [QVAC-21372 item 2 — EXPERIMENTAL] optional two-context (CPU-prefill + GPU-decode).
        if (const char * e = std::getenv("QVAC_TWO_CTX"); e && std::atoi(e) != 0) {
            two_ctx = true;
            common_params params_cpu = params;       // copy
            params_cpu.n_gpu_layers = 0;             // native CPU weights -> fast, correct prefill
            llama_init_cpu = common_init_from_params(params_cpu);
            if (!llama_init_cpu || !llama_init_cpu->context()) {
                LOG_ERR("[QVAC_TWO_CTX] failed to load CPU prefill model\n");
                exit(1);
            }
            lctx_cpu = llama_init_cpu->context();
            LOG_INF("[QVAC_TWO_CTX] two-context mode ON: prefill on CPU (ngl=0), decode on GPU\n");
        }

        // load antiprompt tokens for legacy templates
        if (params.chat_template == "vicuna") {
            antiprompt_tokens = common_tokenize(lctx, "ASSISTANT:", false, true);
        } else if (params.chat_template == "deepseek") {
            antiprompt_tokens = common_tokenize(lctx, "###", false, true);
        }
    }

    ~mtmd_cli_context() {
        llama_batch_free(batch);
        common_sampler_free(smpl);
    }

    void init_vision_context(common_params & params) {
        const char * clip_path = params.mmproj.path.c_str();
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu          = params.mmproj_use_gpu;
        mparams.print_timings    = true;
        mparams.n_threads        = params.cpuparams.n_threads;
        mparams.flash_attn_type  = params.flash_attn_type;
        mparams.warmup           = params.warmup;
        mparams.image_min_tokens = params.image_min_tokens;
        mparams.image_max_tokens = params.image_max_tokens;
        if (std::getenv("MTMD_DEBUG_GRAPH") != nullptr) {
            mparams.cb_eval_user_data = &cb_data;
            mparams.cb_eval = common_debug_cb_eval;
        }
        ctx_vision.reset(mtmd_init_from_file(clip_path, model, mparams));
        if (!ctx_vision.get()) {
            LOG_ERR("Failed to load vision model from %s\n", clip_path);
            exit(1);
        }
    }

    bool check_antiprompt(const llama_tokens & generated_tokens) {
        if (antiprompt_tokens.empty() || generated_tokens.size() < antiprompt_tokens.size()) {
            return false;
        }
        return std::equal(
            generated_tokens.end() - antiprompt_tokens.size(),
            generated_tokens.end(),
            antiprompt_tokens.begin()
        );
    }

    bool load_media(const std::string & fname) {
        mtmd::bitmap bmp(mtmd_helper_bitmap_init_from_file(ctx_vision.get(), fname.c_str()));
        if (!bmp.ptr) {
            return false;
        }
        bitmaps.entries.push_back(std::move(bmp));
        return true;
    }
};

static int generate_response(mtmd_cli_context & ctx, int n_predict) {
    llama_tokens generated_tokens;
    const auto t_decode_0 = std::chrono::steady_clock::now();
    for (int i = 0; i < n_predict; i++) {
        if (i > n_predict || !g_is_generating || g_is_interrupted) {
            LOG("\n");
            break;
        }

        llama_token token_id;
        if (i == 0 && ctx.first_token >= 0) {
            token_id = ctx.first_token; // [QVAC-21372 item 2] pre-sampled from the CPU prefill ctx
        } else {
            token_id = common_sampler_sample(ctx.smpl, ctx.lctx, -1);
        }
        generated_tokens.push_back(token_id);
        common_sampler_accept(ctx.smpl, token_id, true);

        if (llama_vocab_is_eog(ctx.vocab, token_id) || ctx.check_antiprompt(generated_tokens)) {
            LOG("\n");
            break; // end of generation
        }

        LOG("%s", common_token_to_piece(ctx.lctx, token_id).c_str());
        fflush(stdout);

        if (g_is_interrupted) {
            LOG("\n");
            break;
        }

        // eval the token
        common_batch_clear(ctx.batch);
        common_batch_add(ctx.batch, token_id, ctx.n_past++, {0}, true);
        if (llama_decode(ctx.lctx, ctx.batch)) {
            LOG_ERR("failed to decode token\n");
            return 1;
        }
    }

    const auto t_decode_1 = std::chrono::steady_clock::now();
    const double decode_ms = std::chrono::duration<double, std::milli>(t_decode_1 - t_decode_0).count();
    const int n_dec = (int) generated_tokens.size();
    // [qvac A1] decode loop (always GPU-pinned). t/s excludes the first sample.
    fprintf(stderr, "QVAC_TIMING decode_ms=%.2f decode_tokens=%d decode_tps=%.2f\n",
            decode_ms, n_dec, n_dec > 1 ? (n_dec - 1) * 1000.0 / decode_ms : 0.0);

    std::string generated_text = common_detokenize(ctx.lctx, generated_tokens);
    common_chat_msg msg;
    msg.role    = "assistant";
    msg.content = generated_text;
    ctx.chat_history.push_back(std::move(msg));

    return 0;
}

static std::string chat_add_and_format(mtmd_cli_context & ctx, common_chat_msg & new_msg) {
    LOG_DBG("chat_add_and_format: new_msg.role='%s', new_msg.content='%s'\n",
        new_msg.role.c_str(), new_msg.content.c_str());
    auto formatted = common_chat_format_single(ctx.tmpls.get(), ctx.chat_history,
        new_msg, new_msg.role == "user",
        ctx.use_jinja);
    ctx.chat_history.push_back(new_msg);
    return formatted;
}

static int eval_message(mtmd_cli_context & ctx, common_chat_msg & msg) {
    bool add_bos = ctx.chat_history.empty();
    auto formatted_chat = chat_add_and_format(ctx, msg);
    LOG_DBG("formatted_chat.prompt: %s\n", formatted_chat.c_str());

    mtmd_input_text text;
    text.text          = formatted_chat.c_str();
    text.add_special   = add_bos;
    text.parse_special = true;

    if (g_is_interrupted) return 0;

    mtmd::input_chunks chunks(mtmd_input_chunks_init());
    auto bitmaps_c_ptr = ctx.bitmaps.c_ptr();
    int32_t res = mtmd_tokenize(ctx.ctx_vision.get(),
                        chunks.ptr.get(), // output
                        &text, // text
                        bitmaps_c_ptr.data(),
                        bitmaps_c_ptr.size());
    if (res != 0) {
        LOG_ERR("Unable to tokenize prompt, res = %d\n", res);
        return 1;
    }

    ctx.bitmaps.entries.clear();

    llama_pos new_n_past;
    // [QVAC-21372 item 2] in two-ctx mode the prefill (vision-encode injects into the LLM
    // KV cache + prompt eval) runs on the CPU instance; otherwise on the single GPU ctx.
    llama_context * prefill_ctx = ctx.two_ctx ? ctx.lctx_cpu : ctx.lctx;
    const auto t_prefill_0 = std::chrono::steady_clock::now();
    if (mtmd_helper_eval_chunks(ctx.ctx_vision.get(),
                prefill_ctx, // lctx (CPU instance when two_ctx)
                chunks.ptr.get(), // chunks
                ctx.n_past, // n_past
                0, // seq_id
                ctx.n_batch, // n_batch
                true, // logits_last
                &new_n_past)) {
        LOG_ERR("Unable to eval prompt\n");
        return 1;
    }
    const auto t_prefill_1 = std::chrono::steady_clock::now();
    const double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_1 - t_prefill_0).count();
    // [qvac A1] TTFT-relevant section (vision encode + prompt prefill).
    fprintf(stderr, "QVAC_TIMING prefill_ttft_ms=%.2f prompt_tokens=%d\n",
            prefill_ms, (int) new_n_past);

    ctx.n_past = new_n_past;

    // [QVAC-21372 item 2] hand the KV cache from the CPU prefill ctx to the GPU decode ctx.
    // The seq-state blob is backend-agnostic; only the KV cache transfers (not the last-token
    // logits), so we also sample the first token here from the CPU ctx's prefill logits.
    if (ctx.two_ctx) {
        const auto t_x0 = std::chrono::steady_clock::now();
        size_t sz  = llama_state_seq_get_size(ctx.lctx_cpu, 0);
        std::vector<uint8_t> buf(sz);
        size_t got = llama_state_seq_get_data(ctx.lctx_cpu, buf.data(), buf.size(), 0);
        size_t set = llama_state_seq_set_data(ctx.lctx, buf.data(), got, 0);
        if (set == 0) {
            // SWA / partial KV cache (e.g. Gemma4 sliding-window): retry partial-only.
            sz  = llama_state_seq_get_size_ext(ctx.lctx_cpu, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            buf.assign(sz, 0);
            got = llama_state_seq_get_data_ext(ctx.lctx_cpu, buf.data(), buf.size(), 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
            set = llama_state_seq_set_data_ext(ctx.lctx, buf.data(), got, 0, LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY);
        }
        const auto t_x1 = std::chrono::steady_clock::now();
        const double transfer_ms = std::chrono::duration<double, std::milli>(t_x1 - t_x0).count();
        if (set == 0) {
            LOG_ERR("[QVAC_TWO_CTX] KV state transfer failed (set_data returned 0)\n");
            return 1;
        }
        // first decode token: sample from the CPU prefill logits (GPU ctx has none post-transfer)
        ctx.first_token = common_sampler_sample(ctx.smpl, ctx.lctx_cpu, -1);
        fprintf(stderr, "QVAC_TIMING transfer_ms=%.2f kv_bytes=%zu\n", transfer_ms, got);
    }

    LOG("\n");

    return 0;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    ggml_time_init();

    common_params params;

    common_init();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_MTMD, show_additional_info)) {
        return 1;
    }

    mtmd_helper_log_set(common_log_default_callback, nullptr);

    if (params.mmproj.path.empty()) {
        show_additional_info(argc, argv);
        LOG_ERR("ERR: Missing --mmproj argument\n");
        return 1;
    }

    ggml_backend_load_all();

    mtmd_cli_context ctx(params);
    LOG_INF("%s: loading model: %s\n", __func__, params.model.path.c_str());

    bool is_single_turn = !params.prompt.empty() && !params.image.empty();

    int n_predict = params.n_predict < 0 ? INT_MAX : params.n_predict;

    // Ctrl+C handling
    {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (sigint_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    if (g_is_interrupted) return 130;

    auto eval_system_prompt_if_present = [&] {
        if (params.system_prompt.empty()) {
            return 0;
        }

        common_chat_msg msg;
        msg.role = "system";
        msg.content = params.system_prompt;
        return eval_message(ctx, msg);
    };

    LOG_WRN("WARN: This is an experimental CLI for testing multimodal capability.\n");
    LOG_WRN("      For normal use cases, please use the standard llama-cli\n");

    if (eval_system_prompt_if_present()) {
        return 1;
    }

    if (is_single_turn) {
        g_is_generating = true;
        if (params.prompt.find(mtmd_default_marker()) == std::string::npos) {
            for (size_t i = 0; i < params.image.size(); i++) {
                // most models require the marker before each image
                // ref: https://github.com/ggml-org/llama.cpp/pull/17616
                params.prompt = mtmd_default_marker() + params.prompt;
            }
        }

        common_chat_msg msg;
        msg.role = "user";
        msg.content = params.prompt;
        for (const auto & image : params.image) {
            if (!ctx.load_media(image)) {
                return 1; // error is already printed by libmtmd
            }
        }
        if (eval_message(ctx, msg)) {
            return 1;
        }
        if (!g_is_interrupted && generate_response(ctx, n_predict)) {
            return 1;
        }

    } else {
        LOG("\n Running in chat mode, available commands:");
        if (mtmd_support_vision(ctx.ctx_vision.get())) {
            LOG("\n   /image <path>    load an image");
        }
        if (mtmd_support_audio(ctx.ctx_vision.get())) {
            LOG("\n   /audio <path>    load an audio");
        }
        LOG("\n   /clear           clear the chat history");
        LOG("\n   /quit or /exit   exit the program");
        LOG("\n");

        std::string content;

        while (!g_is_interrupted) {
            g_is_generating = false;
            LOG("\n> ");
            console::set_display(DISPLAY_TYPE_USER_INPUT);
            std::string line;
            console::readline(line, false);
            if (g_is_interrupted) break;
            console::set_display(DISPLAY_TYPE_RESET);
            line = string_strip(line);
            if (line.empty()) {
                continue;
            }
            if (line == "/quit" || line == "/exit") {
                break;
            }
            if (line == "/clear") {
                ctx.n_past = 0;
                ctx.chat_history.clear();
                llama_memory_clear(llama_get_memory(ctx.lctx), true);
                if (eval_system_prompt_if_present()) {
                    return 1;
                }
                LOG("Chat history cleared\n\n");
                continue;
            }
            g_is_generating = true;
            bool is_image = line == "/image" || line.find("/image ") == 0;
            bool is_audio = line == "/audio" || line.find("/audio ") == 0;
            if (is_image || is_audio) {
                if (line.size() < 8) {
                    LOG_ERR("ERR: Missing media filename\n");
                    continue;
                }
                std::string media_path = line.substr(7);
                if (ctx.load_media(media_path)) {
                    LOG("%s %s loaded\n", media_path.c_str(), is_image ? "image" : "audio");
                    content += mtmd_default_marker();
                }
                // else, error is already printed by libmtmd
                continue;
            } else {
                content += line;
            }
            common_chat_msg msg;
            msg.role = "user";
            msg.content = content;
            int ret = eval_message(ctx, msg);
            if (ret) {
                return 1;
            }
            if (g_is_interrupted) break;
            if (generate_response(ctx, n_predict)) {
                return 1;
            }
            content.clear();
        }
    }
    if (g_is_interrupted) LOG("\nInterrupted by user\n");
    LOG("\n\n");
    llama_perf_context_print(ctx.lctx);
    return g_is_interrupted ? 130 : 0;
}
