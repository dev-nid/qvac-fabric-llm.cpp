#ifndef LLAMA_H
#define LLAMA_H

#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-opt.h"
#include "gguf.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdbool.h>

#ifdef LLAMA_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef LLAMA_BUILD
#            define LLAMA_API __declspec(dllexport)
#        else
#            define LLAMA_API __declspec(dllimport)
#        endif
#    else
#        define LLAMA_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define LLAMA_API
#endif

#ifdef __GNUC__
#    define DEPRECATED(func, hint) func __attribute__((deprecated(hint)))
#elif defined(_MSC_VER)
#    define DEPRECATED(func, hint) __declspec(deprecated(hint)) func
#else
#    define DEPRECATED(func, hint) func
#endif

#define LLAMA_DEFAULT_SEED 0xFFFFFFFF

#define LLAMA_TOKEN_NULL -1

#define LLAMA_FILE_MAGIC_GGLA 0x67676c61u // 'ggla'
#define LLAMA_FILE_MAGIC_GGSN 0x6767736eu // 'ggsn'
#define LLAMA_FILE_MAGIC_GGSQ 0x67677371u // 'ggsq'

#define LLAMA_SESSION_MAGIC   LLAMA_FILE_MAGIC_GGSN
#define LLAMA_SESSION_VERSION 9

#define LLAMA_STATE_SEQ_MAGIC   LLAMA_FILE_MAGIC_GGSQ
#define LLAMA_STATE_SEQ_VERSION 2

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct llama_vocab;
    struct llama_model;
    struct llama_context;
    struct llama_sampler;

    typedef struct llama_memory_i * llama_memory_t;

    typedef int32_t llama_pos;
    typedef int32_t llama_token;
    typedef int32_t llama_seq_id;

    enum llama_vocab_type {
        LLAMA_VOCAB_TYPE_NONE   = 0, // For models without vocab
        LLAMA_VOCAB_TYPE_SPM    = 1, // LLaMA tokenizer based on byte-level BPE with byte fallback
        LLAMA_VOCAB_TYPE_BPE    = 2, // GPT-2 tokenizer based on byte-level BPE
        LLAMA_VOCAB_TYPE_WPM    = 3, // BERT tokenizer based on WordPiece
        LLAMA_VOCAB_TYPE_UGM    = 4, // T5 tokenizer based on Unigram
        LLAMA_VOCAB_TYPE_RWKV   = 5, // RWKV tokenizer based on greedy tokenization
        LLAMA_VOCAB_TYPE_PLAMO2 = 6, // PLaMo-2 tokenizer based on Aho-Corasick with dynamic programming
    };

    enum llama_rope_type {
        LLAMA_ROPE_TYPE_NONE   = -1,
        LLAMA_ROPE_TYPE_NORM   = 0,
        LLAMA_ROPE_TYPE_NEOX   = GGML_ROPE_TYPE_NEOX,
        LLAMA_ROPE_TYPE_MROPE  = GGML_ROPE_TYPE_MROPE,
        LLAMA_ROPE_TYPE_IMROPE = GGML_ROPE_TYPE_IMROPE,
        LLAMA_ROPE_TYPE_VISION = GGML_ROPE_TYPE_VISION,
    };

    enum llama_token_type { //TODO: remove, required until per token attributes are available from GGUF file
        LLAMA_TOKEN_TYPE_UNDEFINED    = 0,
        LLAMA_TOKEN_TYPE_NORMAL       = 1,
        LLAMA_TOKEN_TYPE_UNKNOWN      = 2,
        LLAMA_TOKEN_TYPE_CONTROL      = 3,
        LLAMA_TOKEN_TYPE_USER_DEFINED = 4,
        LLAMA_TOKEN_TYPE_UNUSED       = 5,
        LLAMA_TOKEN_TYPE_BYTE         = 6,
    };

    enum llama_token_attr {
        LLAMA_TOKEN_ATTR_UNDEFINED    = 0,
        LLAMA_TOKEN_ATTR_UNKNOWN      = 1 << 0,
        LLAMA_TOKEN_ATTR_UNUSED       = 1 << 1,
        LLAMA_TOKEN_ATTR_NORMAL       = 1 << 2,
        LLAMA_TOKEN_ATTR_CONTROL      = 1 << 3,  // SPECIAL?
        LLAMA_TOKEN_ATTR_USER_DEFINED = 1 << 4,
        LLAMA_TOKEN_ATTR_BYTE         = 1 << 5,
        LLAMA_TOKEN_ATTR_NORMALIZED   = 1 << 6,
        LLAMA_TOKEN_ATTR_LSTRIP       = 1 << 7,
        LLAMA_TOKEN_ATTR_RSTRIP       = 1 << 8,
        LLAMA_TOKEN_ATTR_SINGLE_WORD  = 1 << 9,
    };

    // model file types
    enum llama_ftype {
        LLAMA_FTYPE_ALL_F32              = 0,
        LLAMA_FTYPE_MOSTLY_F16           = 1,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_0          = 2,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_1          = 3,  // except 1d tensors
        // LLAMA_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4,  // tok_embeddings.weight and output.weight are F16
        // LLAMA_FTYPE_MOSTLY_Q4_2       = 5,  // support has been removed
        // LLAMA_FTYPE_MOSTLY_Q4_3       = 6,  // support has been removed
        LLAMA_FTYPE_MOSTLY_Q8_0          = 7,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_0          = 8,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_1          = 9,  // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K          = 10, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_S        = 11, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_M        = 12, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q3_K_L        = 13, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_S        = 14, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q4_K_M        = 15, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_S        = 16, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q5_K_M        = 17, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q6_K          = 18, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XXS       = 19, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_XS        = 20, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q2_K_S        = 21, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XS        = 22, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_XXS       = 23, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_S         = 24, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_NL        = 25, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_S         = 26, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ3_M         = 27, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_S         = 28, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ2_M         = 29, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ4_XS        = 30, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_IQ1_M         = 31, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_BF16          = 32, // except 1d tensors
        //LLAMA_FTYPE_MOSTLY_Q4_0_4_4      = 33, // removed from gguf files, use Q4_0 and runtime repack
        //LLAMA_FTYPE_MOSTLY_Q4_0_4_8      = 34, // removed from gguf files, use Q4_0 and runtime repack
        //LLAMA_FTYPE_MOSTLY_Q4_0_8_8      = 35, // removed from gguf files, use Q4_0 and runtime repack
        LLAMA_FTYPE_MOSTLY_TQ1_0         = 36, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_TQ2_0         = 37, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_MXFP4_MOE     = 38, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_NVFP4         = 39, // except 1d tensors
        LLAMA_FTYPE_MOSTLY_Q1_0          = 40, // except 1d tensors

        LLAMA_FTYPE_GUESSED = 1024, // not specified in the model file
    };

    enum llama_rope_scaling_type {
        LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1,
        LLAMA_ROPE_SCALING_TYPE_NONE        = 0,
        LLAMA_ROPE_SCALING_TYPE_LINEAR      = 1,
        LLAMA_ROPE_SCALING_TYPE_YARN        = 2,
        LLAMA_ROPE_SCALING_TYPE_LONGROPE    = 3,
        LLAMA_ROPE_SCALING_TYPE_MAX_VALUE   = LLAMA_ROPE_SCALING_TYPE_LONGROPE,
    };

    enum llama_pooling_type {
        LLAMA_POOLING_TYPE_UNSPECIFIED = -1,
        LLAMA_POOLING_TYPE_NONE = 0,
        LLAMA_POOLING_TYPE_MEAN = 1,
        LLAMA_POOLING_TYPE_CLS  = 2,
        LLAMA_POOLING_TYPE_LAST = 3,
        LLAMA_POOLING_TYPE_RANK = 4, // used by reranking models to attach the classification head to the graph
    };

    enum llama_attention_type {
        LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1,
        LLAMA_ATTENTION_TYPE_CAUSAL      = 0,
        LLAMA_ATTENTION_TYPE_NON_CAUSAL  = 1,
    };

    enum llama_flash_attn_type {
        LLAMA_FLASH_ATTN_TYPE_AUTO     = -1,
        LLAMA_FLASH_ATTN_TYPE_DISABLED = 0,
        LLAMA_FLASH_ATTN_TYPE_ENABLED  = 1,
    };

    LLAMA_API const char * llama_flash_attn_type_name(enum llama_flash_attn_type flash_attn_type);

    enum llama_split_mode {
        LLAMA_SPLIT_MODE_NONE   = 0, // single GPU
        LLAMA_SPLIT_MODE_LAYER  = 1, // split layers and KV across GPUs
        LLAMA_SPLIT_MODE_ROW    = 2, // split layers and KV across GPUs, use tensor parallelism if supported
        LLAMA_SPLIT_MODE_TENSOR = 3,
    };

    // TODO: simplify (https://github.com/ggml-org/llama.cpp/pull/9294#pullrequestreview-2286561979)
    typedef struct llama_token_data {
        llama_token id; // token id
        float logit;    // log-odds of the token
        float p;        // probability of the token
    } llama_token_data;

    typedef struct llama_token_data_array {
        // TODO: consider SoA
        // NOTE: this pointer can be modified by the samplers
        llama_token_data * data;
        size_t size;
        int64_t selected; // this is the index in the data array (i.e. not the token id)
        bool sorted;      // note: do not assume the data is sorted - always check this flag
    } llama_token_data_array;

    typedef bool (*llama_progress_callback)(float progress, void * user_data);

    // Input data for llama_encode/llama_decode
    // A llama_batch object can contain input about one or many sequences
    // The provided arrays (i.e. token, embd, pos, etc.) must have size of n_tokens
    //
    // - token  : the token ids of the input (used when embd is NULL)
    // - embd   : token embeddings (i.e. float vector of size n_embd) (used when token is NULL)
    // - pos    : the positions of the respective token in the sequence
    //            (if set to NULL, the token position will be tracked automatically by llama_encode/llama_decode)
    // - seq_id : the sequence to which the respective token belongs
    //            (if set to NULL, the sequence ID will be assumed to be 0)
    // - logits : if zero, the logits (and/or the embeddings) for the respective token will not be output
    //            (if set to NULL:
    //               - if embeddings: all tokens are output
    //               - if not:        only the last token is output
    //            )
    //
    typedef struct llama_batch {
        int32_t n_tokens;

        llama_token  *  token;
        float        *  embd;
        llama_pos    *  pos;
        int32_t      *  n_seq_id;
        llama_seq_id ** seq_id;
        int8_t       *  logits;   // TODO: rename this to "output"
    } llama_batch;

    enum llama_model_kv_override_type {
        LLAMA_KV_OVERRIDE_TYPE_INT,
        LLAMA_KV_OVERRIDE_TYPE_FLOAT,
        LLAMA_KV_OVERRIDE_TYPE_BOOL,
        LLAMA_KV_OVERRIDE_TYPE_STR,
    };

    enum llama_model_meta_key {
        LLAMA_MODEL_META_KEY_SAMPLING_SEQUENCE,
        LLAMA_MODEL_META_KEY_SAMPLING_TOP_K,
        LLAMA_MODEL_META_KEY_SAMPLING_TOP_P,
        LLAMA_MODEL_META_KEY_SAMPLING_MIN_P,
        LLAMA_MODEL_META_KEY_SAMPLING_XTC_PROBABILITY,
        LLAMA_MODEL_META_KEY_SAMPLING_XTC_THRESHOLD,
        LLAMA_MODEL_META_KEY_SAMPLING_TEMP,
        LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_LAST_N,
        LLAMA_MODEL_META_KEY_SAMPLING_PENALTY_REPEAT,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_TAU,
        LLAMA_MODEL_META_KEY_SAMPLING_MIROSTAT_ETA,
    };

    struct llama_model_kv_override {
        enum llama_model_kv_override_type tag;

        char key[128];

        union {
            int64_t val_i64;
            double  val_f64;
            bool    val_bool;
            char    val_str[128];
        };
    };

    struct llama_model_tensor_buft_override {
        const char * pattern;
        ggml_backend_buffer_type_t buft;
    };

    struct llama_model_params {
        // NULL-terminated list of devices to use for offloading (if NULL, all available devices are used)
        ggml_backend_dev_t * devices;

        // NULL-terminated list of buffer types to use for tensors that match a pattern
        const struct llama_model_tensor_buft_override * tensor_buft_overrides;

        int32_t n_gpu_layers; // number of layers to store in VRAM, a negative value means all layers
        enum llama_split_mode split_mode; // how to split the model across multiple GPUs

        // the GPU that is used for the entire model when split_mode is LLAMA_SPLIT_MODE_NONE
        int32_t main_gpu;

        // proportion of the model (layers or rows) to offload to each GPU, size: llama_max_devices()
        const float * tensor_split;

        // Called with a progress value between 0.0 and 1.0. Pass NULL to disable.
        // If the provided progress_callback returns true, model loading continues.
        // If it returns false, model loading is immediately aborted.
        llama_progress_callback progress_callback;

        // context pointer passed to the progress callback
        void * progress_callback_user_data;

        // override key-value pairs of the model meta data
        const struct llama_model_kv_override * kv_overrides;

        // Keep the booleans together to avoid misalignment during copy-by-value.
        bool vocab_only;      // only load the vocabulary, no weights
        bool use_mmap;        // use mmap if possible
        bool use_direct_io;   // use direct io, takes precedence over use_mmap when supported
        bool use_mlock;       // force system to keep model in RAM
        bool check_tensors;   // validate model tensor data
        bool use_extra_bufts; // use extra buffer types (used for weight repacking)
        bool no_host;         // bypass host buffer allowing extra buffers to be used
        bool no_alloc;        // only load metadata and simulate memory allocations
    };

    struct llama_sampler_seq_config {
        llama_seq_id           seq_id;
        struct llama_sampler * sampler;
    };

    // NOTE: changing the default values of parameters marked as [EXPERIMENTAL] may cause crashes or incorrect results in certain configurations
    //       https://github.com/ggml-org/llama.cpp/pull/7544
    struct llama_context_params {
        uint32_t n_ctx;             // text context, 0 = from model
        uint32_t n_batch;           // logical maximum batch size that can be submitted to llama_decode
        uint32_t n_ubatch;          // physical maximum batch size
        uint32_t n_seq_max;         // max number of sequences (i.e. distinct states for recurrent models)
        int32_t  n_threads;         // number of threads to use for generation
        int32_t  n_threads_batch;   // number of threads to use for batch processing

        enum llama_rope_scaling_type rope_scaling_type; // RoPE scaling type, from `enum llama_rope_scaling_type`
        enum llama_pooling_type      pooling_type;      // whether to pool (sum) embedding results by sequence id
        enum llama_attention_type    attention_type;    // attention type to use for embeddings
        enum llama_flash_attn_type   flash_attn_type;   // when to enable Flash Attention

        // ref: https://github.com/ggml-org/llama.cpp/pull/2054
        float    rope_freq_base;   // RoPE base frequency, 0 = from model
        float    rope_freq_scale;  // RoPE frequency scaling factor, 0 = from model
        float    yarn_ext_factor;  // YaRN extrapolation mix factor, negative = from model
        float    yarn_attn_factor; // YaRN magnitude scaling factor
        float    yarn_beta_fast;   // YaRN low correction dim
        float    yarn_beta_slow;   // YaRN high correction dim
        uint32_t yarn_orig_ctx;    // YaRN original context size
        float    defrag_thold;     // [DEPRECATED] defragment the KV cache if holes/size > thold, <= 0 disabled (default)

        ggml_backend_sched_eval_callback cb_eval;
        void * cb_eval_user_data;

        enum ggml_type type_k; // data type for K cache [EXPERIMENTAL]
        enum ggml_type type_v; // data type for V cache [EXPERIMENTAL]

        // Abort callback
        // if it returns true, execution of llama_decode() will be aborted
        // currently works only with CPU execution
        ggml_abort_callback abort_callback;
        void *              abort_callback_data;

        // Keep the booleans together and at the end of the struct to avoid misalignment during copy-by-value.
        bool embeddings;  // if true, extract embeddings (together with logits)
        bool offload_kqv; // offload the KQV ops (including the KV cache) to GPU
        bool no_perf;     // measure performance timings
        bool op_offload;  // offload host tensor operations to device
        bool swa_full;    // use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
                          // NOTE: setting to false when n_seq_max > 1 can cause bad performance in some cases
                          //       ref: https://github.com/ggml-org/llama.cpp/pull/13845#issuecomment-2924800573
        bool kv_unified;  // use a unified buffer across the input sequences when computing the attention
                          // try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix
                          // ref: https://github.com/ggml-org/llama.cpp/pull/14363

        int32_t dflash_max_ctx;  // sliding-window cap on the DFlash drafter's per-layer K/V side store.
                                 // -1 = auto-scale (the default): cap = clamp(cparams.n_ctx_seq / 4, 512, 1024),
                                 //      then capped to n_ctx_seq.
                                 //  0 = uncapped: use the full cparams.n_ctx_seq.
                                 // >0 = explicit cap: capped to cparams.n_ctx_seq.
                                 // Ignored for non-DFlash drafts.

        uint32_t dflash_topk;    // number of top-K candidate tokens the DFlash drafter emits per output position.
                                 // 1 = chain mode (default), >=2 = tree mode.
                                 // Ignored for non-DFlash drafts.

        bool dflash_emit_logits; // When true, the DFlash draft emits full-vocab logits so the host can
                                 // compute per-position log-probs for best-first tree expansion.
                                 // Default false. Ignored for non-DFlash drafts.
                                 // Auto-enabled by --dflash-tree-best-first.

        // Inline DFlash encoder. When set on a TARGET context, the target's
        // graph builder runs the encoder K/V projection inline after the
        // final captured layer instead of calling a separate decode on the
        // draft context. Companion sizing fields are required so
        // graph_reserve can size the inline ops before
        // llama_dflash_bind_encoder() is called. Ignored for DFlash drafts
        // (these fields are target-only).
        bool     dflash_inline_encoder;
        uint32_t dflash_inline_n_embd_dft;        // draft n_embd
        uint32_t dflash_inline_n_head_kv_dft;     // draft n_head_kv
        uint32_t dflash_inline_n_embd_head_dft;   // draft n_embd_head_v
        uint32_t dflash_inline_n_target_layers;   // len(target_layer_ids)

        // DFlash GatedDeltaNet history kernel. When set on a TARGET context
        // whose architecture has GDN layers (e.g. Qwen3.5), the GDN op writes
        // per-token recurrent states to a persistent per-layer buffer instead
        // of only the final state. The speculative driver consumes those
        // buffers after sampling to select state[K-1] (K = accepted count)
        // and avoid the checkpoint+re-verify round-trip required by
        // COMMON_CONTEXT_SEQ_RM_TYPE_FULL. CUDA-only. Ignored for DFlash
        // drafts (target-only flag); setting it on an LLM_ARCH_DFLASH context
        // throws at init.
        bool     dflash_gdn_history;

        // GDN history persistent buffer dtype. When true,
        // dflash.gdn_history[il] is allocated as GGML_TYPE_F16 instead of
        // GGML_TYPE_F32 (halves the per-layer footprint). Ignored when
        // dflash_gdn_history is false. Default false.
        bool     dflash_gdn_history_f16;

        // [EXPERIMENTAL]
        // backend sampler chain configuration (make sure the caller keeps the sampler chains alive)
        // note: the samplers must be sampler chains (i.e. use llama_sampler_chain_init)
        struct llama_sampler_seq_config * samplers;
        size_t                            n_samplers;
    };

    struct llama_model_tensor_override {
        const char * pattern;
        enum ggml_type type;
    };

    struct llama_model_imatrix_data {
        const char * name;
        const float * data;
        size_t size;
    };

    // model quantization parameters
    typedef struct llama_model_quantize_params {
        int32_t nthread;                                            // number of threads to use for quantizing, if <=0 will use std::thread::hardware_concurrency()
        enum llama_ftype ftype;                                     // quantize to this llama_ftype
        enum ggml_type output_tensor_type;                          // output tensor type
        enum ggml_type token_embedding_type;                        // token embeddings tensor type
        bool allow_requantize;                                      // allow quantizing non-f32/f16 tensors
        bool quantize_output_tensor;                                // quantize output.weight
        bool only_copy;                                             // only copy tensors - ftype, allow_requantize and quantize_output_tensor are ignored
        bool pure;                                                  // quantize all tensors to the default type
        bool keep_split;                                            // quantize to the same number of shards
        bool dry_run;                                               // calculate and show the final quantization size without performing quantization
        const struct llama_model_imatrix_data * imatrix;            // pointer to importance matrix data
        const struct llama_model_kv_override * kv_overrides;        // pointer to kv overrides
        const struct llama_model_tensor_override * tt_overrides;    // pointer to tensor overrides
        const int32_t * prune_layers;                               // pointer to layer indices to prune
    } llama_model_quantize_params;

    typedef struct llama_logit_bias {
        llama_token token;
        float bias;
    } llama_logit_bias;

    typedef struct llama_sampler_chain_params {
        bool no_perf; // whether to measure performance timings
    } llama_sampler_chain_params;

    // used in chat template
    typedef struct llama_chat_message {
        const char * role;
        const char * content;
    } llama_chat_message;

    // lora adapter
    struct llama_adapter_lora;

    // Helpers for getting default parameters
    // TODO: update API to start accepting pointers to params structs (https://github.com/ggml-org/llama.cpp/discussions/9172)
    LLAMA_API struct llama_model_params          llama_model_default_params(void);
    LLAMA_API struct llama_context_params        llama_context_default_params(void);
    LLAMA_API struct llama_sampler_chain_params  llama_sampler_chain_default_params(void);
    LLAMA_API struct llama_model_quantize_params llama_model_quantize_default_params(void);

    // Initialize the llama + ggml backend
    // If numa is true, use NUMA optimizations
    // Call once at the start of the program
    LLAMA_API void llama_backend_init(void);

    // Call once at the end of the program - currently only used for MPI
    LLAMA_API void llama_backend_free(void);

    //optional:
    LLAMA_API void llama_numa_init(enum ggml_numa_strategy numa);

    // Optional: an auto threadpool gets created in ggml if not passed explicitly
    LLAMA_API void llama_attach_threadpool(
            struct llama_context * ctx,
               ggml_threadpool_t   threadpool,
               ggml_threadpool_t   threadpool_batch);

    LLAMA_API void llama_detach_threadpool(struct llama_context * ctx);

    typedef void (*llama_model_set_tensor_data_t)(struct ggml_tensor * tensor, void * userdata);

    // Create a new model from GGUF metadata as well as a function to set the tensor data
    //   - tensors are created as GGML_TYPE_F32 by default,
    //     override by adding a tensor with the same name but a different name to the context
    LLAMA_API struct llama_model * llama_model_init_from_user(
                    struct gguf_context * metadata,
          llama_model_set_tensor_data_t   set_tensor_data,    // function to initialize tensor data with
                                   void * set_tensor_data_ud, // userdata for function
              struct llama_model_params   params);

    DEPRECATED(LLAMA_API struct llama_model * llama_load_model_from_file(
                             const char * path_model,
              struct llama_model_params   params),
            "use llama_model_load_from_file instead");

    // Load a model from a file
    // If the file is split into multiple parts, the file name must follow this pattern: <name>-%05d-of-%05d.gguf
    // If the split file name does not follow this pattern, use llama_model_load_from_splits
    LLAMA_API struct llama_model * llama_model_load_from_file(
                             const char * path_model,
              struct llama_model_params   params);

    // Load a model from an open FILE pointer
    LLAMA_API struct llama_model * llama_model_load_from_file_ptr(
                                   FILE * file,
              struct llama_model_params   params);

    // Load a model from multiple splits (support custom naming scheme)
    // The paths must be in the correct order
    LLAMA_API struct llama_model * llama_model_load_from_splits(
                             const char ** paths,
                                 size_t    n_paths,
              struct llama_model_params    params);

    LLAMA_API void llama_model_save_to_file(
            const struct llama_model * model,
                        const char * path_model);

    DEPRECATED(LLAMA_API void llama_free_model(struct llama_model * model),
            "use llama_model_free instead");

    LLAMA_API void llama_model_free(struct llama_model * model);

    LLAMA_API struct llama_context * llama_init_from_model(
                     struct llama_model * model,
            struct llama_context_params   params);

    DEPRECATED(LLAMA_API struct llama_context * llama_new_context_with_model(
                     struct llama_model * model,
            struct llama_context_params   params),
            "use llama_init_from_model instead");

    // Frees all allocated memory
    LLAMA_API void llama_free(struct llama_context * ctx);

    LLAMA_API int64_t llama_time_us(void);

    LLAMA_API size_t llama_max_devices(void);
    LLAMA_API size_t llama_max_parallel_sequences(void);
    LLAMA_API size_t llama_max_tensor_buft_overrides(void);

    LLAMA_API bool llama_supports_mmap       (void);
    LLAMA_API bool llama_supports_mlock      (void);
    LLAMA_API bool llama_supports_gpu_offload(void);
    LLAMA_API bool llama_supports_rpc        (void);

    // NOTE: After creating a llama_context, it is recommended to query the actual values using these functions
    //       In some cases the requested values via llama_context_params may differ from the actual values used by the context
    //       ref: https://github.com/ggml-org/llama.cpp/pull/17046#discussion_r2503085732
    LLAMA_API uint32_t llama_n_ctx      (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ctx_seq  (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_batch    (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_ubatch   (const struct llama_context * ctx);
    LLAMA_API uint32_t llama_n_seq_max  (const struct llama_context * ctx);

    DEPRECATED(LLAMA_API int32_t llama_n_ctx_train(const struct llama_model * model), "use llama_model_n_ctx_train instead");
    DEPRECATED(LLAMA_API int32_t llama_n_embd     (const struct llama_model * model), "use llama_model_n_embd instead");
    DEPRECATED(LLAMA_API int32_t llama_n_layer    (const struct llama_model * model), "use llama_model_n_layer instead");
    DEPRECATED(LLAMA_API int32_t llama_n_head     (const struct llama_model * model), "use llama_model_n_head instead");

    DEPRECATED(LLAMA_API int32_t llama_n_vocab    (const struct llama_vocab * vocab), "use llama_vocab_n_tokens instead");

    LLAMA_API const struct llama_model * llama_get_model   (const struct llama_context * ctx);
    LLAMA_API           llama_memory_t   llama_get_memory  (const struct llama_context * ctx);
    LLAMA_API  enum llama_pooling_type   llama_pooling_type(const struct llama_context * ctx); // TODO: rename to llama_get_pooling_type

    LLAMA_API const struct llama_vocab * llama_model_get_vocab(const struct llama_model * model);
    LLAMA_API enum llama_rope_type       llama_model_rope_type(const struct llama_model * model);

    LLAMA_API int32_t llama_model_n_ctx_train(const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd     (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd_inp (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_embd_out (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_layer    (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_head     (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_head_kv  (const struct llama_model * model);
    LLAMA_API int32_t llama_model_n_swa      (const struct llama_model * model);

    // Get the model's RoPE frequency scaling factor
    LLAMA_API float llama_model_rope_freq_scale_train(const struct llama_model * model);

    // Returns the number of classifier outputs (only valid for classifier models)
    // Undefined behavior for non-classifier models
    LLAMA_API uint32_t llama_model_n_cls_out(const struct llama_model * model);

    // Returns label of classifier output by index (<n_cls_out). Returns nullptr if no label provided
    LLAMA_API const char * llama_model_cls_label(const struct llama_model * model, uint32_t i);

    LLAMA_API enum llama_vocab_type llama_vocab_type(const struct llama_vocab * vocab);

    LLAMA_API int32_t llama_vocab_n_tokens(const struct llama_vocab * vocab);

    // Functions to access the model's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLAMA_API int32_t llama_model_meta_val_str(const struct llama_model * model, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLAMA_API int32_t llama_model_meta_count(const struct llama_model * model);

    // Get sampling metadata key name. Returns nullptr if the key is invalid
    LLAMA_API const char * llama_model_meta_key_str(enum llama_model_meta_key key);

    // Get metadata key name by index
    LLAMA_API int32_t llama_model_meta_key_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLAMA_API int32_t llama_model_meta_val_str_by_index(const struct llama_model * model, int32_t i, char * buf, size_t buf_size);

    // Get a string describing the model type
    LLAMA_API int32_t llama_model_desc(const struct llama_model * model, char * buf, size_t buf_size);

    // Returns the total size of all the tensors in the model in bytes
    LLAMA_API uint64_t llama_model_size(const struct llama_model * model);

    // Get the default chat template. Returns nullptr if not available
    // If name is NULL, returns the default chat template
    LLAMA_API const char * llama_model_chat_template(const struct llama_model * model, const char * name);

    // Returns the total number of parameters in the model
    LLAMA_API uint64_t llama_model_n_params(const struct llama_model * model);

    // DFlash speculative-decoding draft model accessors.
    //
    // These getters return values stored in the GGUF of a DFlash draft model.
    // They return 0 / LLAMA_TOKEN_NULL / negative values when the model is
    // not a DFlash draft (i.e. the consumer can defensively call them without
    // first checking the architecture).

    // Block size: how many tokens the DFlash draft generates per forward pass.
    // Returns 0 if the model is not a DFlash draft.
    LLAMA_API uint32_t llama_model_dflash_block_size(const struct llama_model * model);

    // Mask token id used to fill the draft block prior to running the draft.
    // Returns LLAMA_TOKEN_NULL if the model is not a DFlash draft.
    LLAMA_API llama_token llama_model_dflash_mask_token_id(const struct llama_model * model);

    // Number of layer indices in the target_layer_ids list. Returns 0 if not DFlash.
    LLAMA_API int32_t llama_model_dflash_n_target_layer_ids(const struct llama_model * model);

    // Get the i-th target layer id (i.e. an index into the *target* model's layers,
    // whose hidden state should be captured and fed to the DFlash draft).
    // Returns -1 on out-of-range or non-DFlash model.
    LLAMA_API int32_t llama_model_dflash_target_layer_id(const struct llama_model * model, int32_t i);

    // DFlash: bind target model's tok_embd + lm_head into the draft
    //
    // Paper §4.2: "the draft model shares the token embedding layer and
    // language modeling head with the target model and keeps them frozen
    // during training." A paper-faithful DFlash draft GGUF therefore does
    // not contain its own tok_embd or output tensors; the inference engine
    // must bind them from the target.
    //
    // Call this once after loading both models, before running any draft
    // decode. It copies non-owning pointers from target → draft; the target
    // model must outlive the draft.
    //
    // No-op if the draft already has its own tok_embd / output (i.e. when
    // the GGUF was converted in self-contained mode). Returns false if
    // either model is null.
    LLAMA_API bool llama_dflash_bind_target(
            struct llama_model *       model_dft,
            const struct llama_model * model_tgt);

    // inline-encoder plumbing: populate non-owning pointers on the TARGET
    // model that reference the encoder weights loaded as part of the DRAFT
    // model (dflash_fc, dflash_hidden_norm, and per-target-layer
    // wk/wv/attn_k_norm). When
    // cparams.dflash_inline_encoder is enabled on the target context,
    // the target's graph builder uses these to run the encoder K/V
    // projection inline after the final captured layer instead of
    // calling a separate decode on the draft context.
    //
    // Must be called AFTER both models are loaded. Returns false if
    // either model is null, draft is not LLM_ARCH_DFLASH, or draft is
    // missing required encoder tensors. Pointers remain valid until
    // either model is freed; the caller is responsible for ordering.
    LLAMA_API bool llama_dflash_bind_encoder(
            struct llama_model *       model_tgt,
            const struct llama_model * model_dft);

    // [EXPERIMENTAL]
    // bind the draft context's per-layer side-store K/V tensors as set_rows
    // destinations on the target context, so the target's graph can write
    // encoder outputs directly into the draft's side store cross-context.
    // Must be called after both contexts are constructed. Returns false on
    // null args or if draft side store is empty (e.g., not a DFlash draft
    // context).
    LLAMA_API bool llama_dflash_bind_inline_side_store(
            struct llama_context * ctx_tgt,
            struct llama_context * ctx_dft);

    // [EXPERIMENTAL]
    // inline-encoder per-decode bookkeeping: set where in the side store to
    // write and the starting RoPE position for the encoder before each
    // llama_decode(ctx_tgt, batch). Read at graph compute time by the inline
    // encoder input class.
    LLAMA_API void llama_dflash_set_inline_encode_state(
            struct llama_context * ctx_tgt,
            int64_t                write_offset,
            int64_t                pos_start);

    // [EXPERIMENTAL]
    // advance the draft context's side-store ctx_filled by n_keep (the
    // accepted-token count). Used after the target's inline encoder has
    // written n_outputs rows to the side store; the speculative driver
    // calls this with n_keep == accepted count so subsequent draft reads
    // see only the accepted prefix. No-op when n_keep <= 0 or ctx is null.
    LLAMA_API void llama_dflash_inline_advance_ctx_filled(
            struct llama_context * ctx_dft,
            int64_t                n_keep);

    // High-water mark of side-store rows the inline encoder has scattered
    // since the last per-request reset. Updated host-side by the encoder's
    // input-class set_input() — survives across chunked llama_decode calls,
    // unlike captured_n_outputs which is per-call. Used by the speculative
    // begin() to detect that the caller's prefill has populated the side
    // store and skip a redundant fallback re-decode.
    // Returns 0 when ctx is null or the inline encoder hasn't run.
    LLAMA_API int64_t llama_dflash_inline_get_n_committed(
            struct llama_context * ctx_tgt);

    // DFlash + GDN partial-tail seq_rm: same as llama_memory_seq_rm but
    // allows partial-tail removal on hybrid memory backends (e.g. GDN+attn
    // combos) by rewinding the recurrent tail cell's pos to p0 - 1 instead
    // of returning false. The caller MUST overwrite the recurrent state
    // buffer for that cell before the next decode; the DFlash in-graph
    // fixup (state_select + cpy into ssm_states_all slot) provides this
    // guarantee when --dflash-gdn-history is enabled.
    //
    // Falls back to plain seq_rm on memory backends that don't have
    // recurrent state (so it's safe to call unconditionally from a
    // generic spec driver).
    //
    // Returns true on success, false on errors that mirror seq_rm's
    // existing rejection rules (invalid seq_id, etc.).
    LLAMA_API bool llama_dflash_memory_seq_rm_partial_tail_state_managed_externally(
            llama_memory_t mem,
            llama_seq_id   seq_id,
            llama_pos      p0,
            llama_pos      p1);

    // post-accept compaction of the unified KV cache after a tree-verify decode.
    //
    // The tree-verify ubatch is written with tree-depth positions (sibling
    // tree nodes share positions, requiring the
    // tree-mode bypass set by llama_set_tree_mask), this call restores the
    // contiguous-positions invariant for the accepted path:
    //
    //   - Cells with seq_id and pos in positions[0..n_positions-1] are
    //     RENAMED to consecutive positions [p_min, p_min + n_positions),
    //     in the same order as supplied. (Each kept cell's existing
    //     ext / 2D position is preserved.)
    //   - Cells with seq_id and pos >= p_min that are NOT in positions[]
    //     are DROPPED.
    //   - Cells with pos < p_min (the committed prefix) are UNTOUCHED.
    //
    // After the call, the seq is contiguous from p_min through
    // p_min + n_positions - 1, and subsequent decodes can extend
    // monotonically as the standard cache expects.
    //
    // Semantics on edge cases:
    //   - n_positions == 0   -> equivalent to seq_rm(seq_id, p_min, -1);
    //                            returns the same status seq_rm would.
    //   - seq_id  < 0        -> rejected (returns false). Tree compaction
    //                            is single-seq by construction.
    //   - positions == NULL with n_positions > 0 -> rejected.
    //
    // For composite memory backends (hybrid, iswa, hybrid+iswa), this
    // forwards to the attention half. Recurrent memory is a no-op: the
    // GDN+conv state is fixed up via the state_select / conv-history-select
    // APIs and needs no separate compaction here.
    //
    // Returns true on success, false on the rejection cases above or on
    // structural errors (cell with the requested seq_id alongside other
    // seq ids - not supported for single-seq compaction).
    LLAMA_API bool llama_memory_keep_positions_range(
            llama_memory_t    mem,
            llama_seq_id      seq_id,
            const llama_pos * positions,
            int32_t           n_positions,
            llama_pos         p_min);

    // [EXPERIMENTAL]
    // tree-aware variant of llama_memory_keep_positions_range that identifies
    // cells by their cell-walk ordinal (= the order in which apply_ubatch
    // wrote them during the immediately-preceding tree-mode ubatch) instead
    // of by position. This disambiguates sibling cells
    // that legitimately share a position in the tree-depth-position
    // verify layout: e.g. main-path token at depth d and an alt-branch
    // token at depth d both have pos = committed + d, and a position-
    // based keep call would non-deterministically pick the first cell
    // (typically the main-path one) — wrong on alt-accept.
    //
    //   - dfs_keep[] : strictly-increasing list of ordinals to keep.
    //                  Ordinal 0 = first cell with seq_id and pos >= p_min
    //                  in cell-index order (= DFS root). Ordinal i = i-th
    //                  such cell (= DFS slot i for the verify just written).
    //   - The kept cells are renamed to consecutive positions
    //     [p_min, p_min + n_keep) matching the order of dfs_keep[].
    //   - Cells with seq_id and pos >= p_min whose ordinal is not in
    //     dfs_keep[] are dropped.
    //   - Cells with pos < p_min are untouched.
    //
    // Edge cases:
    //   - n_keep == 0     -> equivalent to seq_rm(seq_id, p_min, -1)
    //   - seq_id < 0      -> rejected
    //   - dfs_keep == NULL with n_keep > 0 -> rejected
    //   - dfs_keep[] not strictly increasing -> rejected
    //   - dfs_keep[] references an ordinal that doesn't exist (cell walk
    //     ran out before reaching it) -> rejected
    //
    // Forwards to the attn half on composite backends; recurrent half is
    // a no-op (state fixup is handled by the state_select APIs).
    LLAMA_API bool llama_memory_keep_cells_dfs_ordinals_range(
            llama_memory_t  mem,
            llama_seq_id    seq_id,
            const int32_t * dfs_keep,
            int32_t         n_keep,
            llama_pos       p_min);

    // [EXPERIMENTAL]
    // GDN history fixup: set the per-decode k_index that the target's
    // GDN-layer state_select op consumes at the start of the next
    // llama_decode(ctx_tgt, ...).
    //
    // Semantics:
    //   k_index >= 0  : in the next verify graph, every GDN layer's
    //                   ssm_states_all slot is overwritten with
    //                   state_history[k_index] from the PREVIOUS verify
    //                   decode (the chain-verify-iter-N final state at
    //                   token position k_index inside the chain).
    //   k_index <  0  : no fixup — the state_select op falls back to the
    //                   current slot data; net effect is a self-cpy
    //                   (no-op). Use after a full-acceptance iter, before
    //                   the first decode of a new generation, or when
    //                   --dflash-gdn-history is disabled.
    //
    // The value is sticky: set once, applies to the next decode call.
    // The spec driver calls this between target decodes with K - 1
    // (K = accepted-token count) for partial acceptance.
    //
    // No-op on null ctx.
    LLAMA_API void llama_dflash_set_gdn_history_k_index(
            struct llama_context * ctx_tgt,
            int32_t                k_index);

    // [EXPERIMENTAL]
    // tree-mode variant: set per-seq k_index for the next decode.
    // `k_indices` is an array of `n_seqs` int32 entries (one accept depth
    // per branch in the tree-verify). Each entry follows the same
    // semantics as the scalar setter:
    //   k_indices[s] >= 0 : in the next verify graph, seq `s`'s GDN slab
    //                       is rolled back to state_history[k_indices[s], s].
    //   k_indices[s] <  0 : no fixup for seq `s` — the state_select op
    //                       falls back to copying the current
    //                       ssm_states_all slot for that seq.
    //
    // The value is sticky: set once, applies to the next decode call.
    // No-op on null ctx or null k_indices. n_seqs must match the
    // upcoming ubatch's n_seqs (or 1 in the chain compatibility path);
    // passing more is allowed (entries past n_seqs are written but
    // ignored by the state_select op).
    LLAMA_API void llama_dflash_set_gdn_history_k_index_per_seq(
            struct llama_context * ctx_tgt,
            const int32_t *        k_indices,
            int32_t                n_seqs);

    // [EXPERIMENTAL]
    // tree-mode: bind a parent_ids buffer (host-side int32, shape
    // [n_tokens, n_seqs]) for the next decode. Consumed by the
    // ggml_gated_delta_net_with_history_tree + ggml_ssm_conv_tree ops
    // in the Qwen3.5 family graphs. parent_ids[t, s] is the token index
    // within seq `s` of t's parent in the DFS-flattened tree, or
    // `GGML_GDN_TREE_ROOT_PARENT` (= -1) if t's parent is the pre-block
    // recurrent state.
    //
    // The caller retains ownership of the buffer; the context copies it
    // into its own per-decode I32 input tensor at graph set_input time
    // (typical: caller fills a std::vector<int32_t> from
    // common_speculative_tree::write_parent_ids() before each
    // tree-verify llama_decode). Pass `parent_ids = nullptr` to clear
    // (returns the next decode to chain mode).
    //
    // No-op on null ctx.
    LLAMA_API void llama_dflash_set_gdn_history_parent_ids(
            struct llama_context * ctx_tgt,
            const int32_t *        parent_ids,
            int32_t                n_tokens,
            int32_t                n_seqs);

    // DFlash target hidden-state capture (call on the *target* context).
    //
    // Tee out the post-block hidden state at the given target-model layer
    // indices on every llama_decode() call. After decode, retrieve the
    // captures via llama_dflash_get_captured_features().
    LLAMA_API void llama_dflash_set_capture(
            struct llama_context * ctx,
            const int32_t *        layer_ids,
            size_t                 n_layer_ids,
            int64_t                n_embd_target);

    // After llama_decode() on the target context, returns a pointer to the
    // captured features in row-major layout [n_features, n_outputs].
    LLAMA_API const float * llama_dflash_get_captured_features(
            struct llama_context * ctx,
            int64_t *              n_outputs_out);

    // After llama_decode() on a *draft* context, returns top-K candidate
    // tokens per output position. Returns NULL on a non-DFlash context.
    LLAMA_API const int32_t * llama_dflash_get_draft_topk(
            struct llama_context * ctx,
            int64_t *              n_outputs_out,
            uint32_t *             topk_out);

    // DFlash K/V cache reuse (call on the *draft* context).
    //
    // Append `n_new` newly-committed target features to the draft's
    // persistent K/V side store.
    LLAMA_API int32_t llama_dflash_extend(
            struct llama_context * ctx,
            const float *          target_hidden_new,
            int64_t                n_new,
            int64_t                pos_start);

    // [EXPERIMENTAL]
    // Device-to-device variant: reads captures directly from a source
    // (target) context's most recently produced packed-captures tensor,
    // skipping the D2H/H2D bounce that llama_dflash_extend() does via the
    // host buffer. Requires:
    //   - The TARGET context had llama_dflash_set_skip_host_readback(true)
    //     called once at init (or its host buffer is fine to ignore).
    //   - The TARGET context has just completed an llama_decode() that
    //     produced n_keep+src_row_offset captures.
    //   - Both contexts remain alive until the next llama_decode() on
    //     either side.
    // Falls back to the host path on cross-backend pairs that don't
    // support direct copy. Returns 0 on success, negative on error.
    LLAMA_API int32_t llama_dflash_extend_from_ctx(
            struct llama_context * dst_ctx,        // draft
            struct llama_context * src_ctx,        // target
            int64_t                src_row_offset, // start row in target's captures
            int64_t                n_keep,
            int64_t                pos_start);

    // [EXPERIMENTAL]
    // inline-encoder variant of llama_dflash_extend_from_ctx executed on the
    // TARGET context's scheduler instead of the draft's. Reads captures from
    // src_ctx (target) locally, writes K_new/V_new into dst_ctx (draft) side
    // store cross-context. Requires both contexts to
    // share singleton ggml_backend_t pointers per device (standard llama.cpp
    // init). Returns 0 on success; on negative return, the caller should
    // fall back to llama_dflash_extend_from_ctx.
    LLAMA_API int32_t llama_dflash_inline_encode_from_ctx(
            struct llama_context * tgt_ctx,        // target — runs encoder on its sched
            struct llama_context * dft_ctx,        // draft  — provides encoder weights + side store
            int64_t                src_row_offset,
            int64_t                n_keep,
            int64_t                pos_start);

    // [EXPERIMENTAL]
    // Configure the TARGET context to skip the D2H readback of captured
    // features on each llama_decode(). After this is set, consumers MUST
    // use llama_dflash_extend_from_ctx() rather than the host-pointer
    // variant; llama_dflash_get_captured_features() will return NULL.
    LLAMA_API void llama_dflash_set_skip_host_readback(
            struct llama_context * ctx,
            bool                   skip);

    // [EXPERIMENTAL]
    // One-shot D2H readback of the most recently produced packed captures
    // from the TARGET context. Use this in skip_host_readback mode when
    // the consumer needs host bytes for a specific iteration (e.g., the
    // alt-accept remap path). After this returns successfully,
    // llama_dflash_get_captured_features() yields valid host bytes for
    // this iteration. Returns 0 on success, negative on error (no captures
    // produced this step, etc).
    LLAMA_API int32_t llama_dflash_force_host_readback(
            struct llama_context * ctx);

    // Reset the K/V side store (e.g. on a new prompt).
    LLAMA_API void llama_dflash_reset_ctx_kv(struct llama_context * ctx);

    // Full per-request DFlash reset: zeroes the K/V side store buffers AND
    // the per-layer GDN history buffers AND counters (ctx_filled,
    // captured_n_outputs, inline_pos_start, inline_write_offset). Use
    // between requests on the same slot to prevent state leaks. Not safe
    // under multi-slot serving (touches buffers shared across all slots).
    // No-op on non-DFlash contexts.
    LLAMA_API void llama_dflash_reset_for_new_request(struct llama_context * ctx);

    // tree-shaped attention mask
    LLAMA_API void llama_set_tree_mask(
            struct llama_context * ctx,
            const uint8_t        * visibility,
            int                    n_tree_tokens);

    LLAMA_API void llama_clear_tree_mask(struct llama_context * ctx);

    // Returns true if the model contains an encoder that requires llama_encode() call
    LLAMA_API bool llama_model_has_encoder(const struct llama_model * model);

    // Returns true if the model contains a decoder that requires llama_decode() call
    LLAMA_API bool llama_model_has_decoder(const struct llama_model * model);

    // For encoder-decoder models, this function returns id of the token that must be provided
    // to the decoder to start generating output sequence. For other models, it returns -1.
    LLAMA_API llama_token llama_model_decoder_start_token(const struct llama_model * model);

    // Returns true if the model is recurrent (like Mamba, RWKV, etc.)
    LLAMA_API bool llama_model_is_recurrent(const struct llama_model * model);

    // Returns true if the model is hybrid (like Jamba, Granite, etc.)
    LLAMA_API bool llama_model_is_hybrid(const struct llama_model * model);

    // Returns true if the model is diffusion-based (like LLaDA, Dream, etc.)
    LLAMA_API bool llama_model_is_diffusion(const struct llama_model * model);

    // Returns 0 on success
    LLAMA_API uint32_t llama_model_quantize(
            const char * fname_inp,
            const char * fname_out,
            const llama_model_quantize_params * params);

    //
    // Adapters
    //

    // Load a LoRA adapter from file
    // The adapter is valid as long as the associated model is not freed
    LLAMA_API struct llama_adapter_lora * llama_adapter_lora_init(
            struct llama_model * model,
            const char * path_lora);

    // Functions to access the adapter's GGUF metadata scalar values
    // - The functions return the length of the string on success, or -1 on failure
    // - The output string is always null-terminated and cleared on failure
    // - When retrieving a string, an extra byte must be allocated to account for the null terminator
    // - GGUF array values are not supported by these functions

    // Get metadata value as a string by key name
    LLAMA_API int32_t llama_adapter_meta_val_str(const struct llama_adapter_lora * adapter, const char * key, char * buf, size_t buf_size);

    // Get the number of metadata key/value pairs
    LLAMA_API int32_t llama_adapter_meta_count(const struct llama_adapter_lora * adapter);

    // Get metadata key name by index
    LLAMA_API int32_t llama_adapter_meta_key_by_index(const struct llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Get metadata value as a string by index
    LLAMA_API int32_t llama_adapter_meta_val_str_by_index(const struct llama_adapter_lora * adapter, int32_t i, char * buf, size_t buf_size);

    // Manually free a LoRA adapter
    // NOTE: loaded adapters that are not manually freed will be freed when the associated model is deleted
    LLAMA_API void llama_adapter_lora_free(struct llama_adapter_lora * adapter);

    // Get the invocation tokens if the current lora is an alora
    LLAMA_API uint64_t            llama_adapter_get_alora_n_invocation_tokens(const struct llama_adapter_lora * adapter);
    LLAMA_API const llama_token * llama_adapter_get_alora_invocation_tokens  (const struct llama_adapter_lora * adapter);

    // The following functions operate on a llama_context, hence the naming: llama_verb_...

    // Set LoRa adapters on the context. Will only modify if the adapters currently in context are different.
    LLAMA_API int32_t llama_set_adapters_lora(
            struct llama_context * ctx,
            struct llama_adapter_lora ** adapters,
            size_t n_adapters,
            float * scales);

    // Apply a loaded control vector to a llama_context, or if data is NULL, clear
    // the currently loaded vector.
    // n_embd should be the size of a single layer's control, and data should point
    // to an n_embd x n_layers buffer starting from layer 1.
    // il_start and il_end are the layer range the vector should apply to (both inclusive)
    // See llama_control_vector_load in common to load a control vector.
    LLAMA_API int32_t llama_set_adapter_cvec(
            struct llama_context * ctx,
                     const float * data,
                          size_t   len,
                         int32_t   n_embd,
                         int32_t   il_start,
                         int32_t   il_end);

    //
    // Memory
    //

    // Clear the memory contents
    // If data == true, the data buffers will also be cleared together with the metadata
    LLAMA_API void llama_memory_clear(
            llama_memory_t mem,
                      bool data);

    // Removes all tokens that belong to the specified sequence and have positions in [p0, p1)
    // Returns false if a partial sequence cannot be removed. Removing a whole sequence never fails
    // seq_id < 0 : match any sequence
    // p0 < 0     : [0,  p1]
    // p1 < 0     : [p0, inf)
    LLAMA_API bool llama_memory_seq_rm(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1);

    // Copy all tokens that belong to the specified sequence to another sequence
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_cp(
            llama_memory_t mem,
              llama_seq_id seq_id_src,
              llama_seq_id seq_id_dst,
                 llama_pos p0,
                 llama_pos p1);

    // Removes all tokens that do not belong to the specified sequence
    LLAMA_API void llama_memory_seq_keep(
            llama_memory_t mem,
              llama_seq_id seq_id);

    // Adds relative position "delta" to all tokens that belong to the specified sequence and have positions in [p0, p1)
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_add(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1,
                 llama_pos delta);

    // Integer division of the positions by factor of `d > 1`
    // p0 < 0 : [0,  p1]
    // p1 < 0 : [p0, inf)
    LLAMA_API void llama_memory_seq_div(
            llama_memory_t mem,
              llama_seq_id seq_id,
                 llama_pos p0,
                 llama_pos p1,
                       int d);

    // Returns the smallest position present in the memory for the specified sequence
    // This is typically non-zero only for SWA caches
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    LLAMA_API llama_pos llama_memory_seq_pos_min(
            llama_memory_t mem,
              llama_seq_id seq_id);

    // Returns the largest position present in the memory for the specified sequence
    // Note that all positions in the range [pos_min, pos_max] are guaranteed to be present in the memory
    // Return -1 if the sequence is empty
    LLAMA_API llama_pos llama_memory_seq_pos_max(
            llama_memory_t mem,
              llama_seq_id seq_id);

    // Check if the memory supports shifting
    LLAMA_API bool llama_memory_can_shift(llama_memory_t mem);

    //
    // State / sessions
    //

    // Returns the *actual* size in bytes of the state
    // (logits, embedding and memory)
    // Only use when saving the state, not when restoring it, otherwise the size may be too small.
    LLAMA_API size_t llama_state_get_size(struct llama_context * ctx);
    LLAMA_API DEPRECATED(size_t llama_get_state_size(struct llama_context * ctx),
        "use llama_state_get_size instead");

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    LLAMA_API size_t llama_state_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_copy_state_data(
            struct llama_context * ctx,
                         uint8_t * dst),
        "use llama_state_get_data instead");

    // Set the state reading from the specified address
    // Returns the number of bytes read
    LLAMA_API size_t llama_state_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size);
    LLAMA_API DEPRECATED(size_t llama_set_state_data(
            struct llama_context * ctx,
                   const uint8_t * src),
        "use llama_state_set_data instead");

    // Save/load session file
    LLAMA_API bool llama_state_load_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);
    LLAMA_API DEPRECATED(bool llama_load_session_file(
            struct llama_context * ctx,
                      const char * path_session,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out),
        "use llama_state_load_file instead");

    LLAMA_API bool llama_state_save_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count);
    LLAMA_API DEPRECATED(bool llama_save_session_file(
            struct llama_context * ctx,
                      const char * path_session,
               const llama_token * tokens,
                          size_t   n_token_count),
        "use llama_state_save_file instead");

    // Get the exact size needed to copy the state of a single sequence
    LLAMA_API size_t llama_state_seq_get_size(
            struct llama_context * ctx,
                    llama_seq_id   seq_id);

    // Copy the state of a single sequence into the specified buffer
    LLAMA_API size_t llama_state_seq_get_data(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id);

    // Copy the sequence data (originally copied with `llama_state_seq_get_data`) into the specified sequence
    // Returns:
    //  - Positive: Ok
    //  - Zero: Failed to load
    LLAMA_API size_t llama_state_seq_set_data(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id);

    LLAMA_API size_t llama_state_seq_save_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   seq_id,
               const llama_token * tokens,
                          size_t   n_token_count);

    LLAMA_API size_t llama_state_seq_load_file(
            struct llama_context * ctx,
                      const char * filepath,
                    llama_seq_id   dest_seq_id,
                     llama_token * tokens_out,
                          size_t   n_token_capacity,
                          size_t * n_token_count_out);

// for backwards-compat
#define LLAMA_STATE_SEQ_FLAGS_SWA_ONLY 1

// work only with partial states, such as SWA KV cache or recurrent cache (e.g. Mamba)
#define LLAMA_STATE_SEQ_FLAGS_PARTIAL_ONLY 1

// keeps the tensor data on device buffers (i.e. not accessible in host memory, but faster save/load)
#define LLAMA_STATE_SEQ_FLAGS_ON_DEVICE 2

    typedef uint32_t llama_state_seq_flags;

    LLAMA_API size_t llama_state_seq_get_size_ext(
            struct llama_context * ctx,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags);

    LLAMA_API size_t llama_state_seq_get_data_ext(
            struct llama_context * ctx,
                         uint8_t * dst,
                          size_t   size,
                    llama_seq_id   seq_id,
           llama_state_seq_flags   flags);

    LLAMA_API size_t llama_state_seq_set_data_ext(
            struct llama_context * ctx,
                   const uint8_t * src,
                          size_t   size,
                    llama_seq_id   dest_seq_id,
           llama_state_seq_flags   flags);

    //
    // Decoding
    //

    // Return batch for single sequence of tokens
    // The sequence ID will be fixed to 0
    // The position of the tokens will be tracked automatically by llama_decode
    //
    // NOTE: this is a helper function to facilitate transition to the new batch API - avoid using it
    //
    LLAMA_API struct llama_batch llama_batch_get_one(
                  llama_token * tokens,
                      int32_t   n_tokens);

    // Allocates a batch of tokens on the heap that can hold a maximum of n_tokens
    // Each token can be assigned up to n_seq_max sequence ids
    // The batch has to be freed with llama_batch_free()
    // If embd != 0, llama_batch.embd will be allocated with size of n_tokens * embd * sizeof(float)
    // Otherwise, llama_batch.token will be allocated to store n_tokens llama_token
    // The rest of the llama_batch members are allocated with size n_tokens
    // All members are left uninitialized
    LLAMA_API struct llama_batch llama_batch_init(
            int32_t n_tokens,
            int32_t embd,
            int32_t n_seq_max);

    // Frees a batch of tokens allocated with llama_batch_init()
    LLAMA_API void llama_batch_free(struct llama_batch batch);

    // Process a batch of tokens.
    // In contrast to llama_decode() - this call does not use KV cache.
    // For encode-decoder contexts, processes the batch using the encoder.
    // Can store the encoder output internally for later use by the decoder's cross-attention layers.
    //   0 - success
    // < 0 - error. the memory state is restored to the state before this call
    LLAMA_API int32_t llama_encode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Process a batch of tokens.
    // Requires the context to have a memory.
    // For encode-decoder contexts, processes the batch using the decoder.
    // Positive return values does not mean a fatal error, but rather a warning.
    // Upon fatal-error or abort, the ubatches that managed to be been processed will remain in the memory state of the context
    //   To handle this correctly, query the memory state using llama_memory_seq_pos_min() and llama_memory_seq_pos_max()
    // Upon other return values, the memory state is restored to the state before this call
    //    0 - success
    //    1 - could not find a KV slot for the batch (try reducing the size of the batch or increase the context)
    //    2 - aborted     (processed ubatches will remain in the context's memory)
    //   -1 - invalid input batch
    // < -1 - fatal error (processed ubatches will remain in the context's memory)
    LLAMA_API int32_t llama_decode(
            struct llama_context * ctx,
              struct llama_batch   batch);

    // Set the number of threads used for decoding
    // n_threads is the number of threads used for generation (single token)
    // n_threads_batch is the number of threads used for prompt and batch processing (multiple tokens)
    LLAMA_API void llama_set_n_threads(struct llama_context * ctx, int32_t n_threads, int32_t n_threads_batch);

    // Get the number of threads used for generation of a single token.
    LLAMA_API int32_t llama_n_threads(struct llama_context * ctx);

    // Get the number of threads used for prompt and batch processing (multiple token).
    LLAMA_API int32_t llama_n_threads_batch(struct llama_context * ctx);

    // Set whether the context outputs embeddings or not
    // TODO: rename to avoid confusion with llama_get_embeddings()
    LLAMA_API void llama_set_embeddings(struct llama_context * ctx, bool embeddings);

    // Set whether to use causal attention or not
    // If set to true, the model will only attend to the past tokens
    LLAMA_API void llama_set_causal_attn(struct llama_context * ctx, bool causal_attn);

    // Set whether the model is in warmup mode or not
    // If true, all model tensors are activated during llama_decode() to load and cache their weights.
    LLAMA_API void llama_set_warmup(struct llama_context * ctx, bool warmup);

    // Set abort callback
    LLAMA_API void llama_set_abort_callback(struct llama_context * ctx, ggml_abort_callback abort_callback, void * abort_callback_data);

    // Wait until all computations are finished
    // This is automatically done when using one of the functions below to obtain the computation results
    // and is not necessary to call it explicitly in most cases
    LLAMA_API void llama_synchronize(struct llama_context * ctx);

    // Token logits obtained from the last call to llama_decode()
    // The logits for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // Rows: number of tokens for which llama_batch.logits[i] != 0
    // Cols: n_vocab
    // TODO: deprecate in favor of llama_get_logits_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    LLAMA_API float * llama_get_logits(struct llama_context * ctx);

    // Logits for the ith token. For positive indices, Equivalent to:
    // llama_get_logits(ctx) + ctx->output_ids[i]*n_vocab
    // Negative indices can be used to access logits in reverse order, -1 is the last logit.
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_logits_ith(struct llama_context * ctx, int32_t i);

    // Get all output token embeddings.
    // when pooling_type == LLAMA_POOLING_TYPE_NONE or when using a generative model,
    // the embeddings for which llama_batch.logits[i] != 0 are stored contiguously
    // in the order they have appeared in the batch.
    // shape: [n_outputs*n_embd]
    // Otherwise, returns NULL.
    // TODO: deprecate in favor of llama_get_embeddings_ith() (ref: https://github.com/ggml-org/llama.cpp/pull/14853#issuecomment-3113143522)
    LLAMA_API float * llama_get_embeddings(struct llama_context * ctx);

    // Get the embeddings for the ith token. For positive indices, Equivalent to:
    // llama_get_embeddings(ctx) + ctx->output_ids[i]*n_embd
    // Negative indices can be used to access embeddings in reverse order, -1 is the last embedding.
    // shape: [n_embd] (1-dimensional)
    // returns NULL for invalid ids.
    LLAMA_API float * llama_get_embeddings_ith(struct llama_context * ctx, int32_t i);

    // Get the embeddings for a sequence id
    // Returns NULL if pooling_type is LLAMA_POOLING_TYPE_NONE
    // when pooling_type == LLAMA_POOLING_TYPE_RANK, returns float[n_cls_out] with the rank(s) of the sequence
    // otherwise: float[n_embd] (1-dimensional)
    LLAMA_API float * llama_get_embeddings_seq(struct llama_context * ctx, llama_seq_id seq_id);

    //
    // backend sampling API [EXPERIMENTAL]
    // note: use only if the llama_context was created with at least one llama_sampler_seq_config
    //

    // Get the backend sampled token for the ith token.
    // Returns LLAMA_TOKEN_NULL if no token was sampled.
    LLAMA_API llama_token llama_get_sampled_token_ith(struct llama_context * ctx, int32_t i);

    // Get the per-output greedy argmax of the final logits. Populated by an
    // always-on ggml_argmax over t_logits, so available regardless of
    // whether --backend-sampling is enabled. Lets greedy-equivalent
    // consumers (DFlash spec verify accept) avoid the bs * n_vocab * 4
    // byte logits readback.
    //
    // llama_get_logits_argmax_ith: idx semantics match llama_get_logits_ith;
    //                              i = -1 = last output.
    //                              returns LLAMA_TOKEN_NULL if unavailable.
    //
    // llama_get_logits_argmax: returns a pointer to the n_outputs-length
    //                          int32 buffer (post-reorder), or NULL.
    //                          The buffer is owned by ctx and invalidated
    //                          on the next llama_decode.
    LLAMA_API llama_token     llama_get_logits_argmax_ith(struct llama_context * ctx, int32_t i);
    LLAMA_API const int32_t * llama_get_logits_argmax    (struct llama_context * ctx);

    // Get the backend sampled probabilities for the ith token
    // The index matches llama_get_sampled_token_ith().
    // Returns NULL if no probabilities were generated.
    LLAMA_API float *  llama_get_sampled_probs_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t llama_get_sampled_probs_count_ith(struct llama_context * ctx, int32_t i);

    // Get the backend sampled logits for the ith token
    // Returns NULL if no logits were sampled.
    LLAMA_API float *  llama_get_sampled_logits_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t llama_get_sampled_logits_count_ith(struct llama_context * ctx, int32_t i);

    // Get the backend sampled candidates (token ids) for the ith token
    // These are needed to map probability/logit indices to vocab token ids.
    // Returns NULL if no candidates were sampled.
    LLAMA_API llama_token * llama_get_sampled_candidates_ith      (struct llama_context * ctx, int32_t i);
    LLAMA_API uint32_t      llama_get_sampled_candidates_count_ith(struct llama_context * ctx, int32_t i);

    //
    // Vocab
    //

    LLAMA_API const char * llama_vocab_get_text(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API float llama_vocab_get_score(const struct llama_vocab * vocab, llama_token token);

    LLAMA_API enum llama_token_attr llama_vocab_get_attr(const struct llama_vocab * vocab, llama_token token);

    // Check if the token is supposed to end generation (end-of-generation, eg. EOS, EOT, etc.)
    LLAMA_API bool llama_vocab_is_eog(const struct llama_vocab * vocab, llama_token token);

    // Identify if Token Id is a control token or a render-able token
    LLAMA_API bool llama_vocab_is_control(const struct llama_vocab * vocab, llama_token token);

    // Special tokens
    LLAMA_API llama_token llama_vocab_bos(const struct llama_vocab * vocab); // beginning-of-sentence
    LLAMA_API llama_token llama_vocab_eos(const struct llama_vocab * vocab); // end-of-sentence
    LLAMA_API llama_token llama_vocab_eot(const struct llama_vocab * vocab); // end-of-turn
    LLAMA_API llama_token llama_vocab_sep(const struct llama_vocab * vocab); // sentence separator
    LLAMA_API llama_token llama_vocab_nl (const struct llama_vocab * vocab); // next-line
    LLAMA_API llama_token llama_vocab_pad(const struct llama_vocab * vocab); // padding
    LLAMA_API llama_token llama_vocab_mask(const struct llama_vocab * vocab); // mask

    LLAMA_API bool llama_vocab_get_add_bos(const struct llama_vocab * vocab);
    LLAMA_API bool llama_vocab_get_add_eos(const struct llama_vocab * vocab);
    LLAMA_API bool llama_vocab_get_add_sep(const struct llama_vocab * vocab);

    LLAMA_API llama_token llama_vocab_fim_pre(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_suf(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_mid(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_pad(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_rep(const struct llama_vocab * vocab);
    LLAMA_API llama_token llama_vocab_fim_sep(const struct llama_vocab * vocab);

    DEPRECATED(LLAMA_API const char * llama_token_get_text(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_text instead");
    DEPRECATED(LLAMA_API float llama_token_get_score(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_score instead");
    DEPRECATED(LLAMA_API enum llama_token_attr llama_token_get_attr(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_get_attr instead");
    DEPRECATED(LLAMA_API bool llama_token_is_eog(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_is_eog instead");
    DEPRECATED(LLAMA_API bool llama_token_is_control(const struct llama_vocab * vocab, llama_token token), "use llama_vocab_is_control instead");
    DEPRECATED(LLAMA_API llama_token llama_token_bos(const struct llama_vocab * vocab), "use llama_vocab_bos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_eos(const struct llama_vocab * vocab), "use llama_vocab_eos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_eot(const struct llama_vocab * vocab), "use llama_vocab_eot instead");
    DEPRECATED(LLAMA_API llama_token llama_token_cls(const struct llama_vocab * vocab), "use llama_vocab_cls instead");
    DEPRECATED(LLAMA_API llama_token llama_token_sep(const struct llama_vocab * vocab), "use llama_vocab_sep instead");
    DEPRECATED(LLAMA_API llama_token llama_token_nl (const struct llama_vocab * vocab), "use llama_vocab_nl instead");
    DEPRECATED(LLAMA_API llama_token llama_token_pad(const struct llama_vocab * vocab), "use llama_vocab_pad instead");
    DEPRECATED(LLAMA_API bool llama_add_bos_token(const struct llama_vocab * vocab), "use llama_vocab_get_add_bos instead");
    DEPRECATED(LLAMA_API bool llama_add_eos_token(const struct llama_vocab * vocab), "use llama_vocab_get_add_eos instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_pre(const struct llama_vocab * vocab), "use llama_vocab_fim_pre instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_suf(const struct llama_vocab * vocab), "use llama_vocab_fim_suf instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_mid(const struct llama_vocab * vocab), "use llama_vocab_fim_mid instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_pad(const struct llama_vocab * vocab), "use llama_vocab_fim_pad instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_rep(const struct llama_vocab * vocab), "use llama_vocab_fim_rep instead");
    DEPRECATED(LLAMA_API llama_token llama_token_fim_sep(const struct llama_vocab * vocab), "use llama_vocab_fim_sep instead");

    // CLS is equivalent to BOS
    DEPRECATED(LLAMA_API llama_token llama_vocab_cls(const struct llama_vocab * vocab), // classification
            "use llama_vocab_bos instead");

    //
    // Tokenization
    //
    // The API is thread-safe.
    //

    /// @details Convert the provided text into tokens.
    /// @param tokens The tokens pointer must be large enough to hold the resulting tokens.
    /// @return Returns the number of tokens on success, no more than n_tokens_max
    /// @return Returns a negative number on failure - the number of tokens that would have been returned
    /// @return Returns INT32_MIN on overflow (e.g., tokenization result size exceeds int32_t limit)
    /// @param add_special Allow to add BOS and EOS tokens if model is configured to do so.
    /// @param parse_special Allow tokenizing special and/or control tokens which otherwise are not exposed and treated
    ///                      as plaintext. Does not insert a leading space.
    LLAMA_API int32_t llama_tokenize(
        const struct llama_vocab * vocab,
                      const char * text,
                         int32_t   text_len,
                     llama_token * tokens,
                         int32_t   n_tokens_max,
                            bool   add_special,
                            bool   parse_special);

    // Token Id -> Piece.
    // Uses the vocabulary in the provided context.
    // Does not write null terminator to the buffer.
    // User can skip up to 'lstrip' leading spaces before copying (useful when encoding/decoding multiple tokens with 'add_space_prefix')
    // @param special If true, special tokens are rendered in the output.
    LLAMA_API int32_t llama_token_to_piece(
              const struct llama_vocab * vocab,
                           llama_token   token,
                                  char * buf,
                               int32_t   length,
                               int32_t   lstrip,
                                  bool   special);

    /// @details Convert the provided tokens into text (inverse of llama_tokenize()).
    /// @param text The char pointer must be large enough to hold the resulting text.
    /// @return Returns the number of chars/bytes on success, no more than text_len_max.
    /// @return Returns a negative number on failure - the number of chars/bytes that would have been returned.
    /// @param remove_special Allow to remove BOS and EOS tokens if model is configured to do so.
    /// @param unparse_special If true, special tokens are rendered in the output.
    LLAMA_API int32_t llama_detokenize(
        const struct llama_vocab * vocab,
               const llama_token * tokens,
                         int32_t   n_tokens,
                            char * text,
                         int32_t   text_len_max,
                            bool   remove_special,
                            bool   unparse_special);

    //
    // Chat templates
    //

    /// Apply chat template. Inspired by hf apply_chat_template() on python.
    ///
    /// NOTE: This function does not use a jinja parser. It only support a pre-defined list of template. See more: https://github.com/ggml-org/llama.cpp/wiki/Templates-supported-by-llama_chat_apply_template
    /// @param tmpl A Jinja template to use for this chat.
    /// @param chat Pointer to a list of multiple llama_chat_message
    /// @param n_msg Number of llama_chat_message in this chat
    /// @param add_ass Whether to end the prompt with the token(s) that indicate the start of an assistant message.
    /// @param buf A buffer to hold the output formatted prompt. The recommended alloc size is 2 * (total number of characters of all messages)
    /// @param length The size of the allocated buffer
    /// @return The total number of bytes of the formatted prompt. If is it larger than the size of buffer, you may need to re-alloc it and then re-apply the template.
    LLAMA_API int32_t llama_chat_apply_template(
                            const char * tmpl,
       const struct llama_chat_message * chat,
                                size_t   n_msg,
                                  bool   add_ass,
                                  char * buf,
                               int32_t   length);

    // Get list of built-in chat templates
    LLAMA_API int32_t llama_chat_builtin_templates(const char ** output, size_t len);

    //
    // Sampling API
    //
    // Sample usage:
    //
    //    // prepare the sampling chain at the start
    //    auto sparams = llama_sampler_chain_default_params();
    //
    //    llama_sampler * smpl = llama_sampler_chain_init(sparams);
    //
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_k(50));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_top_p(0.9, 1));
    //    llama_sampler_chain_add(smpl, llama_sampler_init_temp (0.8));
    //
    //    // typically, the chain should end with a sampler such as "greedy", "dist" or "mirostat"
    //    // this sampler will be responsible to select the actual token
    //    llama_sampler_chain_add(smpl, llama_sampler_init_dist(seed));
    //
    //    ...
    //
    //    // decoding loop:
    //    while (...) {
    //        ...
    //
    //        llama_decode(ctx, batch);
    //
    //        // sample from the logits of the last token in the batch
    //        const llama_token id = llama_sampler_sample(smpl, ctx, -1);
    //
    //        ...
    //    }
    //
    //    llama_sampler_free(smpl);
    //

    typedef void * llama_sampler_context_t;

    struct llama_sampler_data {
        struct ggml_tensor * logits;
        struct ggml_tensor * probs;
        struct ggml_tensor * sampled;
        struct ggml_tensor * candidates;
    };

    // user code can implement the interface below in order to create custom llama_sampler
    struct llama_sampler_i {
        const char *           (*name)  (const struct llama_sampler * smpl);                                 // can be NULL
        void                   (*accept)(      struct llama_sampler * smpl, llama_token token);              // can be NULL
        void                   (*apply) (      struct llama_sampler * smpl, llama_token_data_array * cur_p); // required
        void                   (*reset) (      struct llama_sampler * smpl);                                 // can be NULL
        struct llama_sampler * (*clone) (const struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL
        void                   (*free)  (      struct llama_sampler * smpl);                                 // can be NULL if ctx is NULL

        // [EXPERIMENTAL]
        // backend sampling interface:

        // return true if the backend supports all ops needed by the sampler
        // note: call once per sampler
        bool (*backend_init)(struct llama_sampler * smpl, ggml_backend_buffer_type_t buft);

        // call after .backend_apply()
        void (*backend_accept)(
                struct llama_sampler * smpl,
                struct ggml_context  * ctx,
                struct ggml_cgraph   * gf,
                struct ggml_tensor   * selected_token);

        // call after .backend_init()
        void (*backend_apply)(
                struct llama_sampler      * smpl,
                struct ggml_context       * ctx,
                struct ggml_cgraph        * gf,
                struct llama_sampler_data * data);

        // called before graph execution to set inputs for the current ubatch
        void (*backend_set_input)(struct llama_sampler * smpl);
    };

    struct llama_sampler {
        struct llama_sampler_i * iface;

        llama_sampler_context_t ctx;
    };

    // [EXPERIMENTAL]
    // attach a sampler to the context
    // note: prefer initializing the context with llama_context_params.samplers when possible
    LLAMA_API bool llama_set_sampler(struct llama_context * ctx, llama_seq_id seq_id, struct llama_sampler * smpl);

    // mirror of llama_sampler_i:
    LLAMA_API struct llama_sampler * llama_sampler_init  (      struct llama_sampler_i * iface, llama_sampler_context_t ctx);
    LLAMA_API const char *           llama_sampler_name  (const struct llama_sampler * smpl);
    LLAMA_API void                   llama_sampler_accept(      struct llama_sampler * smpl, llama_token token);
    LLAMA_API void                   llama_sampler_apply (      struct llama_sampler * smpl, llama_token_data_array * cur_p);
    LLAMA_API void                   llama_sampler_reset (      struct llama_sampler * smpl);
    LLAMA_API struct llama_sampler * llama_sampler_clone (const struct llama_sampler * smpl);
    // important: do not free if the sampler has been added to a llama_sampler_chain (via llama_sampler_chain_add)
    LLAMA_API void                   llama_sampler_free  (      struct llama_sampler * smpl);

    // llama_sampler_chain
    // a type of llama_sampler that can chain multiple samplers one after another

    LLAMA_API struct llama_sampler * llama_sampler_chain_init(struct llama_sampler_chain_params params);

    // important: takes ownership of the sampler object and will free it when llama_sampler_free is called
    LLAMA_API void                   llama_sampler_chain_add(      struct llama_sampler * chain, struct llama_sampler * smpl);

    // return NULL if:
    //   - the sampler is NULL
    //   - the sampler is not a llama_sampler_chain
    //   - the index is out of bounds, unless i == -1
    //   - if i == -1, returns the chain itself (can be used to check if the sampler is a chain)
    LLAMA_API struct llama_sampler * llama_sampler_chain_get(      struct llama_sampler * chain, int32_t i);

    // the total number of samplers in the chain
    LLAMA_API int                    llama_sampler_chain_n  (const struct llama_sampler * chain);

    // after removing a sampler, the chain will no longer own it, and it will not be freed when the chain is freed
    LLAMA_API struct llama_sampler * llama_sampler_chain_remove(   struct llama_sampler * chain, int32_t i);

    // available samplers:

    LLAMA_API struct llama_sampler * llama_sampler_init_greedy(void);

    /// seed == LLAMA_DEFAULT_SEED to use a random seed.
    LLAMA_API struct llama_sampler * llama_sampler_init_dist(uint32_t seed);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    /// Setting k <= 0 makes this a noop
    LLAMA_API struct llama_sampler * llama_sampler_init_top_k      (int32_t k);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    LLAMA_API struct llama_sampler * llama_sampler_init_top_p      (float   p, size_t min_keep);

    /// @details Minimum P sampling as described in https://github.com/ggml-org/llama.cpp/pull/3841
    LLAMA_API struct llama_sampler * llama_sampler_init_min_p      (float   p, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    LLAMA_API struct llama_sampler * llama_sampler_init_typical    (float   p, size_t min_keep);

    /// #details Updates the logits l_i` = l_i/t. When t <= 0.0f, the maximum logit is kept at it's original value, the rest are set to -inf
    LLAMA_API struct llama_sampler * llama_sampler_init_temp       (float   t);

    /// @details Dynamic temperature implementation (a.k.a. entropy) described in the paper https://arxiv.org/abs/2309.02772.
    LLAMA_API struct llama_sampler * llama_sampler_init_temp_ext   (float   t, float   delta, float exponent);

    /// @details XTC sampler as described in https://github.com/oobabooga/text-generation-webui/pull/6335
    LLAMA_API struct llama_sampler * llama_sampler_init_xtc        (float   p, float   t,     size_t min_keep, uint32_t seed);

    /// @details Top n sigma sampling as described in academic paper "Top-nσ: Not All Logits Are You Need" https://arxiv.org/pdf/2411.07641
    LLAMA_API struct llama_sampler * llama_sampler_init_top_n_sigma(float   n);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat(
                             int32_t   n_vocab,
                            uint32_t   seed,
                               float   tau,
                               float   eta,
                             int32_t   m);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `llama_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    LLAMA_API struct llama_sampler * llama_sampler_init_mirostat_v2(
                            uint32_t   seed,
                               float   tau,
                               float   eta);

    /// @details Initializes a GBNF grammar, see grammars/README.md for details.
    /// @param vocab The vocabulary that this grammar will be used with.
    /// @param grammar_str The production rules for the grammar, encoded as a string. Returns an empty grammar if empty. Returns NULL if parsing of grammar_str fails.
    /// @param grammar_root The name of the start symbol for the grammar.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root);

    DEPRECATED(LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy(
            const struct llama_vocab * vocab,
                          const char * grammar_str,
                          const char * grammar_root,
                         const char ** trigger_words,
                                size_t num_trigger_words,
                   const llama_token * trigger_tokens,
                                size_t num_trigger_tokens),
        "use llama_sampler_init_grammar_lazy_patterns instead");


    /// @details Lazy grammar sampler, introduced in https://github.com/ggml-org/llama.cpp/pull/9639
    /// @param trigger_patterns A list of patterns that will trigger the grammar sampler. Pattern will be matched from the start of the generation output, and grammar sampler will be fed content starting from its first match group.
    /// @param trigger_tokens A list of tokens that will trigger the grammar sampler. Grammar sampler will be fed content starting from the trigger token included.
    LLAMA_API struct llama_sampler * llama_sampler_init_grammar_lazy_patterns(
        const struct llama_vocab * vocab,
                      const char * grammar_str,
                      const char * grammar_root,
                     const char ** trigger_patterns,
                            size_t num_trigger_patterns,
               const llama_token * trigger_tokens,
                            size_t num_trigger_tokens);


    /// NOTE: Avoid using on the full vocabulary as searching for repeated tokens can become slow. For example, apply top-k or top-p sampling first.
    LLAMA_API struct llama_sampler * llama_sampler_init_penalties(
                             int32_t   penalty_last_n,   // last n tokens to penalize (0 = disable penalty, -1 = context size)
                               float   penalty_repeat,   // 1.0 = disabled
                               float   penalty_freq,     // 0.0 = disabled
                               float   penalty_present); // 0.0 = disabled

    ///  @details DRY sampler, designed by p-e-w, as described in: https://github.com/oobabooga/text-generation-webui/pull/5677, porting Koboldcpp implementation authored by pi6am: https://github.com/LostRuins/koboldcpp/pull/982
    LLAMA_API struct llama_sampler * llama_sampler_init_dry(
            const struct llama_vocab *  vocab,
                             int32_t    n_ctx_train,
                               float    dry_multiplier,
                               float    dry_base,
                             int32_t    dry_allowed_length,
                             int32_t    dry_penalty_last_n,
                          const char ** seq_breakers,
                              size_t    num_breakers);

    /// adaptive-p: select tokens near a configurable target probability over time.
    ///
    /// the adaptive-p sampler transforms the token probability distribution to favor tokens
    /// that fall near a user-configurable probability target.
    ///
    /// internally, the sampler maintains an exponential moving average of the *ORIGINAL*
    /// probabilities of selected tokens at each sampling step. it uses this EMA to compute an
    /// adapted target probability at each sampling step, thus maintaining the desired target
    /// probability over time.
    ///
    /// adaptive-p selects a token ID rather than just mutating candidates, so it must be last
    /// in the sampler chain (like mirostat, dist, greedy).
    ///
    /// only mild truncation before this sampler is recommended. we suggest applying min-p
    /// before adaptive-p as the only other active sampler in the chain.
    ///
    /// @param target select tokens near this probability (valid range 0.0 to 1.0; negative = disabled)
    /// @param decay  EMA decay for adaptation; history ≈ 1/(1-decay) tokens (valid range 0.0 - 0.99)
    /// @param seed   RNG seed
    ///
    /// ref: https://github.com/ggml-org/llama.cpp/pull/17927
    ///
    LLAMA_API struct llama_sampler * llama_sampler_init_adaptive_p(
                               float   target,
                               float   decay,
                            uint32_t   seed);

    LLAMA_API struct llama_sampler * llama_sampler_init_logit_bias(
                             int32_t   n_vocab,
                             int32_t   n_logit_bias,
              const llama_logit_bias * logit_bias);

    // this sampler is meant to be used for fill-in-the-middle infilling
    // it's supposed to be used after top_k + top_p sampling
    //
    // 1. if the sum of the EOG probs times the number of candidates is higher than the sum of the other probs -> pick EOG
    // 2. combine probs of tokens that have the same prefix
    //
    // example:
    //
    // - before:
    //   "hel":   0.5
    //   "hell":  0.2
    //   "hello": 0.1
    //   "dummy": 0.1
    //
    // - after:
    //   "hel":   0.8
    //   "dummy": 0.1
    //
    // 3. discard non-EOG tokens with low prob
    // 4. if no tokens are left -> pick EOT
    //
    LLAMA_API struct llama_sampler * llama_sampler_init_infill(const struct llama_vocab * vocab);

    // Returns the seed used by the sampler if applicable, LLAMA_DEFAULT_SEED otherwise
    LLAMA_API uint32_t llama_sampler_get_seed(const struct llama_sampler * smpl);

    /// @details Sample and accept a token from the idx-th output of the last evaluation
    //
    // Shorthand for:
    //    const auto * logits = llama_get_logits_ith(ctx, idx);
    //    llama_token_data_array cur_p = { ... init from logits ... };
    //    llama_sampler_apply(smpl, &cur_p);
    //    auto token = cur_p.data[cur_p.selected].id;
    //    llama_sampler_accept(smpl, token);
    //    return token;
    // Returns the sampled token
    LLAMA_API llama_token llama_sampler_sample(struct llama_sampler * smpl, struct llama_context * ctx, int32_t idx);

    // TODO: extend in the future
    //LLAMA_API void llama_decode_with_sampler(struct llama_context * ctx, struct llama_sampler * smpl, struct llama_batch batch, ...);

    //
    // Model split
    //

    /// @details Build a split GGUF final path for this chunk.
    ///          llama_split_path(split_path, sizeof(split_path), "/models/ggml-model-q4_0", 2, 4) => split_path = "/models/ggml-model-q4_0-00002-of-00004.gguf"
    //  Returns the split_path length.
    LLAMA_API int32_t llama_split_path(char * split_path, size_t maxlen, const char * path_prefix, int32_t split_no, int32_t split_count);

    /// @details Extract the path prefix from the split_path if and only if the split_no and split_count match.
    ///          llama_split_prefix(split_prefix, 64, "/models/ggml-model-q4_0-00002-of-00004.gguf", 2, 4) => split_prefix = "/models/ggml-model-q4_0"
    //  Returns the split_prefix length.
    LLAMA_API int32_t llama_split_prefix(char * split_prefix, size_t maxlen, const char * split_path, int32_t split_no, int32_t split_count);

    // Print system information
    LLAMA_API const char * llama_print_system_info(void);

    // Set callback for all future logging events.
    // If this is not called, or NULL is supplied, everything is output on stderr.
    // The logger state is global so these functions are NOT thread safe.
    LLAMA_API void llama_log_get(ggml_log_callback * log_callback, void ** user_data);
    LLAMA_API void llama_log_set(ggml_log_callback   log_callback, void *  user_data);

    //
    // Performance utils
    //
    // NOTE: Used by llama.cpp examples/tools, avoid using in third-party apps. Instead, do your own performance measurements.
    //

    struct llama_perf_context_data {
        // ms == milliseconds
        double t_start_ms;  // absolute start time
        double t_load_ms;   // time needed for loading the model
        double t_p_eval_ms; // time needed for processing the prompt
        double t_eval_ms;   // time needed for generating tokens

        int32_t n_p_eval;   // number of prompt tokens
        int32_t n_eval;     // number of generated tokens
        int32_t n_reused;   // number of times a ggml compute graph had been reused
    };

    struct llama_perf_sampler_data {
        double t_sample_ms; // time needed for sampling in ms

        int32_t n_sample;   // number of sampled tokens
    };

    LLAMA_API struct llama_perf_context_data llama_perf_context      (const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_print(const struct llama_context * ctx);
    LLAMA_API void                           llama_perf_context_reset(      struct llama_context * ctx);

    // NOTE: the following work only with samplers constructed via llama_sampler_chain_init
    LLAMA_API struct llama_perf_sampler_data llama_perf_sampler      (const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_print(const struct llama_sampler * chain);
    LLAMA_API void                           llama_perf_sampler_reset(      struct llama_sampler * chain);

    //
    // training
    //

    // function that returns whether or not a given tensor contains trainable parameters
    typedef bool (*llama_opt_param_filter)(const struct ggml_tensor * tensor, void * userdata);

    // always returns true
    LLAMA_API bool llama_opt_param_filter_all(const struct ggml_tensor * tensor, void * userdata);

    struct llama_opt_params {
        uint32_t n_ctx_train; // assumed context size post training, use context size specified in llama_context if 0

        llama_opt_param_filter param_filter; // callback for determining which tensors contain trainable parameters
        void * param_filter_ud;              // userdata for determining which tensors contain trainable parameters

        ggml_opt_get_optimizer_params get_opt_pars; // callback for calculating optimizer parameters
        void * get_opt_pars_ud;                     // userdata for calculating optimizer parameters

        enum ggml_opt_optimizer_type optimizer_type;
    };

    LLAMA_API void llama_opt_init(struct llama_context * lctx, struct llama_model * model, struct llama_opt_params lopt_params);

    LLAMA_API void llama_opt_epoch(
            struct llama_context    * lctx,
            ggml_opt_dataset_t        dataset,
            ggml_opt_result_t         result_train,
            ggml_opt_result_t         result_eval,
            int64_t                   idata_split,
            ggml_opt_epoch_callback   callback_train,
            ggml_opt_epoch_callback   callback_eval);

#ifdef __cplusplus
}
#endif

#endif // LLAMA_H
