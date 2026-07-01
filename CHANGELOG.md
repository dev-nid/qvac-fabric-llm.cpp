# Changelog

## [b9518] - 2026-06-29

Rebase onto upstream `ggml-org/llama.cpp` tag `b9518`
(previous base: `b9341`).

- Upstream window: `b9341` -> `b9518` (177 commits)
- New downstream commits since `tether/temp-9341`: ~55 (excludes 260 carry-over
  commits already present at the previous rebase point and 8 squash/fixup
  no-ops)

## Upstream (b9341 -> b9518)

Summary of upstream changes pulled in by this rebase. 

#### New Model / Architecture Support

- `model: add Mellum architecture (#23966)`
- `model: support granite multilingual embeddings R2` for `ibm-granite/granite-embedding-{97,311}m-multilingual-r2` (#22716)
- `model: Add EXAONE 4.5 implementations (#21733)`
- `model: support for DeepseekV32ForCausalLM` with generic DeepSeek Sparse Attention (DSA) implementation (#23346)
- `StepFun 3.5 MTP (#23274)`
- `convert: support Step3.7-Flash (#23845)`
- `convert: Fix Gemma 4 Unified conversion (#24118)`
- `convert: add FP8 to Q8 conversion (#23250)`
- `convert: add MiniCPM5 tokenizer support (#23384)`
- `vocab: add normalizer.lowercase support to WPM (#23899)`
- `vocab: add tokenizer support for jina-embeddings-v2-base-zh (#18756)`
- `vocab: support tokenizer for LFM2.5-8B-A1B (#23826)`
- `chat: add Granite 4.1 chat template (#23518)`
- `tests: add support for qwen3 SSM archs (#24031)`
- `qwen35: use post-norm hidden state for MTP (#24025)`

#### Multimodal (mtmd)

- `mtmd: Add DeepSeekOCR 2 Support (#20975)`
- `mtmd: enable non-causal vision for gemma 4 unified (#24082)`
- `mtmd, model: allow skip build_vit() (#24077)`
- `mtmd: fix Gemma 4 unified FPE (#24088)`
- `mtmd: handle Gemma 4 audio projector embedding size (#24091)`
- `mtmd: fix gemma 4 projector pre_norm (#23822)`
- `mtmd: fix gemma 4 audio rms norm eps (#23815)`
- `mtmd: n_head_kv defaults to n_head (#23782)`
- `mtmd-debug: add color and rainbow mode (#23829)`
- `arg: removed unnecessary mmproj download when users pass --no-mmproj (#23425)`

#### Server / WebUI

- `server: real-time reasoning interruption via control endpoint (#23971)`
- `server: add SSE ping interval (#24013)`
- `server: in SSE mode, send HTTP headers when slot starts (#23884)`
- `server, ui: Add support for HTTP ETags in llama-server (#23701)`
- `server: handle If-None-Match weak ETags (#23916)`
- `server: avoid unnecessary checkpoint restore when new tokens are present (#24110)`
- `server: disable on-device spec checkpoints (#24108)`
- `server: bump timeout to 3600s (#23842)`
- `server-bench: add speed-bench for speculative decoding benchmarking (#23869)`
- `ui: Mermaid Diagrams in chat + interactive preview (#24032)`
- `ui: added single line reasoning preview (#23601)`
- `ui: Add Thinking mode toggle with reasoning effort levels (#23434)`
- `webui: fix tool selector toggle/counter, key tools by stable identity (#24065)`
- `webui: [a11y] fix keyboard navigation issues in chat interface and sidebar (#23132)`
- `webui: add custom CSS injection via config (#23904)`

#### KV Cache / Inference

- `kv-cache: SWA checkpoints store only non-masked cells (#23981)`
- `llama: use f16 mask for FA to save VRAM (#23764)`
- `llama: limit max outputs of llama_context (#23861)`
- `llama: deprecate llama_set_warmup (#24009)`
- `llama: only use one iGPU device by default (#23897)`
- `llama: do not skip iGPU when only RPC devices are present (#23868)`
- `speculative: fix n_outputs_max and remove draft-simple auto-enable (#23988)`
- `Support -fa auto in llama-bench (#23714)`

#### Backends

- Vulkan: optimize conv2d and implement coopmat1 support (#22620); cooperative-matrix-decode-vector for faster matmul (#23541); MUL_MAT_VEC 4 K per iteration for F16/32 (#22887); Flash Attention support for BFloat16 KV cache (#23420); Block-load Q3_K/Q6_K and subtract on 32b ints (#23056); fast path for walsh-hadamard transform (#23687); reduce host memory lock contention (#23376).
- Metal: template GLU kernels for f16/f32 (#23882); restore im2col for large kernels (#23901); reduce rset heartbeat 500ms -> 5ms (#24074).
- CUDA: route batch>=4 quantized matmul to MMQ on AMD MFMA hardware (#23227); MMVQ_PARAMETERS_TURING (#23729); KQ mask offset int overflow fix in fattn MMA (#23610); reserve quantize KV-cache at startup (#23907); PDL guards and CTK >= 12.3 restriction (#23530, #23742, #23825).
- SYCL: Q4_1/Q5_0/Q5_1 in Flash-attention (#23812); more types in GET_ROWS (#23710); Q3_K mul_mat reorder (#23725).
- OpenCL: q5_0/q5_1 (#23548); bf16 -> f16 (#23839); OP_GATED_DELTA_NET (#23312); flat q4_K/q6_K gemv for very large M (#24006).
- Hexagon: MUL_MAT, MUL_MAT_ID, FLASH_ATTN, GDN cleanup and optimizations (#23989); basic/generic op fusion + RMS_NORM+MUL fusion (#23835); HMX FA/MM refresh (#23796); OP_GATED_DELTA_NET K>1 (#23531); Q4_1 MUL_MAT/MUL_MAT_ID (#23647); CONCAT (#23648); gelu_quick (#24007).
- WebGPU: FlashAttention refactor + standardized quantization (#23834); q4_0/q8_0 SET_ROWS (#23760).
- ggml-cpu: WASM SIMD128 q4_1_q8_1 vec dot (#22209); RVV vec dot at higher VLENs (#22754); runtime SVE width in FWHT (#24059); LSX support (#23798); Arm SVE fix in vec.h/vec.cpp (#22841).

#### Common / API

- `common: support manually triggering the reasoning budget end sequence (#23949)`
- `common: fix env names to all have LLAMA_ARG_ prefix (#23778)`
- `common: fix state save in common_prompt_batch_decode (#23468)`
- `arg: Add LLAMA_ARG_API_KEY_FILE environment variable for --api-key-file (#23167)`
- `download: add option to skip_download (#23059)`
- `arg: fix double mtp downloads (#24128)`

#### Build / CI / Vendor

- `cmake: skip cvector-generator and export-lora when CPU backend is disabled (#24053)`
- `build: use umbrella Headers directory for XCFramework module map (#23974)`
- `pyproject: add conversion folder and update dependencies (#23746)`
- `vendor: update cpp-httplib to 0.46.1 (#23980)` and `0.46.0 (#23650)`
- `update BoringSSL to 0.20260526.0 (#23794)`
- `docker: add ZenDNN Dockerfile (#23716)`
- `nix: add nix-nodejs facilities to build Web UI (#23846)`
- `app: add llama update self updater (#23865)`
- `app: move licences to llama-app (#23824)` and `app: improve help output (#23805)`
- Numerous CI consolidation PRs: moving jobs to self-hosted, removing SYCL/CANN/WASM builds, MSVC/macOS-26 runner updates.

## Downstream

### Added 

#### MoE Expert LoRA Finetune (Qwen3.5 / Qwen3.6 / Gemma 4 MoE)

The previous rebase already shipped LoRA finetuning. This release adds the
backend support needed to fine-tune **MoE experts** end-to-end:

- `899b5111f llama-lora-training: MoE expert LoRA & tighten module matcher`
- `d68be6ce3 llama-graph: training-mode workarounds for MoE backward in build_moe_ffn`
- `12d2f4aeb ggml: gemma 4 chat template parsing for LoRA finetune`
- `1b3958bc5 docs: update lora finetuning documentation`

#### Backward Ops (training prerequisites for MoE / Gemma 4 / SSM graphs)

New `_BACK` / gradient ops added across CPU, Vulkan and Metal so the MoE-expert
LoRA pipeline above is actually differentiable:

- `MUL_MAT_ID` backward (back_a / back_b):
  - `c5e1f95ed ggml: add MUL_MAT_ID backward (back_a, back_b)`
  - `ae77d8e3d ggml-vulkan: implement MUL_MAT_ID_BACK_A`
  - `de2df6abc ggml-vulkan: implement MUL_MAT_ID_BACK_B`
  - `2bf806896 ggml-metal: implement gelu_back, l2_norm_back, mul_mat_id_back_a/b backward ops`
- `GET_ROWS_BACK` generalized for MoE router gradients:
  - `a538f6a51 ggml: generalize get_rows_back to N-D tensors`
  - `685ece802 tests: fix crash for GET_ROWS_BACK`
- `SSM_CONV_BACK` (SX + C):
  - `5011b9207 ggml: add SSM_CONV_BACK_SX and SSM_CONV_BACK_C ops`
  - `248c2cddb ggml-cpu: implement SSM_CONV_BACK_SX and SSM_CONV_BACK_C`
  - `d8365888d ggml-vulkan: implement SSM_CONV_BACK_SX and SSM_CONV_BACK_C`
  - `a724e5448 ggml-metal: implement ssm_conv_back_sx/_c and gated_delta_net_back backward ops`
- `GATED_DELTA_NET_B`:
  - `42fa75426 ggml-vulkan: implement GATED_DELTA_NET_B`
- Unary / shape backward ops:
  - `6a80fe9d7 ggml: add GELU backward` + `74531cf89` vulkan impl + `7e5860603` test
  - `b6142e3d8 ggml: add GGML_OP_SIGMOID_BACK op` + `d4c896912` cpu + `a63b4a4eb` vulkan + `02df6d51b` test
  - `4b25218f5 ggml: add L2_NORM_BACK op with CPU implementation` + `d272217ff` vulkan + `b9543dfba` stop-grad fix + `71018d488` test
  - `d37344213 ggml: Add TANH backward pass for cpu`
  - `31e05257d ggml: Add concat cpu backward pass`
  - `4a394d0ed ggml: Add stop-gradient backward for SSM, L2_NORM, GATED_DELTA_NET ops`
- Tests + RPC bookkeeping:
  - `bc38e0b61 test-backend-ops: add SSM_CONV_BACK tests and enable ssm_conv grad check`
  - `92a329d6d / 54ca6a2dc / ecacdd033` MUL_MAT_ID_BACK A/B/perf tests
  - `7b1a9b75b ggml-rpc: account for new backward ops (GGML_OP_COUNT 100->105), bump proto patch version`

#### Misc

- `0e36ea55c kv-cache: cover GGSQ M-RoPE ext restore`
- `7598cc382 common: Split jinja headers, add aux functions`

### Fixed

#### Vulkan

- `e763aca4e ggml-vulkan,common,cmake: fix TurboQuant copy_to_quant codebook swap, restore reasoning_budget, fix llama install include path`
- `caf41d8ec ggml-vulkan: align host buf alloc to minMemoryMapAlignment`
- `e32e45620 ggml-vulkan, llama, common, mtmd: SSM_CONV dispatch fix`
- `ea0bf3203 ggml-vulkan: remove TQ MUL_MAT -> CPU fallback on Mali/Adreno coopmat1` (workaround no longer needed after the matmul fixes upstream)

#### OpenCL / Adreno

- `6d9210b02 ggml-opencl: serialize lazy device-context and kernel-program init to fix a multi-instance clBuildProgram race that corrupts the Adreno driver heap`

#### ROCm / HIP

- `66ca71af3 hip: allow-list gated_linear_attn_f32<64> VGPRs`

#### Multimodal

- `dcd05c8b8 mtmd-clip: drop the Gemma 4 set_warmup_n_tokens(256) override` (re-applied on top of the new b9518 mtmd code)

#### Tests / CI

- `6966222e7 tests: download test model with retry + backoff`
- `2ee5f835b ui: Uprev playwright to 1.60 and add timeouts`
- `010e1b1f7 ci fix: download LFS blobs via curl in tokenizer test`
- `6fe8a2301 ci: tolerate Vulkan T4 driver segfault-on-exit in tests`
- `65843e8b0 ci: fix ccache key to include CPU feature hash`
- `645abd843 ci: disable native CPU optimizations for low-perf builds`
- `6dea66950 ci: Reenable sycl and cann jobs`
- `f4d91bbbb remove_cache`

### Changed

- `7ca6374fb ggml-cpu: Add portable CPU prefetch hint macro`
- `3a7689f9a cmake: install convert_hf_to_gguf.py alongside vulkan_profiling_analyzer.py`
- TurboQuant import re-organized into two focused commits during rebase (no
  behavior change vs `tether/temp-9341`'s squashed import):
  - `33d96b7eb ggml,ggml-cpu: TurboQuant core/cpu api changes`
  - `beddffc2f ggml-vulkan: TurboQuant vulkan backend and shader integration`

### Removed

- `ea0bf3203` removed the TQ MUL_MAT CPU-fallback path on Mali/Adreno coopmat1
  (no longer needed; see Fixed > Vulkan)
