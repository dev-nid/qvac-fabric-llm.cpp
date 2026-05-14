#include "common.cuh"

void ggml_cuda_op_ssm_conv(ggml_backend_cuda_context & ctx, ggml_tensor * dst, ggml_tensor * bias_add_node = nullptr, ggml_tensor * silu_dst = nullptr);

// DFlash tree-mode parent-aware ssm_conv. Per new-token i, walks parent_ids
// K-1 times to assemble the K-tap conv window from ancestor slots (rather
// than the sequential i..i+K-1 slide). No bias / silu fusion variants — the
// Qwen3.5 graph builder calls bias-add and silu as separate ops downstream.
void ggml_cuda_op_ssm_conv_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
