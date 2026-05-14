#include "common.cuh"
#include "ggml.h"

void ggml_cuda_op_gated_delta_net(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// DFlash GDN-with-history: same kernel as ggml_cuda_op_gated_delta_net,
// but the dst tensor packs an additional per-token state-history region
// of shape [S_v, S_v, H_v, n_tokens, n_seqs] after the attn-output and
// final-state
// regions. Used by the speculative driver to roll back the recurrent state
// to state[K-1] on partial draft acceptance via a separate fixup graph.
void ggml_cuda_op_gated_delta_net_with_history(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// DFlash state-history fixup: strided copy that selects
// state_history[..., k_index, :] (per-head recurrent state at a chosen
// token-position) and writes it into a [S_v, S_v, H_v, n_seqs] result tensor.
// k_index is an I32 scalar tensor (1 element).
void ggml_cuda_op_gated_delta_net_state_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);

// DFlash conv-state fixup: picks `conv_state_rows` consecutive rows of
// conv_history along dim 0 starting at offset (k_index + 1), and writes
// them into a [conv_state_rows, conv_channels, n_seqs]
// result tensor. When k_index < 0, copies fallback → result instead
// (no-op rollback path).
void ggml_cuda_op_dflash_conv_state_history_select(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
void ggml_cuda_op_dflash_conv_state_history_select_tree(ggml_backend_cuda_context & ctx, ggml_tensor * dst);
