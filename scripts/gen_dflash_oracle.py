#!/usr/bin/env python3
"""
gen_dflash_oracle.py — generate a deterministic numerics-oracle artifact for
the DFlash drafter graph.

Loads the z-lab `DFlashDraftModel` PyTorch reference (e.g. `Qwen3-4B-DFlash-b16`),
runs ONE forward pass on a deterministic synthetic input (random noise embedding
+ random pre-fc target_hidden), and saves the inputs + the expected post-norm
output to a single binary artifact that `tests/test-dflash-numerics.cpp` then
loads to validate the C++ DFlash decoder graph against.

The oracle bypasses `tok_embed` lookup on both sides: the test feeds the
random noise embedding directly into the C++ decoder via a `llama_batch` whose
`embd` field is set (no tokens). This isolates the DRAFT graph (per-layer
projections, RoPE, k_norm, cross-attn, self-attn, MLP, residuals, final norm)
from the bound-target embedding-lookup table, which is a separate concern
covered by the strong-correctness CTest.

Output layout: a single `.bin` file with a fixed header, then three contiguous
float32 blocks (noise, target_hidden, expected). Documented in the C++ test.

Usage (from the repo root):

    .venv/bin/python3 scripts/gen_dflash_oracle.py \
        --model /home/dev/devnid/dflash/models/Qwen3-4B-DFlash-b16 \
        --out   /tmp/dflash_oracle/oracle.bin

The C++ test consumes the artifact via:

    DFLASH_TEST_TARGET=...gguf  DFLASH_TEST_DRAFT=...gguf \
    DFLASH_TEST_ORACLE=/tmp/dflash_oracle/oracle.bin \
    ctest --test-dir build -R test-dflash-numerics --output-on-failure
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import types
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModel


# Header layout (little-endian, 64 bytes):
#   uint32 magic     = 'DORC'  (0x43524F44 in LE)
#   uint32 version   = 1
#   uint32 ctx_len            (cross-attn target context length, in tokens)
#   uint32 block_size         (drafter intra-block length, == 16 for the b16 ckpt)
#   uint32 hidden             (per-token hidden dim, == 2560)
#   uint32 n_target_layers    (number of captured target layers, == 5)
#   uint32 fc_in              (== n_target_layers * hidden)
#   uint32 seed               (RNG seed used for the deterministic inputs)
#   uint32 dtype_ref          (1 = bf16, 2 = fp32; the dtype the reference ran in
#                              before the f32 conversion stored on disk)
#   uint32 reserved[7]        (zero-padded; future use)
HEADER_MAGIC = 0x43524F44  # 'DORC'
HEADER_VERSION = 1
HEADER_FMT = "<IIIIIIIIIIIIIIII"  # 16 uint32 = 64 bytes
HEADER_SIZE = struct.calcsize(HEADER_FMT)
assert HEADER_SIZE == 64


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--model", required=True,
                    help="Path to the DFlash drafter model directory "
                         "(e.g. /home/dev/devnid/dflash/models/Qwen3-4B-DFlash-b16)")
    ap.add_argument("--out", required=True,
                    help="Output .bin path (parent dir created if missing)")
    ap.add_argument("--ctx-len", type=int, default=64,
                    help="Cross-attn context length (default: 64)")
    ap.add_argument("--seed", type=int, default=42,
                    help="RNG seed for the deterministic synthetic inputs (default: 42)")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"],
                    help="Reference dtype to run the draft in (default: bf16, "
                         "matching the published checkpoint's training dtype)")
    args = ap.parse_args()

    model_dir = Path(args.model).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Load the DFlash drafter
    # -----------------------------------------------------------------
    # The model card ships dflash.py + modeling_dflash.py with the auto_map
    # pointing AutoModel at DFlashDraftModel; trust_remote_code=True opts in.
    print(f"Loading drafter from {model_dir} ...")
    ref_dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_dir,
        config=config,
        torch_dtype=ref_dtype,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()

    block_size = int(model.block_size)
    hidden = int(config.hidden_size)
    target_layer_ids = list(model.target_layer_ids)
    n_target_layers = len(target_layer_ids)
    fc_in = n_target_layers * hidden
    ctx_len = int(args.ctx_len)
    seed = int(args.seed)

    print(f"  block_size       = {block_size}")
    print(f"  hidden           = {hidden}")
    print(f"  target_layer_ids = {target_layer_ids}  (n={n_target_layers})")
    print(f"  fc_in            = {fc_in}")
    print(f"  ctx_len          = {ctx_len}")
    print(f"  seed             = {seed}")
    print(f"  ref_dtype        = {ref_dtype}")

    # -----------------------------------------------------------------
    # Deterministic synthetic inputs
    # -----------------------------------------------------------------
    # Reasoning for the magnitudes:
    # * noise: target.embed_tokens output is centred near 0 with std ~0.02 for
    #   Qwen3-4B (initializer_range=0.02). Use the same scale so the first
    #   layer's input distribution roughly matches a real decode step.
    # * target_hidden: each captured target hidden is a post-MLP+residual
    #   activation; std varies layer-to-layer but is O(1). The fc layer
    #   normalises this internally (hidden_norm), so the absolute scale of
    #   the input is not load-bearing for the test — std=0.5 keeps the
    #   pre-norm activations in a sensible range.
    torch.manual_seed(seed)
    noise = torch.randn(1, block_size, hidden, dtype=ref_dtype) * 0.02
    target_hidden = torch.randn(1, ctx_len, fc_in, dtype=ref_dtype) * 0.5

    # position_ids must cover the FULL [ctx + q] range, not just q. The
    # attention path concatenates k_ctx (length ctx_len) and k_noise (length
    # block_size), then applies RoPE over the combined ctx_len + block_size
    # positions. `apply_rotary_pos_emb` (dflash.py:22) takes the LAST q_len
    # slice of cos/sin for q and the full slice for k. The C++ side feeds
    # the same global positions via `pos_ctx`+`pos` in
    # llm_build_dflash::self_attn.
    position_ids = torch.arange(0, ctx_len + block_size, dtype=torch.long).unsqueeze(0)

    # -----------------------------------------------------------------
    # Reference forward
    # -----------------------------------------------------------------
    print("Running reference forward ...")
    with torch.inference_mode():
        out_hidden = model(
            position_ids=position_ids,
            noise_embedding=noise,
            target_hidden=target_hidden,
            past_key_values=None,
            use_cache=False,
        )

    # `model.forward` returns norm(hidden_states) directly (see dflash.py:191).
    # Shape: [1, block_size, hidden].
    assert out_hidden.shape == (1, block_size, hidden), (
        f"unexpected ref output shape: {tuple(out_hidden.shape)}")

    print(f"  out shape={tuple(out_hidden.shape)} dtype={out_hidden.dtype} "
          f"mean={out_hidden.float().mean().item():.6g} "
          f"std={out_hidden.float().std().item():.6g} "
          f"|max|={out_hidden.float().abs().max().item():.6g}")

    # -----------------------------------------------------------------
    # Serialise as fp32 contiguous row-major
    # -----------------------------------------------------------------
    noise_f32 = noise.to(torch.float32).contiguous().cpu().numpy()
    target_f32 = target_hidden.to(torch.float32).contiguous().cpu().numpy()
    out_f32 = out_hidden.to(torch.float32).contiguous().cpu().numpy()

    assert noise_f32.size == block_size * hidden
    assert target_f32.size == ctx_len * fc_in
    assert out_f32.size == block_size * hidden

    dtype_ref_id = 1 if ref_dtype is torch.bfloat16 else 2
    header = struct.pack(
        HEADER_FMT,
        HEADER_MAGIC, HEADER_VERSION,
        ctx_len, block_size, hidden, n_target_layers, fc_in, seed,
        dtype_ref_id,
        0, 0, 0, 0, 0, 0, 0,  # reserved
    )

    with open(out_path, "wb") as f:
        f.write(header)
        f.write(noise_f32.tobytes())
        f.write(target_f32.tobytes())
        f.write(out_f32.tobytes())

    payload_bytes = (noise_f32.size + target_f32.size + out_f32.size) * 4
    total_bytes = HEADER_SIZE + payload_bytes
    print(f"Wrote {out_path}")
    print(f"  total bytes : {total_bytes:>10}  "
          f"(header {HEADER_SIZE} + noise {noise_f32.size*4} + "
          f"target {target_f32.size*4} + expected {out_f32.size*4})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
