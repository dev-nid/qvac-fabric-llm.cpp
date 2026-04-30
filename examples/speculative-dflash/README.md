# llama-speculative-dflash

End-to-end CLI for **DFlash** block-parallel speculative decoding.

DFlash is a "block diffusion" speculative decoding method introduced by
Chen et al. (2026). A small DFlash *draft* model produces an entire block of
candidate tokens in a **single** forward pass (using mask-token slots),
conditioned on intermediate hidden states of the *target* model. The target
then verifies the block in parallel and the longest matching prefix is
accepted (plus one bonus target token).

This example demonstrates the full pipeline ported from the reference Python
implementation at <https://github.com/z-lab/dflash>.

## Build

```bash
cmake -S . -B build
cmake --build build --target llama-speculative-dflash -j
```

## Usage

```bash
./build/bin/llama-speculative-dflash \
    -m  Qwen3-8B.gguf \
    -md Qwen3-8B-DFlash.gguf \
    -p  "How many positive whole-number divisors does 196 have?" \
    --n-predict 256
```

Optional knobs:

* `--draft-max <int>` (`-draft-max`): override the block size baked into the
  DFlash draft GGUF. Defaults to whatever the draft was trained for (typically 16).
* `--n-predict <int>`: maximum number of tokens to generate post-prompt (default 256).
* `-cnv`, `-no-cnv`: chat / one-shot mode flags from the standard CLI.

The driver always uses **greedy** sampling for the draft (matching the Python
reference) but full `common_sampler` flags (top-k, top-p, temp, ...) apply to
the **target** verification step.

## Converting a DFlash draft to GGUF

The HuggingFace converter understands DFlash drafts:

```bash
python convert_hf_to_gguf.py \
    z-lab/Qwen3-8B-DFlash-b16 \
    --outfile Qwen3-8B-DFlash.gguf
```

The resulting GGUF carries the `dflash.block_size`, `dflash.mask_token_id`,
`dflash.target_layer_ids`, and `dflash.num_target_layers` metadata used by
the `LLM_ARCH_DFLASH` arch registered in `src/llama-arch.{h,cpp}`.

## How it works (one block)

```
last accepted token committed at position p_committed.

draft input  = [last_accepted, MASK, MASK, ..., MASK]    (length bs)
draft output = lm_head(DFlash(draft_input, target_hidden))
draft_tokens = argmax(draft_output[1..bs-1])             (length bs-1)

verify input = [last_accepted, draft_tokens[1], ..., draft_tokens[bs-1]]   (length bs)
target_logits, target_hidden = target(verify_input)
target_tokens = sample(target_logits)                    (length bs)

accepted = longest k such that draft_tokens[i+1] == target_tokens[i]  for i < k
commit   = draft_tokens[1..accepted] + [target_tokens[accepted]]      ("bonus token")
```

The target's KV cache is then trimmed back to `p_committed + accepted + 1`,
the draft's KV cache is dropped, and the next iteration begins.

See `core_architecture/01_architecture_analysis.md` and
`core_architecture/02_port_plan.md` at the repo root for the detailed port
narrative.

## Status

The driver and the `LLM_ARCH_DFLASH` arch are fully wired into the build.
The cross-attention with frozen target hidden states is currently scaffolded
inside `src/models/dflash.cpp` (the draft graph is structurally Qwen3 with
non-causal attention); plumbing the full per-layer `target_hidden ->
K_ctx,V_ctx` projection is tracked as TODO inside that file. With the
current scaffolding the algorithm produces correct output but with reduced
speed-up; the speed-up will materialise once the cross-attention input
class is landed.
