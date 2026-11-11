# CLAUDE.md — UNF-nano-llms-training

Guidance for Claude Code in this repo. The workspace conventions in `~/Code/CLAUDE.md` and
`006-research-prototypes/CLAUDE.md` apply; this file wins where they conflict. `README.md` is the
(very long) architecture catalogue of the `nano_moe` stack; this file is the working map.

## What is here

| Path | What it is |
|---|---|
| `nano_moe/` | Python package (`python -m nano_moe.<module>`, see `Makefile` targets `demo-*`). Models (MoE, phase/spectral transformers, CLIP, diffusion, VAE, ...), training utilities, and a partial port of the LLM inference helpers: `inference.py` (`generate_adaptive`, `generate_active_inference`, `speculative_decoding`, ARC System-2), `training/rl.py` (`RewardModel`, `grpo_step`), `training/pruning.py`, `training/distillation.py`. No `NanoGPT` class lives in the package yet. |
| `boilerplates/nano_moe_flashattention_mamba_bpe_grpo.py` | 16.5k-line Colab export of the notebook below. The source of the language-model stack. |
| `notebooks/NANO_MOE_FLASHATTENTION_MAMBA_BPE_GRPO.ipynb` | The notebook. Byte-identical to `006-research-prototypes/NANO_MOE_FLASHATTENTION_MAMBA_BPE_GRPO.ipynb` (the root copy). |
| `boilerplates/*.py` (others) | Other single-file references (phase transformer, reflection layer, diffusion/CLIP, ARC co-evolution). |
| `benchmarks/`, `examples/`, `tests/` | SFPT-vs-transformer benchmark, demos, one integration test. |

No `pyproject.toml` yet. The boilerplate needs `torch`, `regex`, `pandas` (TinyStories CSVs) and
optionally `rich`.

## Lineage map (LM stack)

`root notebook` = `notebooks/NANO_MOE_...ipynb` → exported as `boilerplates/nano_moe_..._grpo.py`.
The `.py` is now the edited copy: the three fixes below were made there only; both `.ipynb` copies
still carry the original code.

Sections of the `.py`, in notebook order (each re-declares its own classes; later sections supersede
earlier ones, nothing imports across sections):

| Lines (approx.) | Section |
|---|---|
| 1-1950 | MoE CNN 1-3 (vision MoE, experiment tracker/report) |
| 1951-2908 | NanoGPT MoE (char-level) + CLI |
| 2909-4574 | BPE NanoGPT MoE; non-uniform experts variant |
| 4575-5117 | Flash/SDPA + `NanoMamba` |
| 5118-6236 | Reward model + GRPO |
| 6237-10167 | "Unified all": NanoGPT/NanoMamba + tokenizers + MoE + RM + GRPO + `TrainingMonitor` (4 iterations) |
| 10168-11896 | BPE NanoGPT + 3 inference modes (+ longer training) |
| 11897-13590 | Full stack with TTA / Active Inference / System-2, full pipelines + inference suite |
| 13591-15360 | Energy-based training (NCE/InfoNCE, sampled softmax) + adaptive computation (easy/hard tokens) |
| 15361-16493 | **"Cells 1-5"** of the earlier chat: Cell 1 data + MinBPE (`BasicTokenizer`, `RegexTokenizer`, GPT-4 split); Cell 2 `NanoGPT` (SDPA causal attention), `MoEFFN` (shared + routed experts, Gumbel `TopKRouter`, balance/entropy aux, EMA usage rescue), `LoRALinear`/`inject_lora`, `KVCache`, pruning/quant, checkpoint I/O; Cell 3 `train_xent`, distillation, `RewardModel`, `grpo_step`, `generate_*`, `apply_pruning`, `apply_quantization`, argparse `main`, `run_cli(cmd)`; Cell 4 commented examples; Cell 5 driver (train MoE + 20% prune, then inference variants). |

Files discussed in the earlier chat that are **not on disk** (chat-only; do not assume they exist):
`full_pipeline_demo.py`, `ultimate_gpt_pipeline.py`, the "Sections 1-4" pipeline (`TinyBPETokenizer`,
`GPTBackbone`, `PonderNetAdapter`, `run_full_pipeline`), `gpt_pipeline.py`, `full_demo_script.py`,
`complete_pipeline.py`.

## Adaptive-inference techniques: present vs unbuilt

All implementations are toys that illustrate the idea; treat speed/memory claims as unverified until
benchmarked, and label toy versions "toy" in output.

| # | Technique | Status on disk |
|---|---|---|
| 1 | Quantised inference | Present: `torch.quantization.quantize_dynamic` (real int8, CPU only) and an fp16 cast. |
| 2 | Adaptive compute | Present as entropy-triggered resampling (`generate_adaptive`) and easy/hard-token adaptive computation (energy section). **PonderNet (learned halting) is unbuilt.** |
| 3 | Active inference | Present (toy): candidates scored by NLL minus entropy; "free energy" is an analogy. |
| 4 | System-2 reasoning | Present (toy): generate, critique prompt, refine, pick by mean log-prob. |
| 5 | Speculative decoding | Present (toy): 1-token draft, threshold accept; no k-token draft, no rejection-sampling rule. |
| 6 | Toolformer | **Unbuilt.** |
| 7 | Meta-learning (MAML) | **Unbuilt.** |
| 8 | Fast weights (Hebbian) | **Unbuilt.** |
| 9 | Differentiable plasticity | **Unbuilt.** |

Also present: MoE, LoRA, distillation, reward model + GRPO, pruning, KV cache, energy-based training,
NanoMamba. Not present: evolutionary test-time compute, token-level Ponder.

## Bug status (Cells 1-5 section)

Fixed on `chore/chat-context-import` (each reproduced first on CPU with a tiny config):
- `train_gpt_bpe` read `train_txt`/`val_txt` after unpacking into `train, val` → `NameError`.
- `infer_gpt_bpe` ignored `--ttc/--active/--system2/--speculative` (always `sample_lm`); now dispatched.
- `KVCache` int8 stored `(k/s).to(int8)` in [-1, 1] (≈ everything 0), shared K's scale with V, never
  re-scaled old entries. Round-trip error 0.86 → 0.008.

Did not reproduce:
- "Pruning breaks weight tying": with torch 2.15 `prune.remove` keeps the same `Parameter`, so
  `lm_head.weight is wte.weight` holds after pruning and after reloading the clean checkpoint.

Open (verified by reading or a tiny run, not fixed):
- **KV-cached decoding is broken beyond int8**: one `KVCache` object is passed to every block (all
  layers append into one buffer), positions restart at 0 (`arange(0, T)`), `generate_standard` feeds
  the full context *and* the cache each step (duplicated keys), SDPA runs with `is_causal=False` whenever
  a cache is present (no causal mask on prefill), and the vanilla-attention mask slice ignores the
  offset. `infer_gpt_bpe --kv_cache` therefore warns and decodes without a cache.
- Global pruning includes the tied `lm_head`, so the input embedding is pruned too (≈ the pruning
  fraction of `wte` becomes zero). `gpt_moe_with_masks.pt` (`weight_orig`/`weight_mask` keys) cannot be
  loaded into a fresh `NanoGPT`; only the clean checkpoint can.
- LoRA path calls `peft.get_peft_model(..., task_type="CAUSAL_LM")` on the non-HF `NanoGPT`; the in-file
  `inject_lora` exists and `GPTCfg.lora_*` fields are never set from the CLI.
- `grpo_step` calls `model.generate` (not defined on `NanoGPT`) and uses the policy's own log-prob as reward.
- `GPTBlock` gradient-checkpoint branch returns aux loss 0, silently disabling MoE balance losses.
- `system2_reasoning.score()` does not crop to `block_size` (assertion once prompt + generations exceed it).
- Cell 5 step 1 hard-codes `use_tinystories=True`: needs `data/tinystories_{train,val}.csv` with a `text` column.
- Earlier sections (chat-reported, not re-checked here): `ToyGPT`-style blocks without a causal mask,
  unshifted targets in the earliest demo, per-token MoE loop with wrong shapes.

## Plan: pick one canonical file

The earlier chat recommended consolidating on the Cells 1-3 `NanoGPT` stack (causal attention, shifted
targets) and porting techniques 6-9 into it. Locally, ports have been landing in the `nano_moe/` package,
so the likely shape is: move `NanoGPT` + `KVCache` + generation into `nano_moe/` (after fixing the KV
path), freeze the boilerplate as a reference export, then:
1. fix the open list above, training correctness first;
2. one CPU test per technique (shape check + one forward pass) so a sweep can run in CI;
3. real speculative decoding (separate small draft model, k proposals, accept/reject) benchmarked in tokens/s;
4. measure quantisation and pruning (size on disk, latency, val loss) instead of asserting gains;
5. only then build techniques 6-9 (Toolformer with an AST-based `math_eval`, never `eval`; MAML without
   `@torch.no_grad()` around `torch.autograd.grad`; Hebbian traces; learned plasticity `w + alpha*hebb`).

## Conventions

- Keep the boilerplate runnable from a notebook via `run_cli("...")`.
- Seed everything (`set_seed`, default 1337).
- Default demo prompts: "In the beginning,", "The quick brown fox", "Explain quantum entanglement
  simply.", "Hello world,", "What is 9*9?".

## Parked ideas (earlier chats)

- A single `complete_pipeline.py` (data → tokenizer → train → compress → inference sweep →
  `artifacts/inference_summary.json`) with every inference function returning `(text, info_dict)`.
  Local generators return id lists and the package route is preferred; revisit if a one-file demo is wanted.
