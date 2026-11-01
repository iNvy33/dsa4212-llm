# CLAUDE.md

Instructions for Claude Code in this repository. Read `PLAN.md` for the full experiment plan; it is the spec.

## Project in one paragraph

Train small GPT-style language models from random initialisation on TinyStories and compare design choices under a fixed budget: 30 minutes of training wall-clock time per run on one RTX 3060 Ti (8 GB). Every result is test bits per byte (bpb). This is a university assessment: correctness, fairness between runs and reproducibility matter more than features.

## Hard rules

- **No pretrained weights, embeddings or models of any kind.** Libraries for tokenizer training (e.g. Hugging Face `tokenizers`) are fine; downloading a pretrained tokenizer or model is not.
- **Never read the test split** except in the final evaluation of a final run. Learning-rate tests, tuning and debugging use validation only.
- **Never start a run longer than 2 minutes unless I ask.** Smoke tests use tiny configs and `--budget-seconds 60` or less. I launch the 30-minute runs myself.
- **Every option must be testable in isolation:** a shape test plus an overfit-one-batch test before it is used in a timed run.
- **One change at a time.** Experiment configs are small overrides of the baseline config, nothing else.
- Ask before adding a dependency or changing the data splits, tokenizer, evaluation or timer once they work.

## Environment

- Native Windows (no WSL). Use a Python virtual environment in `.venv`. Commands must work in PowerShell; use `pathlib` for paths, never hard-coded `/` or `\\`.
- PyTorch with CUDA from pytorch.org, matching the installed NVIDIA driver. Check `torch.cuda.is_available()` and `torch.cuda.is_bf16_supported()`.
- `torch.compile` is optional: try `triton-windows`. If it does not work cleanly, set `compile: false` for every run (it must be all or nothing).
- On Windows, `DataLoader` workers need `if __name__ == "__main__":` guards; prefer a simple in-process loader over a memory-mapped token file.
- Record PyTorch, CUDA and driver versions and the GPU name in every run log.

## Layout

```
PLAN.md, CLAUDE.md, README.md
configs/baseline.yaml          # every setting, documented
configs/stages/*.yaml          # one file per stage: list of overrides
src/config.py                  # dataclass + YAML loading + overrides
src/data_prep.py               # download, clean, split, write story files + metadata
src/tokenizer.py               # train BPE; byte-level "char" tokenizer; encode splits to .bin
src/model.py                   # GPT with configurable layers, width, FFN size, heads
src/optim.py                   # AdamW, SGD+momentum, Muon param groups
src/train.py                   # time-budgeted training loop, logging
src/evaluate.py                # bits per byte, samples
src/lr_find.py                 # learning-rate range test + plot
scripts/run_stage.py           # runs a stage's configs back to back
tests/                         # pytest: shapes, overfit, bpb, timer, schedule
data/ runs/                    # gitignored
```

## Implementation decisions (fixed; ask before changing)

**Data**
- Source: the TinyStories "all data" archive from `roneneldan/TinyStories` on Hugging Face (inspect the repo file list for the exact name). Inspect the JSON fields first and keep the `source` label (GPT-3.5 vs GPT-4) per story.
- Clean: strip whitespace, drop empty stories, exact duplicates and stories with mojibake (e.g. `â`, `€`). Log counts removed.
- Split by whole story with seed 1234: validation 10k stories, test 10k stories, rest train. Save story texts and source labels per split.
- Token files: one `uint16` `.bin` per split and tokenizer, stories in a fixed shuffled order separated by an end-of-text token. Also save total UTF-8 bytes of story text per split.

**Tokenizers**
- BPE: byte-level BPE trained on the train split only, vocab 4,096 including one `<|endoftext|>` token.
- Character level: raw UTF-8 bytes (256 ids) plus `<|endoftext|>`.

**Model**
- Pre-norm blocks: `x = x + attn(rmsnorm(x))`, `x = x + ffn(rmsnorm(x))`, final RMSNorm, tied input/output embeddings, no biases.
- Attention: 4 heads, causal `F.scaled_dot_product_attention`. Width must be divisible by 4 and head size a multiple of 8.
- FFN: `Linear(d, ffn_size) -> GELU -> Linear(ffn_size, d)`; `ffn_size` is set explicitly in the config (not only as a ratio), because E2 solves it for equal parameters.
- Positions: fixed sinusoidal encoding added to token embeddings. Scale token embeddings by `sqrt(d_model)` as in the original transformer, so positions do not swamp them; no learned position parameters.
- Init: normal(0, 0.02) for linear and embedding weights; scale the output projection of attention and FFN by `1/sqrt(2 * n_layers)` (GPT-2 style).
- Report total and non-embedding parameter counts.

**Training**
- Budget: `budget_seconds` (default 1800) of training time. Count forward, backward, optimiser step and data loading. Pause the timer for validation, checkpointing, compile warm-up and final evaluation. Call `torch.cuda.synchronize()` before every timer reading.
- Schedule from elapsed time: `progress = elapsed / budget_seconds`. Baseline: linear warmup over the first 2%, cosine decay to 10% of peak. Also implement warmup-stable-decay (WSD): warmup 2%, constant, then linear decay to 10% of peak over the last 20% of time.
- Batch: about 32,768 tokens per step, as micro-batches with gradient accumulation to fit 8 GB. For the character-level tokenizer keep the same number of sequences per step, each covering the same bytes.
- Optimisers: AdamW (betas 0.9/0.95, weight decay 0.1 on 2D weights only, clip 1.0); SGD with momentum 0.9 (Nesterov); Muon for 2D hidden weight matrices only, with AdamW for embeddings, norms and anything not 2D. Use `torch.optim.Muon` if the installed PyTorch has it, otherwise Keller Jordan's reference implementation.
- bf16 autocast, dropout 0, seeds set for Python, NumPy and torch.
- Validation during training: every 150 s of training time on a fixed 1 MB slice of validation text, timer paused.
- Staged data (E5): during the decay phase, draw training sequences only from GPT-4 stories (or a configured mix for the replay variant).
- Data order: sequential reads through the shuffled train stream; on reaching the end, reshuffle with a new seed and continue; log epochs as bytes seen / train bytes.

**Evaluation**
- bpb = (sum of next-token losses in nats over every predicted token, including end-of-text) / (ln 2 x UTF-8 bytes of the story text, separators excluded).
- Evaluate in non-overlapping windows of the model's training context over the whole split.
- Samples: 5 fixed prompts, temperature 0.8, top-k 50, fixed seed.

**Logging**
- One folder per run in `runs/<stage>/<run_name>/` with `log.json`: full resolved config, git commit, seed, versions, parameter counts, tokens/bytes seen, epochs, steps, tokens per second, peak VRAM, the validation curve as (elapsed seconds, bpb) pairs, final validation bpb, final test bpb (final runs only), and samples.
- Learning-rate range tests save their (lr, loss) data and a PNG plot.

## Workflow

- Work in small steps; propose a plan before large changes.
- After each step run `pytest -q` and a tiny smoke run, then summarise what changed and what you verified.
- Commit after each working step with a clear message.
- When something fails, show the full error and your diagnosis before changing code.
