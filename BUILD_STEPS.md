# Build steps for Claude Code

Paste one prompt per session step. Start each in **Plan** mode, read the plan, then switch to **Accept edits** (or Manual) to let it build. Commit after each step works.

## Step 0: Environment

> Read CLAUDE.md and PLAN.md. Set up this project on native Windows: initialise git with a sensible .gitignore (data/, runs/, .venv/), create a .venv, and install PyTorch with CUDA matching my NVIDIA driver (check `nvidia-smi` first), plus numpy, pyyaml, tokenizers, huggingface_hub, matplotlib and pytest. Then try installing triton-windows and test whether `torch.compile` works on a tiny model on the GPU. Write a short `scripts/check_env.py` that prints GPU name, driver, CUDA and PyTorch versions, bf16 support and whether compile works. Report the results and stop.

## Step 1: Data

> Implement `src/data_prep.py` as specified in CLAUDE.md. First list the files in the `roneneldan/TinyStories` dataset repo and inspect a few records of the all-data archive to confirm the field names, especially the source label. Then download, clean, split (10k validation, 10k test, seed 1234) and save. Print a summary: stories per split, bytes per split, GPT-3.5 vs GPT-4 counts per split, and how many stories each cleaning rule removed. Add a test that the splits do not overlap.

## Step 2: Tokenizers and story lengths

> Implement `src/tokenizer.py`: train the 4,096-vocab byte-level BPE on the train split, add the byte-level character tokenizer, and encode all splits to uint16 .bin files with end-of-text separators. Then compute story-length statistics in tokens for both tokenizers: mean, median, 99th, 99.9th percentile and max. Recommend a context length per the rule in PLAN.md (longest story rounded up to a power of 2, or the 99.9th percentile if the max is an outlier) and the matching character-level context. Show me the numbers before writing them into the config.

## Step 3: Model

> Implement `src/config.py`, `configs/baseline.yaml` and `src/model.py` exactly as specified in CLAUDE.md. Add tests: output shapes, parameter counts for the baseline and the four E2 variants in PLAN.md (check they match the table within 1%), causal masking (changing a future token does not change earlier logits), and overfitting one small batch to near-zero loss on the GPU.

## Step 4: Training loop and evaluation

> Implement `src/train.py` and `src/evaluate.py`: time-budgeted training with the paused-timer rules, time-based cosine and WSD schedules, AdamW with gradient accumulation, bf16, periodic validation, final validation/test bpb, samples and the JSON log described in CLAUDE.md. Add tests for the bpb formula on a hand-computed example, the schedule values at 0%, 2%, 50% and 100% progress, and that paused time is excluded. Then do a 60-second smoke run of the baseline and show me the log summary.

## Step 5: Learning-rate range test

> Implement `src/lr_find.py`: exponentially increasing learning rate over a few hundred steps, smoothed loss vs learning rate saved as data and a PNG, plus a suggested value 3-10x below where loss starts rising. Run it on the baseline and show me the plot and suggestion.

## Step 6: Phase 0 throughput

> Measure training throughput (tokens per second, peak VRAM) for 1, 2, 4 and 8 layers at the baseline width with 60-second runs. Estimate tokens and epochs each would reach in 30 minutes, and flag any configuration likely to pass one epoch. Summarise in a table.

## Step 7: Stage runner and E0

> Implement `scripts/run_stage.py`: reads a stage file in `configs/stages/`, runs each config back to back as a separate process, skips runs whose log already exists, and writes a stage summary CSV. Create the stage file for E0 (baseline, seeds 0, 1, 2) and E1 (1, 2, 4, 8 layers). Do a dry run with `--budget-seconds 20`, then stop; I will launch the real runs.

## Later steps (when the stages need them)

- **E3 and I1:** character-level runs with matched bytes per sequence and sequences per step.
- **E4 and I2:** SGD with momentum and Muon param groups, each with its own range test.
- **E5:** WSD schedule with GPT-4-only data during the decay, plus the 50% replay option.
- **Analysis:** a script that reads all `log.json` files and produces the results tables, the E1 size curve, interaction plots and validation curves over time.
