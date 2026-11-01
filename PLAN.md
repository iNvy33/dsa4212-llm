# Small LM Under a Compute Budget: Project Plan

Oct 30, 2026 · @yuvin

## Goal

Find which design choices give the lowest held-out loss for a GPT-style model trained from scratch on TinyStories in 30 minutes on one RTX 3060 Ti.

The project is an ablation study, not a product. Output quality as an assistant, and complex reasoning, are out of scope. Every result is reported as test bits per byte under the same fixed budget, so each choice is judged by what it buys within the allowance.

## Declarations

These three statements open the report and are fixed before any final run.

| Item | Declaration |
| --- | --- |
| Compute budget | 30 minutes of training wall-clock time per run on one NVIDIA RTX 3060 Ti (8 GB), bf16. Evaluation and compile warm-up are excluded. |
| Initialisation | All weights randomly initialised. No pretrained weights, embeddings or other neural components. Tokenizers are trained by me on the training split. |
| Evaluation objective | Bits per byte on a held-out TinyStories test set, the same text for every configuration. It measures how well the model predicts unseen stories and is comparable across tokenizers. |

Record the exact software versions (PyTorch, CUDA, driver) with the budget, since wall-clock results depend on them.

## Data

Use the original TinyStories release, which mixes stories written by GPT-3.5 and GPT-4, and keep each story's source label so the GPT-4 stories can be separated later.

- **Source file:** `tinystories_all_data.tar.gz` from the [official dataset](https://huggingface.co/datasets/roneneldan/TinyStories). It holds a superset of the stories plus metadata, including a `source` field naming the generating model ([field list](https://huggingface.co/datasets/skeskinen/TinyStories-GPT4/blob/refs%2Fpr%2F1/README.md?code=true)). The plain-text train file (about 2.12M stories) has no labels.
- **Size:** roughly 0.5B GPT-2 tokens for the standard train split (approximate; confirm after tokenizing).
- **Cleaning:** drop exact duplicates, empty stories and stories with broken characters; record how many were removed.
- **Splits, by whole story with a fixed seed:**** **

| Split | Size | Use |
| --- | --- | --- |
| Train | everything else | All training runs |
| Validation | about 10k stories | Learning-rate sweeps, choosing settings, early checks |
| Test | about 10k stories, untouched until final runs | Final bits per byte for every reported result |
|  |  |  |

- **Ordering:** shuffle stories once with a fixed seed, join with an end-of-text token, then read sequentially. Reshuffle with a new seed if a run passes one epoch.
- **Epochs:** measure in bytes of text seen divided by train-set bytes, and log it for every run.

The staged-data experiment (E5) pre-trains on the mixed set, then uses GPT-4 stories only during the learning-rate decay, at the same budget and with the same schedule in both arms.

## Evaluation

The main metric is test bits per byte, computed over every token of the fixed test text, so models with different tokenizers are compared on equal terms.

```latex
\text{bits per byte} = \frac{\sum_i -\ln p(x_i \mid x_{<i})}{\ln 2 \times \text{UTF-8 bytes in the test text}}
```

- **Context at evaluation:** each model is evaluated at its own training context length, in non-overlapping windows. Note this in the context-length results, since longer context lowers loss.
- **Secondary metrics, logged every run:** validation bits per byte over time, tokens and bytes seen, epochs, tokens per second, peak VRAM, total and non-embedding parameters.
- **Noise floor:** the baseline is trained with 3 seeds. Differences smaller than about twice the seed standard deviation are reported as ties.
- **Samples:** 5 fixed prompts per final model, temperature 0.8 with top-k 50 and a fixed seed, shown as illustration only.

**What the metric leaves out:** coherence over a whole story, factual consistency, instruction following and reasoning. A lower bits per byte means better prediction of TinyStories-style text, not a more useful model, and results may not transfer to larger models or other data.

## Stopping criteria

Every final run stops when 1,800 seconds of training time have elapsed, never on convergence or data exhaustion.

- **What counts:** forward and backward passes, optimiser steps and data loading.
- **What doesn't:** periodic validation, checkpoint saving, the final test evaluation and, if `torch.compile` is used, its warm-up steps. The timer pauses for these.
- **Measuring:** wall-clock time with `torch.cuda.synchronize()` before each reading, so queued GPU work is counted.
- **Learning-rate schedule:** driven by elapsed time, `progress = elapsed / 1800`, so every model finishes its decay exactly at the deadline whatever its speed.
- **End of run:** finish the current step, then evaluate the final weights on validation and test. No early stopping or best-checkpoint picking.
- **Same machine state:** native Windows, no other GPU work or heavy apps running; note GPU clocks if results look noisy.
- **Tuning runs:** learning-rate range tests and 10-minute confirmation runs use validation only and sit outside the budget, as the brief allows.

## Baseline configuration

The baseline is a 4-layer model with about 4.7M non-embedding parameters; every experiment changes one setting from it. Sizes are provisional until Phase 0 throughput and E1 show what fits in 30 minutes.

| Setting | Baseline value |
| --- | --- |
| Tokenizer | Own byte-level BPE, 4,096 vocab, trained on the train split |
| Layers × width | 4 × 384 |
| Attention heads | 4, fixed for all runs (head size 96 at the baseline width) |
| FFN | GELU, 2× width (768) |
| Normalisation | RMSNorm, pre-norm, plus a final RMSNorm (fixed for all runs) |
| Positions | Fixed sinusoidal encoding, no learned parameters (fixed for all runs) |
| Embeddings | Input and output tied, no biases |
| Parameters | about 4.7M non-embedding (1.18M per layer) + 1.6M embedding |
| Context length | Longest story in tokens, rounded up to a power of 2 (measured in Phase 0) |
| Batch | About 32,768 tokens per step |
| Optimiser | AdamW, betas 0.9 / 0.95, weight decay 0.1, gradient clip 1.0 |
| Learning rate | From a learning-rate range test per configuration; time-based warmup (first 2%), then cosine to 10% of peak |
| Dropout | 0 |
| Precision and speed | bf16 autocast and `scaled_dot_product_attention`; `torch.compile` only if `triton-windows` installs cleanly, and then for every run |
| Seed | 0 (seeds 1 and 2 for repeats) |

**Context-length check:** a few unusually long stories could push the power of 2 far above typical lengths and slow every run. In Phase 0, compare the longest story with the 99.9th percentile; if it is an outlier, round the percentile up instead. The character-level model needs about 4× more tokens for the same story, so set its context to cover the same number of bytes.

## Experiments

Experiments run as a sequence of stages: each stage fixes its winner, which becomes the baseline for the next. Two small factorial checks (I1, I2) then test whether the main stage choices depend on depth. The core set is 18 final runs (about 9 GPU-hours), plus one more if the depth sweep needs extending. Run counts exclude runs that are reused.

| Stage | Question | Variants (vs. current baseline) | Final runs | Priority |
| --- | --- | --- | --- | --- |
| E0 | How noisy is one run? | Baseline seeds 0, 1, 2 | 3 | Core |
| E1 | How many layers, and so how many parameters, are best in 30 minutes? | 1, 2, 4, 8 layers at width 384, FFN 2× (about 1.2M, 2.4M, 4.7M, 9.4M non-embedding). If 8 wins, add 12 or 16 | 3 (+1) | Core |
| E2 | At the winning size, how should parameters split between width and FFN? | About 1×, 4× and 7× FFN, with width and FFN size set to keep parameters equal (table below) | 3 | Core |
| E3 | Own BPE or character level? | Character level (byte vocabulary), context covering the same bytes | 1 | Core |
| E4 | Which optimiser? | SGD with momentum; Muon (AdamW for embeddings, output layer and norms) | 2 | Core |
| E5 | Does GPT-4-only data at the end help (simulated mid-training)? | Warmup-stable-decay schedule, mixed data throughout vs. GPT-4 stories only during the decay | 2 | Core |
| I1 | Does the tokenizer effect depend on depth? | Character level at 2 and 8 layers (E1 settings); BPE at 2 and 8 layers reused from E1 | 2 | Core |
| I2 | Does the optimiser effect depend on depth? | Muon at 2 and 8 layers (E1 settings); AdamW at 2 and 8 layers reused from E1 | 2 | Core |
| O1 | How reliable is the final winner? | Final configuration, seeds 1 and 2 | 2 | Optional |
| O2 | Does replay help the staged run? | E5 with 50% mixed data during the decay | 1 | Optional |
| O3 | Other vocabulary sizes | BPE 1k, BPE 16k | 2 | Optional |

**E1 measures depth and size together.** Adding layers at a fixed width raises the parameter count, so the winning size is the best size for this width and FFN ratio, not in general. Say so in the limitations. If the best result is at 1 or 8 layers, the optimum may lie outside the range, so extend in that direction before moving on.

**E2 keeps parameters fixed.** Each layer holds 4 × width² attention parameters plus 2 × width × FFN size. Widths are chosen as multiples of 32, so all 4 heads keep a head size that is a multiple of 8, then the FFN size is solved to match the baseline's 1.18M per layer. Values for the 4-layer case are below; recompute them for E1's winner.

| Variant | Width | Head size | FFN size | FFN ratio | Params per layer vs. baseline |
| --- | --- | --- | --- | --- | --- |
| Wide, small FFN | 448 | 112 | 416 | 0.93× | −0.3% |
| Baseline | 384 | 96 | 768 | 2× | 0 |
| Narrower, large FFN | 320 | 80 | 1,216 | 3.8× | +0.7% |
| Narrowest, largest FFN | 256 | 64 | 1,792 | 7× | 0 |

Every run has the same 30 minutes, so tokens seen differ between variants; report them with the loss. Embedding parameters change with width, so compare on non-embedding parameters and report both counts. Choosing one setting per stage (rather than testing all combinations) can miss interactions, such as the best tokenizer depending on size; note this in the limitations.

**Interaction checks (I1, I2).** Each is a 2×2 factorial: two levels of the factor (BPE vs. character level, or AdamW vs. Muon) crossed with 2 vs. 8 layers. The interaction is the difference of differences:

```latex
\text{interaction} = (\text{bpb}_{B,8} - \text{bpb}_{A,8}) - (\text{bpb}_{B,2} - \text{bpb}_{A,2})
```

- With per-run noise σ from E0, this estimate has a standard deviation of about 2σ, so call the interaction real only if it exceeds about 4σ.
- Show each check as an interaction plot: test bits per byte against layers, one line per level. Parallel lines mean no interaction; converging or crossing lines mean the stage choice depends on depth.
- Each variant gets its own learning-rate range test, like every other configuration.
- Run them after E3 and E4, once the character-level and Muon code paths are tested.

## Methodology rules

1. **One change at a time.** Each variant differs from the baseline in one setting only (E1 changes size with depth, and E2 changes width with FFN ratio, by design); experiments are defined as config differences in one file.
2. **Learning rate from a range test, per configuration.** As in Karpathy's makemore lectures (Leslie Smith's range test): one short run of a few hundred steps with the learning rate rising exponentially (e.g. 1e-5 to 1e-1 for AdamW; wider for SGD), plotting smoothed loss against learning rate.
   - Do not use the point where loss starts rising; that is the edge of instability. Take a value about 3–10× below it, near where loss falls fastest.
   - Confirm on the configurations that matter most (E1 depths, E2 ratios, each optimiser) with two 10-minute validation runs at 0.5× and 2× the pick.
   - Repeat the test for every configuration, since the best rate shifts with width and optimiser.
3. **Validation for choices, test for results.** The test set is used only on final runs, and every final run is evaluated on it once.
4. **Report ties honestly.** Differences under twice the E0 seed standard deviation are called ties.
5. **Check every switch.** Each new option passes a shape test and can overfit a single batch before any timed run.
6. **Log everything.** One JSON file per run with the full config, git commit, seed, parameter counts, tokens and bytes seen, epochs, tokens per second, peak VRAM, the validation curve, the range-test plot and final validation and test bits per byte.
7. **State the known gap.** One-at-a-time ablations miss interactions (for example, the best tokenizer may depend on model size); say so in the limitations.

## Schedule

The core experiments need about 13.5 GPU-hours: 9 for final runs and about 4.5 for range tests and confirmation runs. The optional set adds about 2.5. Because each stage needs the previous winner, queue one stage at a time; most stages fit in an evening or overnight session.

&#91;embedded content: one-week schedule · Oct 30 to Nov 6\]

A stage that overruns delays every later one, so check results each morning and evening and start the next stage straight away; cut optional runs first. With a 10-page limit, draft the Declarations, Data and Evaluation sections of the report from this plan while runs are queued.

## Risks and open decisions

| Risk | Fallback |
| --- | --- |
| Code and debugging take longer than planned | Drop the optional set, then the interaction checks, then SGD from E4. E0 + E1 + E2 alone still make a complete report. |
| `torch.compile` unavailable on native Windows | Try `triton-windows`; if it fails, run every run without compile. All runs are treated the same, so comparisons stay fair; models just see fewer tokens. |
| One very long story sets the context length | Use the 99.9th-percentile length instead (Phase 0 check). |
| The range test picks an unstable learning rate | Take a value 3–10× below where loss rises; confirm with two 10-minute runs. |
| SGD trains very poorly | Expected for transformers. Use momentum 0.9 and its own range test so it is a fair comparison, then report the gap as the result. |
| Muon is set up wrongly | Use Keller Jordan's reference implementation; apply Muon only to 2D hidden weight matrices, with AdamW for embeddings, output layer and norms. |
| Small models pass one epoch | Log epochs; up to about 4 is acceptable. Reshuffle between epochs and note it in the results. |
| The GPT-4 share of the data is small | Count it in Phase 0. If the decay phase would repeat GPT-4 stories many times, shorten the decay. |
| GPU heat or background apps make timings noisy | Queue runs overnight with nothing else open; log GPU clocks; the E0 seed runs show the spread. |

**N****otes****:**

Report length is maximum 10 pages

I will just be using Windows

- [ ] Final baseline size after Phase 0 throughput and E1
