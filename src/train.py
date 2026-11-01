"""Time-budgeted training with periodic validation and a JSON run log.

Usage:
  .venv\\Scripts\\python.exe -m src.train --config configs/baseline.yaml [--set key=value ...]
                                          [--budget-seconds 60] [--final]

The timer counts forward, backward, optimiser step and data loading. It is paused
for validation, and compile warm-up and final evaluation happen outside it.
The test split is read only with --final.
"""
import argparse
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # pin Triton's default (changes in 3.10)

import numpy as np
import torch

from src.config import Config, load_config, parse_overrides
from src.data_prep import SOURCES
from src.evaluate import evaluate_bpb, evaluate_source_sets, load_eval_text, make_samples
from src.model import GPT
from src.optim import build_optimizer, lr_multiplier
from src.plots import plot_run
from src.tokenizer import get_tokenizer

EVAL_BATCH = 16  # sequences per evaluation forward pass


class Timer:
    """Wall-clock training timer that can be paused. Synchronises CUDA before each reading."""

    def __init__(self, clock=time.perf_counter, sync=None):
        self.clock = clock
        self.sync = sync if sync is not None else (
            torch.cuda.synchronize if torch.cuda.is_available() else lambda: None)
        self.total = 0.0
        self.started_at = None

    def start(self):
        assert self.started_at is None, "timer already running"
        self.sync()
        self.started_at = self.clock()

    def pause(self):
        assert self.started_at is not None, "timer not running"
        self.sync()
        self.total += self.clock() - self.started_at
        self.started_at = None

    def elapsed(self) -> float:
        if self.started_at is None:
            return self.total
        self.sync()
        return self.total + self.clock() - self.started_at


class TrainStream:
    """Sequential reads through the train stories in a seed-dependent order.

    Stories (each followed by its EOT) are concatenated in a permutation drawn
    from (seed, epoch); after the last story a new permutation starts the next
    epoch. Consecutive sequences overlap by one token, so every token is a
    target exactly once. With story_ids, only that subset of stories is read
    (e.g. GPT-4 stories for the staged-data decay phase).
    """

    def __init__(self, tokens: np.ndarray, offsets: np.ndarray, seed: int, seq_len: int,
                 eot_id: int, story_ids: np.ndarray | None = None):
        self.tokens, self.offsets = tokens, offsets
        self.seed, self.seq_len = seed, seq_len
        self.story_ids = story_ids
        if story_ids is None:
            self.n_stories = len(offsets) - 1
            self.total_tokens = len(tokens) - 1  # every token except the first EOT is a target once
        else:
            self.n_stories = len(story_ids)
            self.total_tokens = int((offsets[story_ids + 1] - offsets[story_ids]).sum())
        self.epoch = 0
        self.order = self._permutation(0)
        self.pos = 0
        self.buf = np.array([eot_id], dtype=np.uint16)
        self.tokens_seen = 0

    def _permutation(self, epoch: int) -> np.ndarray:
        perm = np.random.default_rng([self.seed, epoch]).permutation(self.n_stories)
        return perm if self.story_ids is None else self.story_ids[perm]

    def _next_story(self) -> np.ndarray:
        if self.pos == self.n_stories:
            self.epoch += 1
            self.order = self._permutation(self.epoch)
            self.pos = 0
        i = self.order[self.pos]
        self.pos += 1
        return np.asarray(self.tokens[self.offsets[i] + 1:self.offsets[i + 1] + 1])

    def next_batch(self, n: int) -> np.ndarray:
        """(n, seq_len + 1) array of token ids; inputs are [:, :-1], targets [:, 1:]."""
        need = n * self.seq_len + 1
        parts, have = [self.buf], len(self.buf)
        while have < need:
            parts.append(self._next_story())
            have += len(parts[-1])
        stream = np.concatenate(parts)
        batch = np.lib.stride_tricks.sliding_window_view(stream[:need], self.seq_len + 1)[::self.seq_len]
        self.buf = stream[n * self.seq_len:]
        self.tokens_seen += n * self.seq_len
        return batch

    @property
    def epochs(self) -> float:
        return self.tokens_seen / self.total_tokens


def set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def environment() -> dict:
    try:
        driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        driver = "unknown"
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    return {
        "python": platform.python_version(), "platform": platform.platform(),
        "torch": torch.__version__, "cuda": torch.version.cuda, "driver": driver,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "triton": triton_version,
        "TRITON_DEFAULT_FP_FUSION": os.environ.get("TRITON_DEFAULT_FP_FUSION"),
    }


def git_state() -> dict:
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def to_device(batch: np.ndarray, device: str) -> torch.Tensor:
    t = torch.from_numpy(batch.astype(np.int64))
    return t.pin_memory().to(device, non_blocking=True) if device == "cuda" else t


def compile_warmup(model, cfg: Config, vocab_size: int, device: str, steps: int = 3):
    """Compile forward and backward on random tokens, with no optimiser step.

    The weights are untouched, so warm-up is not untimed training.
    """
    before = [p.detach().clone() for p in model.parameters()]
    x = torch.randint(0, vocab_size, (cfg.micro_batch_sequences, cfg.context_length + 1),
                      device=device)
    for _ in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
            _, loss = model(x[:, :-1], x[:, 1:])
        loss.backward()
    model.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    assert all(torch.equal(a, p) for a, p in zip(before, model.parameters()))


def train(cfg: Config, final: bool = False, out_root: Path = Path("runs")) -> dict:
    device = "cuda"
    assert torch.cuda.is_available(), "training needs a GPU"
    run_dir = out_root / cfg.stage / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    set_seeds(cfg.seed)

    tok = get_tokenizer(cfg.tokenizer, Path(cfg.data_dir))
    tok_dir = Path(cfg.data_dir) / "tokens" / cfg.tokenizer
    train_tokens = np.memmap(tok_dir / "train.bin", dtype=np.uint16, mode="r")
    train_offsets = np.load(tok_dir / "train.offsets.npy")
    stream = TrainStream(train_tokens, train_offsets, cfg.seed, cfg.context_length, tok.eot_id)
    # Staged data (E5): during the WSD decay phase, read GPT-4 stories only
    decay_stream = None
    if cfg.decay_data == "gpt4":
        gpt4_ids = np.nonzero(np.load(tok_dir / "train.sources.npy") == SOURCES.index("GPT-4"))[0]
        decay_stream = TrainStream(train_tokens, train_offsets, cfg.seed, cfg.context_length,
                                   tok.eot_id, story_ids=gpt4_ids)
    decay_start = 1.0 - cfg.wsd_decay_frac
    train_bytes = json.loads((tok_dir / "meta.json").read_text(encoding="utf-8"))["splits"]["train"]["utf8_bytes"]
    val_slice, val_slice_bytes = load_eval_text(Path(cfg.data_dir), cfg.tokenizer, "val", cfg.val_slice_bytes)

    model = GPT(cfg, tok.vocab_size).to(device)
    counts = model.param_counts()
    opt = build_optimizer(model, cfg)
    step_model = torch.compile(model) if cfg.compile else model

    log = {"config": cfg.to_dict(), "git": git_state(), "seed": cfg.seed,
           "environment": environment(), "params": counts, "final_run": final}
    print(f"{cfg.stage}/{cfg.run_name}: {counts['non_embedding']:,} non-embedding params, "
          f"{counts['total']:,} total; budget {cfg.budget_seconds:.0f} s")

    t0 = time.perf_counter()
    if cfg.compile:
        compile_warmup(step_model, cfg, tok.vocab_size, device)
    log["compile_warmup_seconds"] = time.perf_counter() - t0
    torch.cuda.reset_peak_memory_stats()

    timer = Timer()
    val_curve, train_curve = [], []
    next_val = cfg.val_interval_seconds
    next_print = 10.0
    interval_loss = torch.zeros((), device=device)  # summed step losses since the last log point
    interval_steps = 0
    step = 0
    timer.start()
    elapsed = 0.0
    while elapsed < cfg.budget_seconds:
        lr = cfg.lr * lr_multiplier(elapsed / cfg.budget_seconds, cfg)
        for group in opt.param_groups:
            group["lr"] = lr
        loss_sum = torch.zeros((), device=device)
        in_decay = decay_stream is not None and elapsed / cfg.budget_seconds >= decay_start
        source = decay_stream if in_decay else stream
        for _ in range(cfg.grad_accum_steps):
            batch = to_device(source.next_batch(cfg.micro_batch_sequences), device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                _, loss = step_model(batch[:, :-1], batch[:, 1:])
            (loss / cfg.grad_accum_steps).backward()
            loss_sum += loss.detach()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        interval_loss += loss_sum / cfg.grad_accum_steps
        interval_steps += 1
        elapsed = timer.elapsed()

        if elapsed >= next_print or elapsed >= cfg.budget_seconds:
            train_loss = interval_loss.item() / interval_steps
            train_curve.append([elapsed, train_loss, lr])
            tokens_now = stream.tokens_seen + (decay_stream.tokens_seen if decay_stream else 0)
            print(f"  step {step:6d}  {elapsed:7.1f}s  loss {train_loss:.4f}  lr {lr:.2e}  "
                  f"grad norm {grad_norm.item():.3f}  {tokens_now / elapsed:,.0f} tok/s"
                  + ("  [GPT-4 only]" if in_decay else ""))
            interval_loss.zero_()
            interval_steps = 0
            next_print += 10.0
        if elapsed >= next_val and elapsed < cfg.budget_seconds:
            timer.pause()
            bpb = evaluate_bpb(model, val_slice, val_slice_bytes, cfg.context_length,
                               EVAL_BATCH, device, cfg.bf16)
            val_curve.append([elapsed, bpb])
            print(f"  val slice bpb {bpb:.4f} at {elapsed:.0f}s")
            next_val += cfg.val_interval_seconds
            timer.start()
    timer.pause()
    train_seconds = timer.elapsed()

    # Final evaluation of the final weights (outside the budget)
    val_tokens, val_bytes = load_eval_text(Path(cfg.data_dir), cfg.tokenizer, "val")
    val_curve.append([train_seconds, evaluate_bpb(model, val_slice, val_slice_bytes,
                                                  cfg.context_length, EVAL_BATCH, device, cfg.bf16)])
    final_val = evaluate_bpb(model, val_tokens, val_bytes, cfg.context_length, EVAL_BATCH, device, cfg.bf16)
    val_source_sets = evaluate_source_sets(model, Path(cfg.data_dir), cfg.tokenizer, tok.eot_id,
                                           cfg.context_length, EVAL_BATCH, device, cfg.bf16)
    final_test = None
    if final:
        test_tokens, test_bytes = load_eval_text(Path(cfg.data_dir), cfg.tokenizer, "test")
        final_test = evaluate_bpb(model, test_tokens, test_bytes, cfg.context_length,
                                  EVAL_BATCH, device, cfg.bf16)
    samples = make_samples(model, tok, cfg, device)
    torch.save(model.state_dict(), run_dir / "model.pt")

    tokens_seen = stream.tokens_seen + (decay_stream.tokens_seen if decay_stream else 0)
    epochs = tokens_seen / stream.total_tokens  # in units of the full train set
    log.update({
        "train_curve_format": "[elapsed seconds, mean train loss (nats/token) over the interval, lr]",
        "steps": step,
        "train_seconds": train_seconds,
        "tokens_seen": tokens_seen,
        # stories are read in random order, so the token fraction estimates the byte fraction
        "bytes_seen_estimate": epochs * train_bytes,
        "epochs": epochs,
        "tokens_per_second": tokens_seen / train_seconds,
        "decay_gpt4_tokens_seen": decay_stream.tokens_seen if decay_stream else None,
        "decay_gpt4_epochs": decay_stream.epochs if decay_stream else None,
        "peak_vram_bytes": torch.cuda.max_memory_allocated(),
        "val_slice_bytes": val_slice_bytes,
        "val_curve": val_curve,
        "train_curve": train_curve,
        "final_val_bpb": final_val,
        "val_source_sets": val_source_sets,
        "final_test_bpb": final_test,
        "samples": samples,
    })
    (run_dir / "log.json").write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    plot_run(log, run_dir / "curves.png")
    print(f"Done: {step} steps, {tokens_seen:,} tokens ({epochs:.3f} epochs) in "
          f"{train_seconds:.1f}s; final val bpb {final_val:.4f} "
          f"(GPT-4-only set {val_source_sets['gpt4_only']['bpb']:.4f}, "
          f"matched mixed set {val_source_sets['mixed']['bpb']:.4f})"
          + (f", test bpb {final_test:.4f}" if final else "") + f"\nLog: {run_dir / 'log.json'}")
    return log


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                        help="config overrides, e.g. n_layers=8 lr=1e-3")
    parser.add_argument("--budget-seconds", type=float, help="shortcut for budget_seconds=...")
    parser.add_argument("--final", action="store_true", help="also evaluate on the test split")
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    overrides = parse_overrides(args.set)
    if args.budget_seconds is not None:
        overrides["budget_seconds"] = args.budget_seconds
    train(load_config(args.config, overrides), final=args.final, out_root=args.out_dir)


if __name__ == "__main__":
    main()
