"""Learning-rate range test (Leslie Smith): exponentially rising LR, smoothed loss vs LR.

Usage:
  .venv\\Scripts\\python.exe -m src.lr_find --config configs/baseline.yaml [--set key=value ...]
                                            [--lr-min 1e-5] [--lr-max 1e-1] [--steps 300]

Saves runs/<stage>/lr_find_<run_name>/lr_find.json (lr, loss, smoothed loss,
suggestion) and lr_find.png. Uses training batches only (no validation or test).

Suggestion: "rise" is where the smoothed loss first climbs clearly (2%) above
its lowest value so far. Take a value 3-10x below it, near where the loss falls fastest: the
steepest-descent LR, clamped into [rise / 10, rise / 3].
"""
import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from src.config import Config, load_config, parse_overrides
from src.model import GPT
from src.optim import build_optimizer
from src.tokenizer import get_tokenizer
from src.train import TrainStream, set_seeds, to_device

DIVERGE_FACTOR = 4.0  # stop once smoothed loss exceeds this multiple of its minimum


def exp_lr(step: int, steps: int, lr_min: float, lr_max: float) -> float:
    """LR at step 0..steps-1, rising geometrically from lr_min to lr_max."""
    return lr_min * (lr_max / lr_min) ** (step / (steps - 1))


def smooth(losses: list[float], beta: float = 0.95) -> np.ndarray:
    """Bias-corrected exponential moving average."""
    out, avg = [], 0.0
    for i, loss in enumerate(losses, 1):
        avg = beta * avg + (1 - beta) * loss
        out.append(avg / (1 - beta**i))
    return np.array(out)


def suggest(lrs: np.ndarray, smoothed: np.ndarray, skip: int = 10, tol: float = 0.02) -> dict:
    """Rise point, steepest-descent LR and the suggested LR (see module docstring).

    The rise point is the lowest smoothed loss before the first time the smoothed
    loss climbs more than tol above its running minimum. (With gradient clipping
    the loss can spike, recover and keep falling at higher LRs, so the global
    minimum is not the edge of instability.)
    """
    run_min = np.minimum.accumulate(smoothed)
    rising = np.nonzero((np.arange(len(smoothed)) >= skip) & (smoothed > run_min * (1 + tol)))[0]
    end = int(rising[0]) if rising.size else len(smoothed)
    i_min = int(np.argmin(smoothed[skip:end])) + skip
    rise = float(lrs[i_min])
    slope = np.gradient(smoothed, np.log(lrs))
    i_steep = int(np.argmin(slope[skip:i_min + 1])) + skip if i_min > skip else i_min
    steepest = float(lrs[i_steep])
    return {"rise_lr": rise, "steepest_lr": steepest,
            "suggested_lr": float(min(max(steepest, rise / 10), rise / 3))}


def lr_find(cfg: Config, lr_min: float, lr_max: float, steps: int, out_root: Path) -> dict:
    device = "cuda"
    set_seeds(cfg.seed)
    tok = get_tokenizer(cfg.tokenizer, Path(cfg.data_dir))
    tok_dir = Path(cfg.data_dir) / "tokens" / cfg.tokenizer
    stream = TrainStream(np.memmap(tok_dir / "train.bin", dtype=np.uint16, mode="r"),
                         np.load(tok_dir / "train.offsets.npy"), cfg.seed, cfg.context_length,
                         tok.eot_id)
    model = GPT(cfg, tok.vocab_size).to(device)
    opt = build_optimizer(model, cfg)

    lrs, losses = [], []
    for step in range(steps):
        lr = exp_lr(step, steps, lr_min, lr_max)
        for group in opt.param_groups:
            group["lr"] = lr
        loss_sum = 0.0
        for _ in range(cfg.grad_accum_steps):
            batch = to_device(stream.next_batch(cfg.micro_batch_sequences), device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=cfg.bf16):
                _, loss = model(batch[:, :-1], batch[:, 1:])
            (loss / cfg.grad_accum_steps).backward()
            loss_sum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        lrs.append(lr)
        losses.append(loss_sum / cfg.grad_accum_steps)
        s = smooth(losses)
        if not math.isfinite(losses[-1]) or (step > 10 and s[-1] > DIVERGE_FACTOR * s.min()):
            print(f"  stopped at step {step}: loss diverged at lr {lr:.2e}")
            break
        if step % 25 == 0:
            print(f"  step {step:4d}  lr {lr:.2e}  loss {losses[-1]:.4f}  smoothed {s[-1]:.4f}")

    lrs_a, smoothed = np.array(lrs), smooth(losses)
    result = {"config": cfg.to_dict(), "lr_min": lr_min, "lr_max": lr_max, "steps": steps,
              "lr": lrs, "loss": losses, "smoothed_loss": smoothed.tolist(),
              **suggest(lrs_a, smoothed)}

    out_dir = out_root / cfg.stage / f"lr_find_{cfg.run_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "lr_find.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    plot(result, out_dir / "lr_find.png", f"{cfg.stage}/{cfg.run_name} ({cfg.optimizer})")
    print(f"rise {result['rise_lr']:.2e}, steepest {result['steepest_lr']:.2e}, "
          f"suggested {result['suggested_lr']:.2e}\nSaved {out_dir}")
    return result


def plot(result: dict, path: Path, title: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(result["lr"], result["loss"], color="0.75", lw=1, label="loss")
    ax.plot(result["lr"], result["smoothed_loss"], color="C0", lw=2, label="smoothed loss")
    for key, color in (("steepest_lr", "C2"), ("suggested_lr", "C3"), ("rise_lr", "C1")):
        ax.axvline(result[key], color=color, ls="--", lw=1.2,
                   label=f"{key.replace('_lr', '')}: {result[key]:.1e}")
    ax.axvspan(result["rise_lr"] / 10, result["rise_lr"] / 3, color="C3", alpha=0.08,
               label="3-10x below rise")
    ax.set_xscale("log")
    s = np.array(result["smoothed_loss"])
    ax.set_ylim(s.min() - 0.3, min(s[0], DIVERGE_FACTOR * s.min()) + 0.3)
    ax.set_xlabel("learning rate")
    ax.set_ylabel("training loss (nats/token)")
    ax.set_title(f"LR range test: {title}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=Path("configs/baseline.yaml"))
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--lr-max", type=float, default=1e-1)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out-dir", type=Path, default=Path("runs"))
    args = parser.parse_args()
    cfg = load_config(args.config, parse_overrides(args.set))
    lr_find(cfg, args.lr_min, args.lr_max, args.steps, args.out_dir)


if __name__ == "__main__":
    main()
