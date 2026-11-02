"""Figures across all reported BPE runs (the byte-level runs are excluded):
  runs/all_val_overlay.png  validation bpb (1 MB slice) over training time
  runs/all_test_bpb.png     final TinyStories test bpb per run
  runs/all_test_ce.png      the same results as cross-entropy (nats per BPE token)

Test cross-entropy is converted exactly from the logged test bpb: the test evaluation sums
the loss over every token of test.bin except the first, so
  CE = bpb * ln 2 * test bytes / (test tokens - 1),
with both counts taken from the tokenizer's meta.json. No model is re-evaluated.

Usage: .venv\\Scripts\\python.exe scripts\\plot_all.py
"""
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import GRID, INK, SERIES, SURFACE, _curves, _style, _zoom  # noqa: E402

# (label, run folder, colour slot, line style). The seed repeat shares the 8-layer colour.
RUNS = [
    ("E1 4 layers", "runs/E1/L4", 0, "-"),
    ("E1 8 layers", "runs/E1/L8", 1, "-"),
    ("E1 8 layers, seed 1", "runs/E1/L8_seed1", 1, "--"),
    ("E1 12 layers", "runs/E1/L12", 2, "-"),
    ("E2 8L 448/416", "runs/E2/L8_w448_f416", 3, "-"),
    ("E2 8L 320/1216", "runs/E2/L8_w320_f1216", 4, "-"),
    ("E5 8L WSD + GPT-4 decay", "runs/E5/wsd_gpt4", 5, "-"),
]


def load():
    return [(label, json.loads((Path(d) / "log.json").read_text(encoding="utf-8")), SERIES[c], ls)
            for label, d, c, ls in RUNS]


def plot_val(runs, path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor(SURFACE)
    all_t, all_v = [], []
    for label, log, color, ls in runs:
        _, _, _, vt, vbpb = _curves(log)
        all_t += vt
        all_v += vbpb
        ax.plot(vt, vbpb, color=color, ls=ls, lw=2, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1, label=f"{label} ({vbpb[-1]:.3f})")
    _zoom(ax, all_t, all_v, max(all_t))
    _style(ax)
    ax.set_xlim(left=0)
    ax.set_ylabel("val bpb (1 MB slice)", fontsize=9)
    ax.set_xlabel("training time (minutes)  -  axis zoomed to the last 90% of training")
    ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    fig.suptitle("Validation bpb over training, BPE runs (30-minute budget)", fontsize=11, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def test_ce(log: dict) -> float:
    """Test cross-entropy in nats per token, recovered from the logged test bpb."""
    tok_dir = Path(log["config"]["data_dir"]) / "tokens" / log["config"]["tokenizer"]
    test = json.loads((tok_dir / "meta.json").read_text(encoding="utf-8"))["splits"]["test"]
    return log["final_test_bpb"] * math.log(2) * test["utf8_bytes"] / (test["tokens"] - 1)


def plot_test(runs, path: Path, value=lambda log: log["final_test_bpb"],
              xlabel="test bpb (TinyStories test split, lower is better)",
              title="Final test bits per byte, BPE runs"):
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(runs)))[::-1]
    vals = [value(log) for _, log, _, _ in runs]
    for y, v, (label, log, color, ls) in zip(ys, vals, runs):
        ax.scatter(v, y, s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=2,
                   marker="D" if ls == "--" else "o")
        ax.annotate(f"{v:.4f}", (v, y), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK)
    lo, hi = min(vals), max(vals)
    ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.3 * (hi - lo))
    ax.set_yticks(ys, [r[0] for r in runs], fontsize=8)
    _style(ax)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    ax.set_xlabel(xlabel, fontsize=9)
    fig.suptitle(title, fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    runs = load()
    plot_val(runs, Path("runs/all_val_overlay.png"))
    plot_test(runs, Path("runs/all_test_bpb.png"))
    plot_test(runs, Path("runs/all_test_ce.png"), value=test_ce,
              xlabel="test cross-entropy (nats per BPE token, TinyStories test split, lower is better)",
              title="Final test cross-entropy, BPE runs")
    for label, log, _, _ in runs:
        print(f"{label:28s} test bpb {log['final_test_bpb']:.4f}  test CE {test_ce(log):.4f} nats/token")
    print("Wrote runs/all_val_overlay.png, runs/all_test_bpb.png and runs/all_test_ce.png")


if __name__ == "__main__":
    main()
