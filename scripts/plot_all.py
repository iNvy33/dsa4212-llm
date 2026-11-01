"""Figures across all reported runs:
  runs/all_val_overlay.png  validation bpb (1 MB slice) over training time, every run
                            (BPE and byte-level runs in separate panels: very different ranges)
  runs/all_test_bpb.png     final TinyStories test bpb per run (all runs, plus a BPE zoom)

Usage: .venv\\Scripts\\python.exe scripts\\plot_all.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import GRID, INK, INK_2, SERIES, SURFACE, _curves, _style, _zoom, label_ends  # noqa: E402

# (label, run folder, colour slot, line style). The seed repeat shares the 8-layer colour.
RUNS = [
    ("E1 4 layers", "runs/E1/L4", 0, "-"),
    ("E1 8 layers", "runs/E1/L8", 1, "-"),
    ("E1 8 layers, seed 1", "runs/E1/L8_seed1", 1, "--"),
    ("E1 12 layers", "runs/E1/L12", 2, "-"),
    ("E2 8L 448/416", "runs/E2/L8_w448_f416", 3, "-"),
    ("E2 8L 320/1216", "runs/E2/L8_w320_f1216", 4, "-"),
    ("E5 8L WSD + GPT-4 decay", "runs/E5/wsd_gpt4", 5, "-"),
    ("E3 8L byte-level", "runs/E3/L8_char", 6, "-"),
    ("I1 4L byte-level", "runs/I1/L4_char", 7, "-"),
]


def load():
    return [(label, json.loads((Path(d) / "log.json").read_text(encoding="utf-8")), SERIES[c], ls)
            for label, d, c, ls in RUNS]


def plot_val(runs, path: Path):
    groups = [("BPE (4,096 vocab)", [r for r in runs if r[1]["config"]["tokenizer"] != "char"]),
              ("Byte-level", [r for r in runs if r[1]["config"]["tokenizer"] == "char"])]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 2]})
    fig.patch.set_facecolor(SURFACE)
    for ax, (name, group) in zip(axes, groups):
        all_t, all_v, ends = [], [], []
        for label, log, color, ls in group:
            _, _, _, vt, vbpb = _curves(log)
            all_t += vt
            all_v += vbpb
            ax.plot(vt, vbpb, color=color, ls=ls, lw=2, marker="o", ms=5,
                    markeredgecolor=SURFACE, markeredgewidth=1, label=label)
            ends.append((vt[-1], vbpb[-1], f"{label} {vbpb[-1]:.3f}"))
        t_end = max(all_t)
        _zoom(ax, all_t, all_v, t_end)
        _style(ax)
        ax.set_ylabel(f"val bpb, {name}\n(1 MB slice)", fontsize=9)
        ax.set_xlim(0, t_end * 1.2)
        if len(group) <= 4:
            label_ends(ax, ends, t_end)
        else:
            ax.legend(fontsize=8, frameon=False, loc="upper right", ncol=2)
    axes[-1].set_xlabel("training time (minutes)  -  axes zoomed to the last 90% of training")
    fig.suptitle("Validation bpb over training, all runs (30-minute budget)", fontsize=11, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_test(runs, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True, gridspec_kw={"width_ratios": [1, 1.2]})
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(runs)))[::-1]
    bpe = [r[1]["final_test_bpb"] for r in runs if r[1]["config"]["tokenizer"] != "char"]
    for ax, zoom in zip(axes, (False, True)):
        for y, (label, log, color, ls) in zip(ys, runs):
            v = log["final_test_bpb"]
            if zoom and log["config"]["tokenizer"] == "char":
                continue
            ax.scatter(v, y, s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=2,
                       marker="D" if ls == "--" else "o")
            ax.annotate(f"{v:.4f}", (v, y), xytext=(8, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK)
        _style(ax)
        ax.grid(True, axis="x", color=GRID)
        ax.grid(False, axis="y")
        if zoom:
            lo, hi = min(bpe), max(bpe)
            ax.set_xlim(lo - 0.15 * (hi - lo), hi + 0.35 * (hi - lo))
            ax.set_title("zoom: BPE runs only", fontsize=9, color=INK_2, loc="left")
        else:
            vals = [r[1]["final_test_bpb"] for r in runs]
            ax.set_xlim(min(vals) - 0.05, max(vals) + 0.12)
            ax.set_title("all runs", fontsize=9, color=INK_2, loc="left")
        ax.set_xlabel("test bpb (TinyStories test split, lower is better)", fontsize=9)
    axes[0].set_yticks(ys, [r[0] for r in runs], fontsize=8)
    fig.suptitle("Final test bits per byte, all runs", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def main():
    runs = load()
    plot_val(runs, Path("runs/all_val_overlay.png"))
    plot_test(runs, Path("runs/all_test_bpb.png"))
    print("Wrote runs/all_val_overlay.png and runs/all_test_bpb.png")


if __name__ == "__main__":
    main()
