"""Full vs reduced training data:
  runs/D5/data_size.png       one panel per tokenizer: validation bpb (solid) and training
                              loss in approx. bits/byte (dashed), so the gap shows overfitting
  runs/D5/data_size_test.png  test bpb against training-data size, one line per tokenizer

Usage: .venv\\Scripts\\python.exe scripts\\plot_data_size.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import INK, SERIES, SURFACE, _curves, _style, _zoom  # noqa: E402

# (panel title, [(label, run folder), ...]); full data first, reduced second.
PANELS = [
    ("BPE, 4 layers", [("full data", "runs/E1/L4"), ("5% data", "runs/D5/L4_bpe")]),
    ("char (32 x 1,024), 4 layers", [("full data", "runs/I1/L4_char_ctx1024"),
                                     ("5% data", "runs/D5/L4_char_ctx1024")]),
]
OUT = Path("runs/D5/data_size.png")
OUT_TEST = Path("runs/D5/data_size_test.png")
SIZES = ["5% data\n(191 MB)", "full data\n(3.8 GB)"]
LINES = [("BPE", PANELS[0][1]), ("char (32 x 1,024)", PANELS[1][1])]


def plot_test():
    """Test bpb at 5% and full data; each point labelled with its value and epochs."""
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    fig.patch.set_facecolor(SURFACE)
    for i, (level, runs) in enumerate(LINES):
        logs = [json.loads((Path(d) / "log.json").read_text(encoding="utf-8")) for _, d in runs[::-1]]
        ys = [log["final_test_bpb"] for log in logs]
        ax.plot([0, 1], ys, color=SERIES[i], lw=2, marker="o", ms=8, markeredgecolor=SURFACE,
                markeredgewidth=2, label=level)
        for x, y, log in zip([0, 1], ys, logs):
            ax.annotate(f"{y:.4f}\n{log['epochs']:.1f} epochs", (x, y), xytext=(0, 9),
                        textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)
        ax.annotate(f"{ys[1] - ys[0]:+.3f}", (0.5, (ys[0] + ys[1]) / 2), xytext=(0, -12),
                    textcoords="offset points", ha="center", fontsize=8, color=SERIES[i])
    ax.set_xticks([0, 1], SIZES)
    ax.set_xlim(-0.4, 1.4)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.12 * (hi - lo))
    ax.set_ylabel("test bpb (TinyStories test split, lower is better)", fontsize=9)
    _style(ax)
    ax.grid(False, axis="x")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    fig.suptitle("Test bpb vs training-data size, 4 layers, 30 minutes", fontsize=11, color=INK,
                 x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT_TEST, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {OUT_TEST}")


def main():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    fig.patch.set_facecolor(SURFACE)
    for ax, (title, runs) in zip(axes, PANELS):
        all_t, all_y = [], []
        for i, (label, d) in enumerate(runs):
            log = json.loads((Path(d) / "log.json").read_text(encoding="utf-8"))
            t, loss, _, vt, vbpb = _curves(log, train_in_bpb=True)
            name = f"{label}, {log['epochs']:.1f} epochs"
            ax.plot(vt, vbpb, color=SERIES[i], lw=2, marker="o", ms=4, markeredgecolor=SURFACE,
                    label=f"{name}: val ({vbpb[-1]:.3f})")
            ax.plot(t, loss, color=SERIES[i], lw=1.2, ls="--", label=f"{name}: train")
            all_t += t + vt
            all_y += loss + vbpb
        _zoom(ax, all_t, all_y, max(all_t), frac=0.2)
        _style(ax)
        ax.set_xlim(left=0)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_xlabel("training time (minutes)  -  axis zoomed to the last 80%", fontsize=9)
        ax.legend(fontsize=7.5, frameon=False, loc="upper right")
    axes[0].set_ylabel("bits per byte (val: 1 MB slice; train: approx., 10 s mean)", fontsize=9)
    fig.suptitle("Full vs 5% training data: validation (solid) and training loss (dashed)",
                 fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(f"Wrote {OUT}")
    plot_test()


if __name__ == "__main__":
    main()
