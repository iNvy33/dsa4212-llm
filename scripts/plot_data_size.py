"""Full vs reduced training data, one panel per tokenizer: validation bpb (solid) and
training loss in approx. bits/byte (dashed), so the train-validation gap shows overfitting.

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


if __name__ == "__main__":
    main()
