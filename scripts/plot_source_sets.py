"""Final validation bpb on the GPT-4-only and matched-mixed sets, one line per run.

Usage:
  .venv\\Scripts\\python.exe scripts\\plot_source_sets.py runs\\E1\\L8 runs\\E5\\wsd_gpt4 --out runs\\E5\\source_sets.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import INK, SERIES, SURFACE, _style  # noqa: E402

SETS = [("gpt35_only", "GPT-3.5 only"), ("mixed", "matched mixed\n(GPT-3.5 + GPT-4)"),
        ("gpt4_only", "GPT-4 only")]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor(SURFACE)
    for i, run in enumerate(args.runs):
        log = json.loads((run / "log.json").read_text(encoding="utf-8"))
        cfg = log["config"]
        ys = [log["val_source_sets"][key]["bpb"] for key, _ in SETS]
        label = f"{cfg['stage']}/{cfg['run_name']} ({cfg['schedule']}, decay data: {cfg.get('decay_data', 'mixed')})"
        ax.plot(range(len(SETS)), ys, color=SERIES[i], lw=2, marker="o", ms=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=label)
        for x, y in enumerate(ys):
            ax.annotate(f"{y:.4f}", (x, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK)
    sizes = log["val_source_sets"]
    ax.set_xticks(range(len(SETS)), [f"{name}\n{sizes[key]['stories']:,} stories, "
                                     f"{sizes[key]['utf8_bytes'] / 1e6:.2f} MB" for key, name in SETS], fontsize=8)
    ax.set_xlim(-0.6, len(SETS) - 0.4)
    ax.set_ylabel("final validation bpb", fontsize=9)
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    fig.suptitle("Validation bpb by story source", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
