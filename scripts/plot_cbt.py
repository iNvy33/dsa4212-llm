"""Dumbbell chart of CBT test bpb per book for two runs (from their log.json "cbt_test").

Usage:
  .venv\\Scripts\\python.exe scripts\\plot_cbt.py runs\\E1\\L8 runs\\E5\\wsd_gpt4 --out runs\\E5\\cbt_test.png
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import GRID, INK, INK_2, SERIES, SURFACE, _style  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs=2, type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    logs = [json.loads((r / "log.json").read_text(encoding="utf-8")) for r in args.runs]
    labels = [f"{log['config']['stage']}/{log['config']['run_name']}" for log in logs]
    books = [b["title"] for b in logs[0]["cbt_test"]["books"]]
    rows = [[log["cbt_test"]["books"][i]["bpb"] for log in logs] for i in range(len(books))]
    rows.append([log["cbt_test"]["bpb"] for log in logs])
    names = [t.split(": ", 1)[-1] if ": " in t else t for t in books]
    names = [n.replace("Lucy Maud Montgomery Short Stories, 1902 to 1903", "Short Stories, 1902-1903")
             for n in names] + ["All 5 books"]
    authors = [t.split(": ", 1)[0] for t in books] + [f"{logs[0]['cbt_test']['utf8_bytes'] / 1e6:.2f} MB"]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ys = list(range(len(rows)))[::-1]
    for y, (a, b) in zip(ys, rows):
        ax.plot([a, b], [y, y], color=GRID, lw=3, zorder=1, solid_capstyle="round")
        for i, v in enumerate((a, b)):
            ax.scatter(v, y, s=70, color=SERIES[i], edgecolor=SURFACE, linewidth=2, zorder=2,
                       label=labels[i] if y == ys[0] else None)
        ax.annotate(f"{b - a:+.3f}", (max(a, b), y), xytext=(10, 0), textcoords="offset points",
                    va="center", fontsize=8, color=INK_2)
    ax.axhline(0.5, color=INK_2, lw=0.8)  # separates the overall row
    ax.set_yticks(ys, [f"{n}\n{au}" for n, au in zip(names, authors)], fontsize=8)
    ax.set_xlabel("bits per byte (lower is better)")
    lo = min(min(r) for r in rows)
    hi = max(max(r) for r in rows)
    ax.set_xlim(lo - 0.05, hi + 0.08)
    _style(ax)
    ax.grid(True, axis="x", color=GRID)
    ax.grid(False, axis="y")
    ax.legend(fontsize=8, frameon=False, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    fig.suptitle("Children's Book Test (test books): bpb per book", fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, facecolor=SURFACE)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
