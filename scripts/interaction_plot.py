"""2x2 interaction check: test bpb against layers, one line per factor level.

Usage:
  .venv\\Scripts\\python.exe scripts\\interaction_plot.py --title "I1: tokenizer x depth" ^
      --line BPE runs\\E1\\L4 runs\\E1\\L8 --line char runs\\I1\\L4_char runs\\E3\\L8_char ^
      --out runs\\I1\\interaction.png

Prints each line's depth effect (deep - shallow) and its interaction with the first
line (difference of differences). Any number of lines.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import plot_interaction  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--line", nargs=3, action="append", required=True,
                        metavar=("LEVEL", "SHALLOW_RUN", "DEEP_RUN"))
    parser.add_argument("--title", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--metric", default="final_test_bpb")
    args = parser.parse_args()

    lines, effects = {}, {}
    for level, *runs in args.line:
        logs = [json.loads((Path(r) / "log.json").read_text(encoding="utf-8")) for r in runs]
        logs.sort(key=lambda log: log["config"]["n_layers"])
        lines[level] = logs
        (n0, y0), (n1, y1) = ((log["config"]["n_layers"], log[args.metric]) for log in logs)
        effects[level] = y1 - y0
        print(f"{level:>8}: {n0} layers {y0:.4f}  {n1} layers {y1:.4f}  depth effect {y1 - y0:+.4f}")
    a, *others = list(effects)
    for b in others:
        print(f"interaction ({b} effect - {a} effect): {effects[b] - effects[a]:+.4f}")
    plot_interaction(lines, args.out, args.title, args.metric)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
