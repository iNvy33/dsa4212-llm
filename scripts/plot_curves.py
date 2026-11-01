"""Plot loss over time: per-run curves.png for each run, plus one overlay of all of them.

Usage:
  .venv\\Scripts\\python.exe scripts\\plot_curves.py runs\\E1               (every run in a stage)
  .venv\\Scripts\\python.exe scripts\\plot_curves.py runs\\E1\\L2 runs\\E1\\L8  (chosen runs)
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.plots import plot_overlay, plot_run  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", nargs="+", type=Path, help="run folders or stage folders")
    parser.add_argument("--out", type=Path, help="overlay PNG (default: <first path>/curves_overlay.png)")
    args = parser.parse_args()

    run_dirs = []
    for p in args.paths:
        run_dirs += [p] if (p / "log.json").exists() else sorted(d.parent for d in p.glob("*/log.json"))
    if not run_dirs:
        sys.exit("no log.json found")
    logs = [json.loads((d / "log.json").read_text(encoding="utf-8")) for d in run_dirs]
    for d, log in zip(run_dirs, logs):
        plot_run(log, d / "curves.png")
    out = args.out or (args.paths[0] / "curves_overlay.png")
    labels = [log["config"]["run_name"] for log in logs]
    plot_overlay(logs, labels, out, " vs ".join(labels) if len(labels) <= 4 else f"{len(labels)} runs")
    print(f"Wrote curves.png in {len(run_dirs)} run folders and {out}")


if __name__ == "__main__":
    main()
