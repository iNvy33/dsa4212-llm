"""Story-length statistics in tokens (train split only) and a suggested context length.

Usage: .venv\\Scripts\\python.exe scripts\\length_stats.py [--tokenizers bpe4096 char]

Lengths exclude the EOT separators. Seeing a whole story plus the EOT before it
needs a context of length + 1.
"""
import argparse
from pathlib import Path

import numpy as np

QUANTILES = (50, 99, 99.9, 99.99)


def next_pow2(n: int) -> int:
    return 1 << int(np.ceil(np.log2(n)))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizers", nargs="+", default=["bpe4096", "char"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    print(f"{'tokenizer':10s} {'mean':>8s} " + " ".join(f"{'p' + str(q):>8s}" for q in QUANTILES)
          + f" {'max':>8s}  {'pow2(max+1)':>11s} {'pow2(p99.9+1)':>13s} {'> p99.9 ctx':>11s}")
    for name in args.tokenizers:
        offsets = np.load(args.data_dir / "tokens" / name / "train.offsets.npy")
        lengths = np.diff(offsets) - 1
        qs = np.percentile(lengths, QUANTILES)
        ctx_max = next_pow2(lengths.max() + 1)
        ctx_p999 = next_pow2(int(np.ceil(qs[2])) + 1)
        over = (lengths + 1 > ctx_p999).mean()
        print(f"{name:10s} {lengths.mean():8.1f} " + " ".join(f"{q:8.0f}" for q in qs)
              + f" {lengths.max():8d}  {ctx_max:11d} {ctx_p999:13d} {over:11.4%}")


if __name__ == "__main__":
    main()
