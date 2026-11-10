"""Make a reduced-data copy of the story splits: a fraction of the train stories, same val/test.

Usage: .venv\\Scripts\\python.exe -m src.data_subset --fraction 0.05 --out data_5pct

The train subset is a seeded random sample taken separately within each source
(GPT-3.5, GPT-4), so the source mix matches the full train split. Selected stories
keep their order in the full train file. val.jsonl and test.jsonl are copied
byte for byte, so evaluation text is identical to the full-data runs.
Then tokenize as usual with --data-dir <out> (BPE is retrained on the subset).
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from src.data_prep import SOURCES, split_summary, write_jsonl


def subset_indices(sources: list[str], fraction: float, seed: int) -> np.ndarray:
    """Sorted indices of a per-source random sample of round(fraction * count) stories."""
    sources = np.asarray(sources)
    rng = np.random.default_rng(seed)
    picked = []
    for source in SOURCES:
        idx = np.flatnonzero(sources == source)
        picked.append(rng.permutation(idx)[:round(fraction * len(idx))])
    return np.sort(np.concatenate(picked))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--fraction", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    assert 0 < args.fraction < 1, args.fraction

    src_dir, out_dir = args.data_dir / "stories", args.out / "stories"
    if out_dir.exists():
        raise SystemExit(f"{out_dir} already exists; delete it to rebuild")
    out_dir.mkdir(parents=True)

    with (src_dir / "train.jsonl").open(encoding="utf-8") as f:
        train = [json.loads(line) for line in f]
    keep = subset_indices([s["source"] for s in train], args.fraction, args.seed)
    subset = [train[i] for i in keep]
    write_jsonl(out_dir / "train.jsonl", subset)
    for split in ("val", "test"):
        shutil.copyfile(src_dir / f"{split}.jsonl", out_dir / f"{split}.jsonl")

    meta = json.loads((src_dir / "meta.json").read_text(encoding="utf-8"))
    meta["subset"] = {"of": str(args.data_dir), "fraction": args.fraction, "seed": args.seed,
                      "stratified_by": "source", "full_train": meta["splits"]["train"]}
    meta["splits"]["train"] = split_summary(subset)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    t = meta["splits"]["train"]
    print(f"Train subset: {t['stories']:,} stories, {t['utf8_bytes'] / 1e6:.1f} MB, sources {t['sources']}")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
