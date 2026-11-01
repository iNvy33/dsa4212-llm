"""Download, clean and split TinyStories (all-data archive, with source labels).

Usage: .venv\\Scripts\\python.exe -m src.data_prep

Outputs in data/stories/:
  train.jsonl, val.jsonl, test.jsonl  one {"story", "source"} object per line,
                                      in a fixed shuffled order (seed 1234)
  meta.json                           counts, bytes and source mix per split,
                                      stories removed by each cleaning rule
"""
import argparse
import json
import tarfile
from collections import Counter
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

REPO_ID = "roneneldan/TinyStories"
ARCHIVE = "TinyStories_all_data.tar.gz"
REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"  # pinned for reproducibility
SOURCES = ("GPT-3.5", "GPT-4")
# Characters that appear when UTF-8 text was decoded as Latin-1/cp1252, plus
# the Unicode replacement character. None occur in normal TinyStories text.
MOJIBAKE = ("â", "Ã", "Â", "€", "�")
SPLITS = ("train", "val", "test")


def load_records(archive: Path) -> list[dict]:
    """Read every JSON file in the archive, in sorted file-name order."""
    files = {}
    with tarfile.open(archive) as tf:
        for member in tf:
            if member.isfile() and member.name.endswith(".json"):
                files[Path(member.name).name] = json.load(tf.extractfile(member))
    records = []
    for name in sorted(files):
        records.extend(files[name])
    return records


def is_mojibake(text: str) -> bool:
    return any(c in text for c in MOJIBAKE)


def clean(records: list[dict]) -> tuple[list[dict], dict]:
    """Strip whitespace, then drop empty, mojibake and exact-duplicate stories.

    Rules are applied in that order; each story is counted under the first rule
    that removes it. The first occurrence of a duplicate is kept.
    """
    removed = Counter({"empty": 0, "mojibake": 0, "duplicate": 0})
    seen = set()
    kept = []
    for r in records:
        story = r["story"].strip()
        if not story:
            removed["empty"] += 1
        elif is_mojibake(story):
            removed["mojibake"] += 1
        elif story in seen:
            removed["duplicate"] += 1
        else:
            seen.add(story)
            if r["source"] not in SOURCES:
                raise ValueError(f"unexpected source label {r['source']!r}")
            kept.append({"story": story, "source": r["source"]})
    return kept, dict(removed)


def split_indices(n: int, n_val: int, n_test: int, seed: int) -> dict[str, np.ndarray]:
    """Shuffle story indices once; first n_val are validation, next n_test test."""
    perm = np.random.default_rng(seed).permutation(n)
    return {
        "val": perm[:n_val],
        "test": perm[n_val:n_val + n_test],
        "train": perm[n_val + n_test:],
    }


def write_jsonl(path: Path, stories: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for s in stories:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")


def split_summary(stories: list[dict]) -> dict:
    sources = Counter(s["source"] for s in stories)
    return {
        "stories": len(stories),
        "utf8_bytes": sum(len(s["story"].encode("utf-8")) for s in stories),
        "sources": {k: sources.get(k, 0) for k in SOURCES},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--n-val", type=int, default=10_000)
    parser.add_argument("--n-test", type=int, default=10_000)
    args = parser.parse_args()

    raw_dir = args.data_dir / "raw"
    out_dir = args.data_dir / "stories"
    out_dir.mkdir(parents=True, exist_ok=True)

    archive = Path(hf_hub_download(REPO_ID, ARCHIVE, repo_type="dataset",
                                   revision=REVISION, local_dir=raw_dir))
    print(f"Reading {archive} ...")
    records = load_records(archive)
    raw_sources = Counter(r["source"] for r in records)
    print(f"  {len(records):,} raw stories")

    stories, removed = clean(records)
    del records
    print(f"  {len(stories):,} after cleaning; removed {removed}")

    idx = split_indices(len(stories), args.n_val, args.n_test, args.seed)
    sets = [set(v.tolist()) for v in idx.values()]
    assert sum(len(s) for s in sets) == len(set().union(*sets)) == len(stories)

    meta = {
        "repo_id": REPO_ID, "archive": ARCHIVE, "revision": REVISION,
        "split_seed": args.seed,
        "raw_stories": sum(raw_sources.values()),
        "raw_sources": dict(raw_sources),
        "removed": removed,
        "splits": {},
    }
    for name in SPLITS:
        split = [stories[i] for i in idx[name]]
        write_jsonl(out_dir / f"{name}.jsonl", split)
        meta["splits"][name] = split_summary(split)
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"\n{'split':6s} {'stories':>10s} {'MB':>9s} {'GPT-3.5':>10s} {'GPT-4':>10s}")
    for name in SPLITS:
        m = meta["splits"][name]
        print(f"{name:6s} {m['stories']:10,d} {m['utf8_bytes'] / 1e6:9.1f} "
              f"{m['sources']['GPT-3.5']:10,d} {m['sources']['GPT-4']:10,d}")
    print(f"\nWrote {out_dir}")


if __name__ == "__main__":
    main()
