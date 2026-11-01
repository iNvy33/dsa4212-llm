"""Tokenizers (own byte-level BPE, byte-level "char") and encoding splits to .bin.

Usage:
  .venv\\Scripts\\python.exe -m src.tokenizer --tokenizer bpe --vocab-size 4096
  .venv\\Scripts\\python.exe -m src.tokenizer --tokenizer char

Outputs in data/tokens/<name>/ (name is e.g. "bpe4096" or "char"):
  <split>.bin           uint16 tokens: EOT s1 EOT s2 ... EOT sN EOT
  <split>.offsets.npy   int64, N+1 positions of the EOT before each story
                        (last entry: the final EOT). Story i with its closing
                        EOT is tokens[offsets[i] + 1 : offsets[i + 1] + 1].
  <split>.sources.npy   uint8 per story, index into SOURCES (0 GPT-3.5, 1 GPT-4)
  meta.json             vocab size, EOT id, tokens / stories / UTF-8 bytes per split
The BPE tokenizer itself is saved as data/tokenizers/<name>.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer as HFTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers

from src.data_prep import SOURCES, SPLITS

EOT = "<|endoftext|>"


class CharTokenizer:
    """Raw UTF-8 bytes (ids 0-255) plus one end-of-text token (id 256)."""

    name = "char"
    vocab_size = 257
    eot_id = 256

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8"))

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def decode(self, ids: list[int]) -> str:
        return bytes(i for i in ids if i < 256).decode("utf-8", errors="replace")


class BPETokenizer:
    """Wrapper around a trained Hugging Face byte-level BPE tokenizer."""

    def __init__(self, hf: HFTokenizer, name: str):
        self.hf = hf
        self.name = name
        self.vocab_size = hf.get_vocab_size()
        self.eot_id = hf.token_to_id(EOT)

    def encode(self, text: str) -> list[int]:
        return self.hf.encode(text).ids

    def encode_batch(self, texts: list[str]) -> list[list[int]]:
        return [e.ids for e in self.hf.encode_batch(texts)]

    def decode(self, ids: list[int]) -> str:
        return self.hf.decode(ids, skip_special_tokens=True)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.hf.save(str(path))

    @classmethod
    def load(cls, path: Path) -> "BPETokenizer":
        return cls(HFTokenizer.from_file(str(path)), path.stem)


def train_bpe(texts, vocab_size: int) -> BPETokenizer:
    """Train byte-level BPE. vocab_size counts the 256 bytes, merges and EOT."""
    hf = HFTokenizer(models.BPE())
    hf.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    hf.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=[EOT],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )
    hf.train_from_iterator(texts, trainer=trainer)
    return BPETokenizer(hf, f"bpe{vocab_size}")


def get_tokenizer(name: str, data_dir: Path = Path("data")):
    """Load a tokenizer by name: "char" or "bpe<vocab>"."""
    if name == "char":
        return CharTokenizer()
    return BPETokenizer.load(data_dir / "tokenizers" / f"{name}.json")


def read_stories(path: Path) -> tuple[list[str], np.ndarray]:
    texts, sources = [], []
    with path.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            texts.append(r["story"])
            sources.append(SOURCES.index(r["source"]))
    return texts, np.array(sources, dtype=np.uint8)


def encode_stories(tok, texts: list[str], batch: int = 10_000) -> tuple[np.ndarray, np.ndarray]:
    """Return (uint16 tokens laid out as EOT s1 EOT ... sN EOT, int64 offsets)."""
    assert tok.vocab_size <= 2**16
    chunks = [np.array([tok.eot_id], dtype=np.uint16)]
    lengths = []
    for i in range(0, len(texts), batch):
        for ids in tok.encode_batch(texts[i:i + batch]):
            ids.append(tok.eot_id)
            chunks.append(np.array(ids, dtype=np.uint16))
            lengths.append(len(ids))
    tokens = np.concatenate(chunks)
    offsets = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    assert (tokens == tok.eot_id).sum() == len(texts) + 1, "EOT token inside a story"
    return tokens, offsets


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tokenizer", choices=["bpe", "char"], required=True)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    args = parser.parse_args()

    story_dir = args.data_dir / "stories"
    split_meta = json.loads((story_dir / "meta.json").read_text(encoding="utf-8"))["splits"]
    stories = {s: read_stories(story_dir / f"{s}.jsonl") for s in SPLITS}

    if args.tokenizer == "char":
        tok = CharTokenizer()
    else:
        name = f"bpe{args.vocab_size}"
        path = args.data_dir / "tokenizers" / f"{name}.json"
        if path.exists():
            print(f"Loading existing {path}")
            tok = BPETokenizer.load(path)
        else:
            print(f"Training {name} on the train split ...")
            tok = train_bpe(stories["train"][0], args.vocab_size)  # train split only
            tok.save(path)
        assert tok.vocab_size == args.vocab_size, tok.vocab_size

    out_dir = args.data_dir / "tokens" / tok.name
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = {"tokenizer": tok.name, "vocab_size": tok.vocab_size, "eot_id": tok.eot_id,
            "layout": "EOT s1 EOT s2 ... EOT sN EOT", "splits": {}}
    for split, (texts, sources) in stories.items():
        print(f"Encoding {split} ({len(texts):,} stories) ...")
        tokens, offsets = encode_stories(tok, texts)
        tokens.tofile(out_dir / f"{split}.bin")
        np.save(out_dir / f"{split}.offsets.npy", offsets)
        np.save(out_dir / f"{split}.sources.npy", sources)
        utf8_bytes = split_meta[split]["utf8_bytes"]
        meta["splits"][split] = {
            "stories": len(texts), "tokens": int(len(tokens)), "utf8_bytes": utf8_bytes,
            "bytes_per_token": utf8_bytes / (len(tokens) - len(texts) - 1),
        }
        print(f"  {len(tokens):,} tokens, {meta['splits'][split]['bytes_per_token']:.3f} "
              "bytes per story token")
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir}")


if __name__ == "__main__":
    main()
