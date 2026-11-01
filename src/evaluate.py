"""Bits per byte and text samples.

bpb = (sum of next-token losses in nats over every predicted token, EOT included)
      / (ln 2 x UTF-8 bytes of the story text, separators excluded)

Text is evaluated in windows of context_length + 1 tokens that advance by
context_length, so every token after the first EOT is predicted exactly once.
"""
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.data_prep import SOURCES

PROMPTS = [
    "Once upon a time",
    "One day, a little girl named Lily",
    "Tom and his dog went to the park.",
    "The sun was shining and the birds were singing.",
    "In a small house by the sea, there lived",
]


def load_eval_text(data_dir: Path, tokenizer: str, split: str,
                   max_bytes: int | None = None) -> tuple[np.ndarray, int]:
    """Tokens and UTF-8 story bytes of a split, or of its first stories up to max_bytes.

    With max_bytes, whole stories are taken in file order until their bytes
    reach max_bytes (the fixed validation slice used during training).
    """
    tok_dir = Path(data_dir) / "tokens" / tokenizer
    tokens = np.memmap(tok_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    offsets = np.load(tok_dir / f"{split}.offsets.npy")
    if max_bytes is None:
        meta = json.loads((tok_dir / "meta.json").read_text(encoding="utf-8"))
        return np.asarray(tokens), meta["splits"][split]["utf8_bytes"]
    n_bytes, n_stories = 0, 0
    with (Path(data_dir) / "stories" / f"{split}.jsonl").open(encoding="utf-8") as f:
        for line in f:
            n_bytes += len(json.loads(line)["story"].encode("utf-8"))
            n_stories += 1
            if n_bytes >= max_bytes:
                break
    return np.asarray(tokens[:offsets[n_stories] + 1]), n_bytes


def select_source_sets(sources: np.ndarray, story_bytes: np.ndarray, seed: int = 1234) -> dict:
    """Story ids for two equal-size evaluation sets (E5).

    gpt4_only: every GPT-4 story.
    mixed:     a seeded random sample of all stories with the same UTF-8 bytes as
               gpt4_only, stratified so each source keeps its natural byte share.
               Its GPT-4 stories may also be in gpt4_only.
    """
    gpt4 = SOURCES.index("GPT-4")
    a = np.nonzero(sources == gpt4)[0]
    target = story_bytes[a].sum()
    rng = np.random.default_rng(seed)
    parts = []
    for s in range(len(SOURCES)):
        ids = np.nonzero(sources == s)[0]
        share = story_bytes[ids].sum() / story_bytes.sum()
        ids = rng.permutation(ids)
        k = int(np.searchsorted(np.cumsum(story_bytes[ids]), share * target)) + 1
        parts.append(ids[:min(k, len(ids))])
    return {"gpt4_only": a, "mixed": np.sort(np.concatenate(parts))}


def gather_stories(tokens: np.ndarray, offsets: np.ndarray, ids: np.ndarray, eot_id: int) -> np.ndarray:
    """Token layout EOT s_a EOT s_b ... EOT for the chosen stories, in the given order."""
    return np.concatenate([np.array([eot_id], dtype=np.uint16)]
                          + [np.asarray(tokens[offsets[i] + 1:offsets[i + 1] + 1]) for i in ids])


def load_source_sets(data_dir: Path, tokenizer: str, eot_id: int, split: str = "val") -> dict:
    """{name: (tokens, utf8 bytes, story count)} for the sets from select_source_sets."""
    tok_dir = Path(data_dir) / "tokens" / tokenizer
    tokens = np.memmap(tok_dir / f"{split}.bin", dtype=np.uint16, mode="r")
    offsets = np.load(tok_dir / f"{split}.offsets.npy")
    sources = np.load(tok_dir / f"{split}.sources.npy")
    with (Path(data_dir) / "stories" / f"{split}.jsonl").open(encoding="utf-8") as f:
        story_bytes = np.array([len(json.loads(line)["story"].encode("utf-8")) for line in f])
    return {name: (gather_stories(tokens, offsets, ids, eot_id), int(story_bytes[ids].sum()), len(ids))
            for name, ids in select_source_sets(sources, story_bytes).items()}


def evaluate_source_sets(model, data_dir: Path, tokenizer: str, eot_id: int, context_length: int,
                         batch_size: int, device: str, bf16: bool = True) -> dict:
    out = {}
    for name, (tokens, n_bytes, n_stories) in load_source_sets(data_dir, tokenizer, eot_id).items():
        out[name] = {"stories": n_stories, "utf8_bytes": n_bytes,
                     "bpb": evaluate_bpb(model, tokens, n_bytes, context_length, batch_size, device, bf16)}
    return out


@torch.no_grad()
def sum_nats(model, tokens: np.ndarray, context_length: int, batch_size: int,
             device: str, bf16: bool = True) -> tuple[float, int]:
    """Sum of next-token losses (nats) over all predicted tokens, and their count."""
    n_pred = len(tokens) - 1
    starts = list(range(0, n_pred, context_length))
    full = [s for s in starts if s + context_length <= n_pred]
    last = [s for s in starts if s + context_length > n_pred]
    data = torch.from_numpy(tokens.astype(np.int64))
    total, count = 0.0, 0

    def run(windows: torch.Tensor):
        nonlocal total, count
        windows = windows.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device == "cuda"):
            _, loss = model(windows[:, :-1], windows[:, 1:], reduction="none")
        total += loss.double().sum().item()
        count += loss.numel()

    for i in range(0, len(full), batch_size):
        run(torch.stack([data[s:s + context_length + 1] for s in full[i:i + batch_size]]))
    for s in last:  # final partial window
        run(data[s:][None])
    assert count == n_pred
    return total, count


def bits_per_byte(nats: float, n_bytes: int) -> float:
    return nats / (math.log(2) * n_bytes)


def evaluate_bpb(model, tokens: np.ndarray, n_bytes: int, context_length: int,
                 batch_size: int, device: str, bf16: bool = True) -> float:
    was_training = model.training
    model.eval()
    nats, _ = sum_nats(model, tokens, context_length, batch_size, device, bf16)
    model.train(was_training)
    return bits_per_byte(nats, n_bytes)


@torch.no_grad()
def generate(model, tok, prompt: str, max_tokens: int, temperature: float, top_k: int,
             generator: torch.Generator, device: str) -> str:
    """Sample a continuation of EOT + prompt, stopping at EOT or max_tokens."""
    ctx = model.cfg.context_length
    ids = [tok.eot_id] + tok.encode(prompt)
    out = []
    for _ in range(max_tokens):
        x = torch.tensor([ids[-ctx:]], device=device)
        logits, _ = model(x)
        logits = logits[0, -1].float() / temperature
        kth = torch.topk(logits, min(top_k, logits.numel())).values[-1]
        logits[logits < kth] = -float("inf")
        nxt = torch.multinomial(torch.softmax(logits, -1), 1, generator=generator).item()
        if nxt == tok.eot_id:
            break
        ids.append(nxt)
        out.append(nxt)
    return prompt + tok.decode(out)


def make_samples(model, tok, cfg, device: str) -> list[dict]:
    was_training = model.training
    model.eval()
    gen = torch.Generator(device=device).manual_seed(cfg.sample_seed)
    samples = [{"prompt": p, "text": generate(model, tok, p, cfg.sample_max_tokens,
                                              cfg.sample_temperature, cfg.sample_top_k, gen, device)}
               for p in PROMPTS]
    model.train(was_training)
    return samples
