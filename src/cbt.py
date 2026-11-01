"""Children's Book Test (CBT) raw books as an out-of-domain evaluation set.

The `raw` config of cam-cst/cbt holds whole Project Gutenberg children's books,
PTB-tokenised with one sentence per line (spaces before punctuation, "wo n't",
`` '' and ` ' quotes, -LRB- -RRB-). detokenize() restores normal text so bpb
measures the language, not the tokenisation. Evaluation only: never trained on.
"""
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

from src.evaluate import bits_per_byte, sum_nats

REPO_ID = "cam-cst/cbt"
REVISION = "72b5c46b1248e3316360f0f2f0b2c39e773b68e4"  # pinned for reproducibility

OPEN = {"``": '"', "`": "'", "-LRB-": "(", "-LSB-": "[", "-LCB-": "{"}
CLOSE = {"''": '"', "'": "'", "-RRB-": ")", "-RSB-": "]", "-RCB-": "}"}
ATTACH_LEFT = {",", ".", ";", ":", "!", "?", "...", "%"}


def _is_clitic(token: str) -> bool:
    return token.lower() == "n't" or (token.startswith("'") and token[1:].isalpha())


def detokenize_line(line: str) -> str:
    out, glue = [], True  # glue: no space before the next token
    for token in line.split():
        if token in OPEN:
            out.append(("" if glue else " ") + OPEN[token])
            glue = True
        elif token in CLOSE or token in ATTACH_LEFT or _is_clitic(token):
            out.append(CLOSE.get(token, token))
            glue = False
        else:
            out.append(("" if glue else " ") + token)
            glue = False
    return "".join(out)


def detokenize(content: str) -> str:
    """Whole book: detokenize each sentence line, join sentences with single spaces."""
    return " ".join(s for s in (detokenize_line(l) for l in content.splitlines()) if s)


def load_books(split: str = "test", data_dir: Path = Path("data")) -> list[dict]:
    """[{title, text}] for the CBT raw split, detokenized."""
    path = hf_hub_download(REPO_ID, f"raw/{split}-00000-of-00001.parquet", repo_type="dataset",
                           revision=REVISION, local_dir=Path(data_dir) / "raw" / "cbt")
    return [{"title": b["title"].replace(".txt.out", "").replace("___", ": ").replace("_", " "),
             "text": detokenize(b["content"])} for b in pq.read_table(path).to_pylist()]


def evaluate_books(model, tok, books: list[dict], context_length: int, batch_size: int,
                   device: str, bf16: bool = True) -> dict:
    """bpb per book (each laid out as EOT book EOT) and over all books together."""
    per_book, total_nats, total_bytes = [], 0.0, 0
    was_training = model.training
    model.eval()
    for book in books:
        tokens = np.array([tok.eot_id] + tok.encode(book["text"]) + [tok.eot_id], dtype=np.uint16)
        n_bytes = len(book["text"].encode("utf-8"))
        nats, _ = sum_nats(model, tokens, context_length, batch_size, device, bf16)
        per_book.append({"title": book["title"], "utf8_bytes": n_bytes, "bpb": bits_per_byte(nats, n_bytes)})
        total_nats += nats
        total_bytes += n_bytes
    model.train(was_training)
    return {"bpb": bits_per_byte(total_nats, total_bytes), "utf8_bytes": total_bytes, "books": per_book}
