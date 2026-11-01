"""Out-of-domain evaluation: bpb on the Children's Book Test raw test books.

Usage: .venv\\Scripts\\python.exe scripts\\eval_cbt.py runs\\E1\\L8 runs\\E5\\wsd_gpt4

Loads each run's model.pt, scores the 5 detokenized CBT test books, and writes
"cbt_test" (overall and per-book bpb) into the run's log.json.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cbt import evaluate_books, load_books  # noqa: E402
from src.config import load_config  # noqa: E402
from src.model import GPT  # noqa: E402
from src.tokenizer import get_tokenizer  # noqa: E402
from src.train import EVAL_BATCH  # noqa: E402


def main():
    run_dirs = [Path(p) for p in sys.argv[1:]]
    books = load_books("test")
    results = {}
    for run_dir in run_dirs:
        log_path = run_dir / "log.json"
        log = json.loads(log_path.read_text(encoding="utf-8"))
        cfg = load_config(overrides=log["config"])
        tok = get_tokenizer(cfg.tokenizer, Path(cfg.data_dir))
        model = GPT(cfg, tok.vocab_size).cuda()
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cuda"))
        results[run_dir] = evaluate_books(model, tok, books, cfg.context_length, EVAL_BATCH, "cuda", cfg.bf16)
        log["cbt_test"] = results[run_dir]
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")

    names = [f"{d.parent.name}/{d.name}" for d in run_dirs]
    print(f"{'CBT test book':55s} {'MB':>5s} " + " ".join(f"{n:>15s}" for n in names))
    for i, book in enumerate(books):
        print(f"{book['title'][:55]:55s} {len(book['text'].encode()) / 1e6:5.2f} "
              + " ".join(f"{results[d]['books'][i]['bpb']:15.4f}" for d in run_dirs))
    total = results[run_dirs[0]]["utf8_bytes"]
    print(f"{'ALL 5 BOOKS':55s} {total / 1e6:5.2f} " + " ".join(f"{results[d]['bpb']:15.4f}" for d in run_dirs))


if __name__ == "__main__":
    main()
