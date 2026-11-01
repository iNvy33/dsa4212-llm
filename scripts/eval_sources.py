"""Add GPT-4-only / matched-mixed validation bpb to existing runs (from their model.pt).

Usage: .venv\\Scripts\\python.exe scripts\\eval_sources.py runs\\E1\\L8 [more run folders]

Writes "val_source_sets" into each run's log.json (validation split only).
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import load_config  # noqa: E402
from src.evaluate import evaluate_source_sets  # noqa: E402
from src.model import GPT  # noqa: E402
from src.tokenizer import get_tokenizer  # noqa: E402
from src.train import EVAL_BATCH  # noqa: E402


def main():
    for run_dir in map(Path, sys.argv[1:]):
        log_path = run_dir / "log.json"
        log = json.loads(log_path.read_text(encoding="utf-8"))
        cfg = load_config(overrides=log["config"])
        tok = get_tokenizer(cfg.tokenizer, Path(cfg.data_dir))
        model = GPT(cfg, tok.vocab_size).cuda()
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location="cuda"))
        sets = evaluate_source_sets(model, Path(cfg.data_dir), cfg.tokenizer, tok.eot_id,
                                    cfg.context_length, EVAL_BATCH, "cuda", cfg.bf16)
        log["val_source_sets"] = sets
        log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{run_dir}: " + ", ".join(f"{k} {v['bpb']:.4f} ({v['stories']} stories, "
                                         f"{v['utf8_bytes'] / 1e6:.2f} MB)" for k, v in sets.items()))


if __name__ == "__main__":
    main()
