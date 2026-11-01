"""Training FLOPs per run, estimated from each run's config and tokens seen.

Usage: .venv\\Scripts\\python.exe scripts\\flops_table.py [runs root, default runs]

Per training token (forward + backward = 3 x forward = 6 FLOPs per weight):
  matrices:  6 * (n_layers * (4 d^2 + 2 d ffn) + vocab * d)   (tied output layer;
             the input embedding is a lookup and costs no FLOPs)
  attention: 6 * n_layers * d * ctx                            (QK^T and AV, causal:
             on average ctx / 2 keys per query)
Norms, GELU, softmax and the loss are ignored (< 1%).
"""
import json
import sys
from pathlib import Path

VOCAB = {"bpe4096": 4096, "char": 257}


def flops_per_token(cfg: dict) -> tuple[float, float]:
    d, L, f, ctx = cfg["d_model"], cfg["n_layers"], cfg["ffn_size"], cfg["context_length"]
    matrices = 6 * (L * (4 * d * d + 2 * d * f) + VOCAB[cfg["tokenizer"]] * d)
    attention = 6 * L * d * ctx
    return matrices, attention


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("runs")
    stages = ("E1", "E2", "E3", "I1", "E5")
    rows = []
    for stage in stages:
        for log_path in sorted((root / stage).glob("*/log.json")):
            log = json.loads(log_path.read_text(encoding="utf-8"))
            cfg = log["config"]
            m, a = flops_per_token(cfg)
            tokens = log["tokens_seen"]
            total = (m + a) * tokens
            rows.append((f"{stage}/{cfg['run_name']}", cfg, total))
    print(f"{'config':58s} {'total training FLOPs':>20s}")
    for name, cfg, total in rows:
        desc = (f"{name}: {cfg['n_layers']}x{cfg['d_model']}/{cfg['ffn_size']} {cfg['tokenizer']}"
                f"{', seed ' + str(cfg['seed']) if cfg['seed'] else ''}"
                f"{', ' + cfg['schedule'] + ' + ' + cfg['decay_data'] + ' decay' if cfg['schedule'] != 'cosine' else ''}")
        print(f"{desc:58s} {total:20.2e}")


if __name__ == "__main__":
    main()
