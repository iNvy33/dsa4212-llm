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
            rows.append((f"{stage}/{cfg['run_name']}", cfg, tokens, m, a, total,
                         total / log["train_seconds"], log["final_test_bpb"]))
    print(f"{'run':18s} {'config':28s} {'tokens':>8s} {'MFLOP/tok':>9s} {'attn %':>6s} "
          f"{'total FLOPs':>11s} {'TFLOP/s':>7s} {'test bpb':>8s}")
    for name, cfg, tokens, m, a, total, rate, bpb in rows:
        desc = (f"{cfg['n_layers']}x{cfg['d_model']}/{cfg['ffn_size']} {cfg['tokenizer']}"
                f"{' ' + cfg['schedule'] if cfg['schedule'] != 'cosine' else ''}")
        print(f"{name:18s} {desc:28s} {tokens / 1e6:7.0f}M {(m + a) / 1e6:9.1f} {100 * a / (m + a):5.0f}% "
              f"{total:11.2e} {rate / 1e12:7.1f} {bpb:8.4f}")


if __name__ == "__main__":
    main()
