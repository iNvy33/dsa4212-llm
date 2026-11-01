"""Optimisers with parameter groups, and time-based learning-rate schedules."""
import math

import torch
import torch.nn as nn

from src.config import Config


def param_groups(model: nn.Module) -> tuple[list, list]:
    """Split parameters into (2D hidden weight matrices, everything else).

    Only the first group gets weight decay. "Everything else" is the tied
    token embedding / output layer and the RMSNorm weights.
    """
    embedding = model.tok_emb.weight
    hidden, other = [], []
    for p in model.parameters():
        (hidden if p.ndim == 2 and p is not embedding else other).append(p)
    return hidden, other


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    hidden, other = param_groups(model)
    if cfg.optimizer == "adamw":
        groups = [{"params": hidden, "weight_decay": cfg.weight_decay},
                  {"params": other, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=cfg.lr, betas=(cfg.beta1, cfg.beta2),
                                 fused=torch.cuda.is_available())
    raise NotImplementedError(f"optimizer {cfg.optimizer!r} is added in E4")


def lr_multiplier(progress: float, cfg: Config) -> float:
    """Fraction of peak LR at progress = elapsed / budget_seconds (clamped to [0, 1]).

    cosine: linear warmup over warmup_frac, then cosine decay to final_lr_frac.
    wsd:    linear warmup, constant, then linear decay to final_lr_frac over
            the last wsd_decay_frac of the budget.
    """
    p = min(max(progress, 0.0), 1.0)
    if p < cfg.warmup_frac:
        return p / cfg.warmup_frac
    floor = cfg.final_lr_frac
    if cfg.schedule == "cosine":
        t = (p - cfg.warmup_frac) / (1.0 - cfg.warmup_frac)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * t))
    decay_start = 1.0 - cfg.wsd_decay_frac
    if p < decay_start:
        return 1.0
    t = (p - decay_start) / cfg.wsd_decay_frac
    return 1.0 - (1.0 - floor) * t
