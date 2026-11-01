"""Run configuration: one flat dataclass, loaded from YAML plus overrides.

Every setting is documented in configs/baseline.yaml. Experiments are small
override dicts on top of it (configs/stages/*.yaml).
"""
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml


@dataclass
class Config:
    # Run
    run_name: str = "baseline"
    stage: str = "dev"
    seed: int = 0
    # Data
    data_dir: str = "data"
    tokenizer: str = "bpe4096"
    context_length: int = 1024
    # Model
    n_layers: int = 4
    d_model: int = 384
    n_heads: int = 4
    ffn_size: int = 768
    # Batch
    batch_sequences: int = 32
    micro_batch_sequences: int = 16
    # Optimiser
    optimizer: str = "adamw"
    lr: float = 3e-3
    beta1: float = 0.9
    beta2: float = 0.95
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    momentum: float = 0.9
    # Schedule
    schedule: str = "cosine"
    warmup_frac: float = 0.02
    final_lr_frac: float = 0.1
    wsd_decay_frac: float = 0.2
    # Budget and precision
    budget_seconds: float = 1800.0
    bf16: bool = True
    compile: bool = True
    # Validation and samples
    val_interval_seconds: float = 150.0
    val_slice_bytes: int = 1_000_000
    sample_temperature: float = 0.8
    sample_top_k: int = 50
    sample_max_tokens: int = 200
    sample_seed: int = 0

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
        assert self.head_size % 8 == 0, "head size must be a multiple of 8"
        assert self.batch_sequences % self.micro_batch_sequences == 0, \
            "batch_sequences must be a multiple of micro_batch_sequences"
        assert self.optimizer in ("adamw", "sgd", "muon"), self.optimizer
        assert self.schedule in ("cosine", "wsd"), self.schedule

    @property
    def head_size(self) -> int:
        return self.d_model // self.n_heads

    @property
    def grad_accum_steps(self) -> int:
        return self.batch_sequences // self.micro_batch_sequences

    def to_dict(self) -> dict:
        return asdict(self)


def load_config(path: Path | str | None = None, overrides: dict | None = None) -> Config:
    """Load a YAML file of settings, apply overrides, and validate the keys."""
    values = {}
    if path is not None:
        values.update(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})
    values.update(overrides or {})
    known = {f.name for f in fields(Config)}
    unknown = set(values) - known
    if unknown:
        raise KeyError(f"unknown config keys: {sorted(unknown)}")
    return Config(**values)


def parse_overrides(items: list[str]) -> dict:
    """Parse command-line overrides like ["n_layers=8", "lr=1e-3"]."""
    out = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        out[key] = yaml.safe_load(value)
    return out
