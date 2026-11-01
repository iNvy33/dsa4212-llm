"""GPT-style decoder: pre-norm RMSNorm blocks, causal SDPA attention, GELU FFN,
fixed sinusoidal positions, tied input/output embeddings, no biases."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import Config


def sinusoidal_positions(length: int, d_model: int) -> torch.Tensor:
    """Fixed encoding from "Attention Is All You Need": sin on even dims, cos on odd."""
    pos = torch.arange(length, dtype=torch.float32)[:, None]
    inv_freq = torch.exp(-math.log(10000.0) * torch.arange(0, d_model, 2, dtype=torch.float32) / d_model)
    pe = torch.zeros(length, d_model)
    pe[:, 0::2] = torch.sin(pos * inv_freq)
    pe[:, 1::2] = torch.cos(pos * inv_freq)
    return pe


class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(x).split(d, dim=-1)
        q, k, v = (z.view(b, t, self.n_heads, -1).transpose(1, 2) for z in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(y.transpose(1, 2).reshape(b, t, d))


class FFN(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.fc = nn.Linear(cfg.d_model, cfg.ffn_size, bias=False)
        self.proj = nn.Linear(cfg.ffn_size, cfg.d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(F.gelu(self.fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.ffn = FFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        return x + self.ffn(self.norm2(x))


class GPT(nn.Module):
    def __init__(self, cfg: Config, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.tok_emb = nn.Embedding(vocab_size, cfg.d_model)
        self.register_buffer("pos_emb", sinusoidal_positions(cfg.context_length, cfg.d_model),
                             persistent=False)
        self.emb_scale = math.sqrt(cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.RMSNorm(cfg.d_model)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
        # GPT-2: scale residual-branch output projections by 1/sqrt(2 * n_layers)
        out_std = 0.02 / math.sqrt(2 * self.cfg.n_layers)
        for block in self.blocks:
            nn.init.normal_(block.attn.proj.weight, mean=0.0, std=out_std)
            nn.init.normal_(block.ffn.proj.weight, mean=0.0, std=out_std)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                reduction: str = "mean"):
        """idx, targets: (batch, time) token ids. Returns (logits, loss or None).

        Loss is cross-entropy in nats, computed in float32; reduction="none"
        gives per-token losses of shape (batch, time).
        """
        t = idx.shape[1]
        if t > self.cfg.context_length:
            raise ValueError(f"sequence length {t} > context_length {self.cfg.context_length}")
        x = self.tok_emb(idx) * self.emb_scale + self.pos_emb[:t]
        for block in self.blocks:
            x = block(x)
        logits = F.linear(self.norm(x), self.tok_emb.weight)  # tied output layer
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, self.vocab_size), targets.reshape(-1),
                               reduction=reduction)
        if reduction == "none":
            loss = loss.view(targets.shape)
        return logits, loss

    def param_counts(self) -> dict:
        """Total and non-embedding parameter counts (the tied output layer is counted once)."""
        total = sum(p.numel() for p in self.parameters())
        embedding = self.tok_emb.weight.numel()
        return {"total": total, "embedding": embedding, "non_embedding": total - embedding}
