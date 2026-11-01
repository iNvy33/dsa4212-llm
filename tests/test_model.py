from pathlib import Path

import pytest
import torch

from src.config import load_config, parse_overrides
from src.model import GPT

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"
VOCAB = 4096


def small_cfg(**kw):
    base = dict(n_layers=2, d_model=64, ffn_size=128, context_length=32)
    return load_config(BASELINE, {**base, **kw})


def test_config_loading_and_validation():
    cfg = load_config(BASELINE)
    assert (cfg.n_layers, cfg.d_model, cfg.ffn_size, cfg.context_length) == (4, 384, 768, 1024)
    assert cfg.head_size == 96 and cfg.grad_accum_steps == 2
    assert load_config(BASELINE, parse_overrides(["n_layers=8", "lr=1e-3"])).n_layers == 8
    with pytest.raises(KeyError):
        load_config(BASELINE, {"n_layer": 8})
    with pytest.raises(AssertionError):
        load_config(BASELINE, {"d_model": 360})  # head size 90 is not a multiple of 8


def test_output_shapes():
    cfg = small_cfg()
    model = GPT(cfg, VOCAB)
    idx = torch.randint(0, VOCAB, (3, 20))
    logits, loss = model(idx)
    assert logits.shape == (3, 20, VOCAB) and loss is None
    _, loss = model(idx, idx)
    assert loss.shape == ()
    _, per_token = model(idx, idx, reduction="none")
    assert per_token.shape == (3, 20)
    with pytest.raises(ValueError):
        model(torch.randint(0, VOCAB, (1, cfg.context_length + 1)))


# (width, ffn_size, change in params per layer vs. baseline) from the E2 table in PLAN.md
E2_VARIANTS = [(448, 416, -0.003), (384, 768, 0.0), (320, 1216, 0.007), (256, 1792, 0.0)]


@pytest.mark.parametrize("d_model,ffn_size,rel", E2_VARIANTS)
def test_parameter_counts_match_plan(d_model, ffn_size, rel):
    cfg = load_config(BASELINE, {"d_model": d_model, "ffn_size": ffn_size})
    counts = GPT(cfg, VOCAB).param_counts()
    per_layer = 4 * d_model**2 + 2 * d_model * ffn_size + 2 * d_model  # + two RMSNorm weights
    assert counts["non_embedding"] == cfg.n_layers * per_layer + d_model  # + final RMSNorm
    assert counts["embedding"] == VOCAB * d_model
    baseline_non_emb = 4 * 1_179_648  # PLAN.md: 1.18M per layer at 4 x 384, FFN 768
    actual_rel = counts["non_embedding"] / baseline_non_emb - 1
    assert abs(actual_rel) < 0.01
    assert abs(actual_rel - rel) < 0.001


def test_causal_masking():
    torch.manual_seed(0)
    model = GPT(small_cfg(), VOCAB).eval()
    idx = torch.randint(0, VOCAB, (2, 16))
    changed = idx.clone()
    changed[:, 10] = (idx[:, 10] + 1) % VOCAB
    with torch.no_grad():
        a, _ = model(idx)
        b, _ = model(changed)
    assert torch.allclose(a[:, :10], b[:, :10], atol=1e-5)
    assert not torch.allclose(a[:, 10:], b[:, 10:], atol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_overfit_one_batch_on_gpu():
    torch.manual_seed(0)
    model = GPT(small_cfg(), VOCAB).cuda()
    batch = torch.randint(0, VOCAB, (4, 33), device="cuda")
    x, y = batch[:, :-1], batch[:, 1:]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(300):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert loss.item() < 0.05, loss.item()
