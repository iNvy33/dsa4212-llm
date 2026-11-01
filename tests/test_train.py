import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.config import load_config
from src.evaluate import evaluate_bpb, gather_stories, select_source_sets, sum_nats
from src.model import GPT
from src.optim import lr_multiplier, param_groups
from src.train import Timer, TrainStream

BASELINE = Path(__file__).resolve().parents[1] / "configs" / "baseline.yaml"


class FixedLossModel(nn.Module):
    """Stand-in model whose per-token loss is a known function of the target."""

    def __init__(self, context_length: int, loss_fn):
        super().__init__()
        self.context_length = context_length
        self.loss_fn = loss_fn

    def forward(self, x, y, reduction="none"):
        assert x.shape[1] <= self.context_length
        return None, self.loss_fn(y)


def test_bpb_hand_computed():
    # 9 tokens, context 4 -> windows [0:5] and [4:9] -> 8 predicted tokens.
    # Uniform over 4 symbols: ln 4 nats = 2 bits per token -> 16 bits over 4 bytes = 4 bpb.
    tokens = np.arange(9, dtype=np.uint16)
    model = FixedLossModel(4, lambda y: torch.full(y.shape, math.log(4)))
    assert evaluate_bpb(model, tokens, n_bytes=4, context_length=4, batch_size=8,
                        device="cpu") == pytest.approx(4.0)


def test_every_token_predicted_exactly_once():
    # loss = target id, so the sum of losses is the sum of all targets iff each is
    # predicted exactly once (11 tokens, context 4: two full windows + a partial one).
    tokens = np.arange(1, 12, dtype=np.uint16)
    model = FixedLossModel(4, lambda y: y.double())
    nats, count = sum_nats(model, tokens, context_length=4, batch_size=1, device="cpu")
    assert count == 10
    assert nats == pytest.approx(tokens[1:].sum())


@pytest.mark.parametrize("schedule,expected", [
    ("cosine", {0.0: 0.0, 0.01: 0.5, 0.02: 1.0,
                0.5: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * 0.48 / 0.98)), 1.0: 0.1}),
    ("wsd", {0.0: 0.0, 0.01: 0.5, 0.02: 1.0, 0.5: 1.0, 0.8: 1.0, 0.9: 0.55, 1.0: 0.1}),
])
def test_schedule_values(schedule, expected):
    cfg = load_config(BASELINE, {"schedule": schedule})
    for progress, value in expected.items():
        assert lr_multiplier(progress, cfg) == pytest.approx(value), progress
    assert lr_multiplier(1.5, cfg) == pytest.approx(0.1)  # last step may overshoot the budget


def test_source_sets_gpt4_only_and_matched_mixed():
    rng = np.random.default_rng(0)
    sources = (rng.random(2000) < 0.55).astype(np.uint8)          # 1 = GPT-4
    story_bytes = rng.integers(300, 1500, size=2000) + 200 * sources  # GPT-4 stories longer
    sets = select_source_sets(sources, story_bytes, seed=1234)
    a, b = sets["gpt4_only"], sets["mixed"]
    assert (sources[a] == 1).all() and len(a) == sources.sum()
    target = story_bytes[a].sum()
    assert abs(story_bytes[b].sum() - target) <= 2 * story_bytes.max()  # same size, within a story per source
    natural = story_bytes[sources == 1].sum() / story_bytes.sum()
    assert story_bytes[b][sources[b] == 1].sum() / story_bytes[b].sum() == pytest.approx(natural, abs=0.01)
    assert np.array_equal(select_source_sets(sources, story_bytes, seed=1234)["mixed"], b)


def test_gather_stories_layout():
    tokens = np.array([0, 1, 2, 0, 3, 4, 5, 0, 6, 0], dtype=np.uint16)
    offsets = np.array([0, 3, 7, 9])
    assert gather_stories(tokens, offsets, np.array([2, 0]), eot_id=0).tolist() == [0, 6, 0, 1, 2, 0]


def test_timer_excludes_paused_time():
    now = [0.0]
    syncs = []
    timer = Timer(clock=lambda: now[0], sync=lambda: syncs.append(1))
    timer.start(); now[0] += 2.0
    assert timer.elapsed() == pytest.approx(2.0)
    timer.pause(); now[0] += 100.0           # e.g. validation
    assert timer.elapsed() == pytest.approx(2.0)
    timer.start(); now[0] += 3.0
    timer.pause()
    assert timer.elapsed() == pytest.approx(5.0)
    assert len(syncs) == 5                    # every reading synchronises first


def make_stream(seed, seq_len=4):
    # stories [1,2] [3,4,5] [6] [7,8,9,10], EOT = 0
    tokens = np.array([0, 1, 2, 0, 3, 4, 5, 0, 6, 0, 7, 8, 9, 10, 0], dtype=np.uint16)
    offsets = np.array([0, 3, 7, 9, 14])
    return TrainStream(tokens, offsets, seed, seq_len, eot_id=0)


def test_train_stream_covers_epoch_and_reshuffles():
    s = make_stream(seed=0, seq_len=7)
    batch = s.next_batch(2)                   # 14 targets = one epoch (10 story tokens + 4 EOTs)
    assert batch.shape == (2, 8)
    assert batch[0, 0] == 0                   # stream starts with an EOT
    assert batch[1, 0] == batch[0, -1]        # sequences overlap by one token
    assert sorted(batch[:, 1:].ravel().tolist()) == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    assert s.epochs == pytest.approx(1.0)
    s.next_batch(1)
    assert s.epoch == 1                       # rolled over to a new permutation


def test_train_stream_order_depends_on_seed():
    orders = {tuple(make_stream(seed).order) for seed in range(10)}
    assert len(orders) > 1
    assert np.array_equal(make_stream(3).order, make_stream(3).order)


def test_train_stream_subset_reads_only_those_stories():
    # stories [1,2] [3,4,5] [6] [7,8,9,10]; keep stories 1 and 3 only -> [3,4,5] and [7,8,9,10]
    tokens = np.array([0, 1, 2, 0, 3, 4, 5, 0, 6, 0, 7, 8, 9, 10, 0], dtype=np.uint16)
    offsets = np.array([0, 3, 7, 9, 14])
    s = TrainStream(tokens, offsets, seed=0, seq_len=9, eot_id=0, story_ids=np.array([1, 3]))
    assert s.total_tokens == 9                # 7 story tokens + 2 EOTs
    batch = s.next_batch(1)                   # exactly one epoch of the subset
    assert sorted(batch[0, 1:].tolist()) == [0, 0, 3, 4, 5, 7, 8, 9, 10]
    assert s.epochs == pytest.approx(1.0)
    more = s.next_batch(3)                    # later epochs stay inside the subset
    assert set(more.ravel().tolist()) <= {0, 3, 4, 5, 7, 8, 9, 10}


def test_gpt4_decay_requires_wsd():
    with pytest.raises(AssertionError):
        load_config(BASELINE, {"decay_data": "gpt4"})            # cosine schedule
    assert load_config(BASELINE, {"decay_data": "gpt4", "schedule": "wsd"}).decay_data == "gpt4"


def test_weight_decay_groups():
    model = GPT(load_config(BASELINE, {"n_layers": 2, "d_model": 64, "ffn_size": 128,
                                       "context_length": 32}), 100)
    hidden, other = param_groups(model)
    assert all(p.ndim == 2 for p in hidden) and len(hidden) == 2 * 4   # qkv, proj, fc, proj
    assert any(p is model.tok_emb.weight for p in other)
    assert all(p.ndim == 1 for p in other if p is not model.tok_emb.weight)
