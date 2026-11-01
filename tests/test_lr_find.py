import numpy as np
import pytest

from src.lr_find import exp_lr, smooth, suggest


def test_exp_lr_endpoints_and_geometric():
    assert exp_lr(0, 300, 1e-5, 1e-1) == pytest.approx(1e-5)
    assert exp_lr(299, 300, 1e-5, 1e-1) == pytest.approx(1e-1)
    assert exp_lr(150, 301, 1e-5, 1e-1) == pytest.approx(1e-3)  # midpoint in log space


def test_smooth_constant_is_unbiased():
    assert np.allclose(smooth([2.0] * 20), 2.0)


def test_suggest_on_synthetic_curve():
    # loss falls fastest around 1e-3, is lowest at 1e-2, then rises
    lrs = np.logspace(-5, -1, 200)
    x = np.log10(lrs)
    loss = 5 - 2 * np.tanh(2 * (x + 3)) + 3 * np.clip(x + 2, 0, None) ** 2
    s = suggest(lrs, loss)
    assert s["rise_lr"] == pytest.approx(1e-2, rel=0.15)
    assert s["steepest_lr"] == pytest.approx(1e-3, rel=0.15)
    assert s["suggested_lr"] == pytest.approx(s["rise_lr"] / 3)


def test_suggest_ignores_recovery_after_spike():
    # loss falls, spikes at ~7e-3, recovers and keeps falling to a lower value at 1e-1
    lrs = np.logspace(-5, -1, 300)
    x = np.log10(lrs)
    loss = 5 - 1.2 * np.tanh(2 * (x + 3)) + 0.6 * np.exp(-((x + 2.1) / 0.1) ** 2) - 0.3 * (x > -2) * (x + 2)
    loss[x > -1.5] = loss[x > -1.5].min()  # flat tail, below the pre-spike minimum
    s = suggest(lrs, loss)
    assert 3e-3 < s["rise_lr"] < 8e-3
