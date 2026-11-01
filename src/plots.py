"""Loss-over-time plots for one run (curves.png) and for several runs overlaid."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Validated categorical palette (fixed order, never cycled) and neutral inks.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.yaxis.label.set_color(INK)
    ax.xaxis.label.set_color(INK)


def _zoom(ax, t: list[float], y: list[float], t_end: float, frac: float = 0.1):
    """Fit the y-axis to points after the first `frac` of training, so the tail is readable."""
    tail = [v for tv, v in zip(t, y) if tv >= frac * t_end]
    if len(tail) >= 2:
        lo, hi = min(tail), max(tail)
        pad = 0.08 * (hi - lo or hi * 0.05)
        ax.set_ylim(lo - pad, hi + pad)
        return True
    return False


def _curves(log: dict):
    train = log.get("train_curve", [])
    t_min = [p[0] / 60 for p in train]
    loss = [p[1] for p in train]
    lr = [p[2] if len(p) > 2 else None for p in train]
    val = log.get("val_curve", [])
    return t_min, loss, lr, [p[0] / 60 for p in val], [p[1] for p in val]


def plot_run(log: dict, path: Path):
    """Training loss, validation bpb and learning rate against training minutes."""
    cfg = log["config"]
    t, loss, lr, vt, vbpb = _curves(log)
    has_lr = any(v is not None for v in lr)
    rows = 3 if has_lr else 2
    fig, axes = plt.subplots(rows, 1, figsize=(7, 2.3 * rows + 0.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 3, 1.4][:rows]})
    fig.patch.set_facecolor(SURFACE)
    color = SERIES[0]
    t_end = log.get("train_seconds", 60 * max(t + vt, default=1)) / 60

    axes[0].plot(t, loss, color=color, lw=2)
    axes[0].set_ylabel("train loss\n(nats/token, 10 s mean)", fontsize=9)
    axes[1].plot(vt, vbpb, color=color, lw=2, marker="o", ms=6,
                 markeredgecolor=SURFACE, markeredgewidth=1, label="1 MB val slice")
    axes[1].set_ylabel("val bpb", fontsize=9)
    final = log.get("final_val_bpb")
    if final is not None:
        axes[1].plot([t_end], [final], linestyle="none", marker="D", ms=7, color=INK,
                     markeredgecolor=SURFACE, markeredgewidth=1, label="full val split (final)")
        axes[1].annotate(f"{final:.4f}", (t_end, final), xytext=(-8, -4), textcoords="offset points",
                         ha="right", va="top", fontsize=8, color=INK)
    axes[1].legend(fontsize=8, frameon=False, loc="upper right")
    zoomed = _zoom(axes[0], t, loss, t_end)
    _zoom(axes[1], vt + ([t_end] if final is not None else []),
          vbpb + ([final] if final is not None else []), t_end)
    if has_lr:
        axes[2].plot(t, lr, color=INK_2, lw=1.5)
        axes[2].set_ylabel("LR", fontsize=9)
        axes[2].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    for ax in axes:
        _style(ax)
    axes[-1].set_xlim(left=0)
    axes[-1].set_xlabel("training time (minutes)"
                        + ("  -  loss axes zoomed to the last 90% of training" if zoomed else ""))
    tokens = log.get("tokens_seen")
    sub = f"{cfg['n_layers']} x {cfg['d_model']}, FFN {cfg['ffn_size']}, {cfg['tokenizer']}, " \
          f"{cfg['optimizer']} lr {cfg['lr']:.1e}"
    if tokens:
        sub += f"  |  {tokens / 1e6:,.0f}M tokens, {log['epochs']:.2f} epochs"
    fig.suptitle(f"{cfg['stage']}/{cfg['run_name']}\n{sub}", fontsize=10, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_overlay(logs: list[dict], labels: list[str], path: Path, title: str):
    """Validation bpb and training loss for several runs (one colour per run, fixed order)."""
    assert len(logs) <= len(SERIES), "more than 8 runs: split into small multiples"
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    all_t, all_loss, all_vt, all_v = [], [], [], []
    for i, (log, label) in enumerate(zip(logs, labels)):
        t, loss, _, vt, vbpb = _curves(log)
        all_t += t; all_loss += loss; all_vt += vt; all_v += vbpb
        color = SERIES[i]
        axes[0].plot(vt, vbpb, color=color, lw=2, marker="o", ms=6, label=label,
                     markeredgecolor=SURFACE, markeredgewidth=1)
        axes[1].plot(t, loss, color=color, lw=2, label=label)
        if len(logs) <= 4 and vt:  # direct labels at line ends
            axes[0].annotate(f"{label} {vbpb[-1]:.3f}", (vt[-1], vbpb[-1]), xytext=(6, 0),
                             textcoords="offset points", va="center", fontsize=8, color=INK)
    axes[0].set_ylabel("val bpb (1 MB slice)", fontsize=9)
    axes[1].set_ylabel("train loss (nats/token, 10 s mean)", fontsize=9)
    t_end = max(all_t + all_vt, default=1)
    zoomed = _zoom(axes[0], all_vt, all_v, t_end)
    _zoom(axes[1], all_t, all_loss, t_end)
    axes[1].set_xlabel("training time (minutes)"
                       + ("  -  axes zoomed to the last 90% of training" if zoomed else ""))
    for ax in axes:
        _style(ax)
    axes[0].legend(fontsize=8, frameon=False, loc="upper right")
    axes[0].set_xlim(0, t_end * 1.15)
    fig.suptitle(title, fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
