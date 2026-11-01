"""Loss-over-time plots for one run (curves.png) and for several runs overlaid."""
import math
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


def _train_bpb_scale(log: dict) -> float:
    """Factor converting train loss in nats/token to approximate bits/byte for this run."""
    bytes_per_token = log["bytes_seen_estimate"] / log["tokens_seen"]
    return 1.0 / (math.log(2) * bytes_per_token)


def _curves(log: dict, train_in_bpb: bool = False):
    train = log.get("train_curve", [])
    t_min = [p[0] / 60 for p in train]
    scale = _train_bpb_scale(log) if train_in_bpb else 1.0
    loss = [p[1] * scale for p in train]
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


def plot_interaction(lines: dict[str, list[dict]], path: Path, title: str, metric: str = "final_test_bpb"):
    """Interaction plot: metric against n_layers, one line per factor level (e.g. BPE vs char)."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    fig.patch.set_facecolor(SURFACE)
    for i, (level, logs) in enumerate(lines.items()):
        pts = sorted((log["config"]["n_layers"], log[metric]) for log in logs)
        xs, ys = zip(*pts)
        ax.plot(xs, ys, color=SERIES[i], lw=2, marker="o", ms=8, markeredgecolor=SURFACE,
                markeredgewidth=2, label=level)
        for x, y in pts:
            ax.annotate(f"{y:.4f}", (x, y), xytext=(0, 9), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8, color=INK)
        ax.annotate(level, (xs[0], ys[0]), xytext=(-10, 0), textcoords="offset points",
                    ha="right", va="center", fontsize=9, color=INK)
    xs_all = sorted({log["config"]["n_layers"] for logs in lines.values() for log in logs})
    ax.set_xticks(xs_all)
    ax.set_xlim(xs_all[0] - 0.35 * (xs_all[-1] - xs_all[0]), xs_all[-1] + 0.3 * (xs_all[-1] - xs_all[0]))
    ax.set_xlabel("layers")
    ax.set_ylabel(metric.replace("final_", "").replace("_", " "), fontsize=9)
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="best")
    fig.suptitle(title, fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def label_ends(ax, ends: list[tuple], t_end: float):
    """Direct labels (x, y, text) at line ends, spread apart vertically so they never overlap."""
    lo, hi = ax.get_ylim()
    gap = 0.045 * (hi - lo)
    ends = sorted(ends, key=lambda e: e[1])
    placed = []
    for x, y, text in ends:
        placed.append(max(y, placed[-1] + gap) if placed else y)
    shift = max(0.0, placed[-1] - (hi - gap)) if placed else 0.0
    for (x, y, text), ly in zip(ends, placed):
        ax.annotate(text, (x, y), xytext=(x + 0.012 * t_end * 1.15, ly - shift),
                    textcoords="data", va="center", fontsize=8, color=INK)


def plot_overlay(logs: list[dict], labels: list[str], path: Path, title: str):
    """Training loss (approx. bits/byte, comparable across tokenizers) for several runs."""
    assert len(logs) <= len(SERIES), "more than 8 runs: split into small multiples"
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    fig.patch.set_facecolor(SURFACE)
    all_t, all_loss, ends = [], [], []
    for i, (log, label) in enumerate(zip(logs, labels)):
        t, loss, _, _, _ = _curves(log, train_in_bpb=True)
        all_t += t
        all_loss += loss
        ax.plot(t, loss, color=SERIES[i], lw=2, label=label)
        if t:
            ends.append((t[-1], loss[-1], f"{label} {loss[-1]:.3f}"))
    ax.set_ylabel("train loss (approx. bits/byte, 10 s mean)", fontsize=9)
    t_end = max(all_t, default=1)
    zoomed = _zoom(ax, all_t, all_loss, t_end)
    ax.set_xlabel("training time (minutes)" + ("  -  axis zoomed to the last 90% of training" if zoomed else ""))
    _style(ax)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_xlim(0, t_end * 1.15)
    if len(ends) <= 4:
        label_ends(ax, ends, t_end)
    fig.suptitle(title, fontsize=11, color=INK, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
