"""Графики итерации 2."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=130)
    plt.close(fig)


def make_charts_iter2(out_dir, experiments, tennis_bets_25, last25_bank_curve):
    # Kelly vs flat
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.staking import run_backtest

    fig, ax = plt.subplots(figsize=(9, 5))
    labels = ["flat", "kelly_quarter", "kelly_half", "kelly_full"]
    sel = [b for b in tennis_bets_25]  # placeholder
    vals = []
    for name in labels:
        key = f"preclose_{name}_5"
        if key in experiments:
            vals.append(experiments[key]["roi"] * 100)
    ax.bar([l.replace("kelly_", "Kelly ") for l in labels[:len(vals)]], vals, color="#d95f02")
    ax.axhline(0, color="black", lw=1)
    ax.set_title("EPL Test 2024/25: ROI по схемам стейкинга (edge>5%, pre-close)")
    ax.set_ylabel("ROI, %")
    _save(fig, out_dir, "kelly_curve.png")

    # tennis bank curve
    if last25_bank_curve is not None and "curve" in last25_bank_curve:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(last25_bank_curve["curve"], label="ATP 2025, Quarter Kelly, edge>5%")
        ax.axhline(10000, color="gray", ls=":")
        ax.set_title("Tennis ATP 2025: кривая банка")
        ax.set_xlabel("Ставка №")
        ax.set_ylabel("Банк")
        ax.legend()
        _save(fig, out_dir, "tennis_curve.png")
