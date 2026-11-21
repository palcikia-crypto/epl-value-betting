"""Графики итерации 3."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def make_charts_iter3(out_dir, multi_rows, ablation, live_monthly):
    ml = [r for r in multi_rows if r["league"] != "POOLED"]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = [r["league"] for r in ml]
    w = 0.38
    ax.bar([i - w / 2 for i in range(len(ml))], [r["roi"] * 100 for r in ml],
           width=w, label="Value-model ROI", color="#d95f02")
    ax.bar([i + w / 2 for i in range(len(ml))], [r["baseline_roi"] * 100 for r in ml],
           width=w, label="Baseline (фаворит)", color="#1f77b4")
    ax.set_xticks(range(len(ml)), xs)
    ax.axhline(0, color="black", lw=1)
    ax.set_title("Мульти-лига: Test 2024/25 — модель vs baseline")
    ax.set_ylabel("ROI, %")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "multi_league.png"), dpi=130)
    plt.close(fig)

    names = list(ablation.keys())
    vals = [ablation[n]["logloss"] for n in names]
    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ["#d62728" if n != "market_only" else "#2ca02c" for n in names]
    ax.barh(names, vals, color=colors)
    ax.set_title("Аблация: logloss по фич-блокам (EPL val 2023/24). Ниже = лучше")
    ax.set_xlabel("logloss (рынок = 0.946)")
    for i, v in enumerate(vals):
        ax.text(v + 0.002, i, f"{v:.3f}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "ablation.png"), dpi=130)
    plt.close(fig)

    if live_monthly is not None and len(live_monthly):
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.bar(live_monthly["m"], live_monthly["pnl"] * 100,
               color=["#2ca02c" if v >= 0 else "#d62728" for v in live_monthly["pnl"]])
        ax.set_title("Live EPL 2025/26: PnL по месяцам (flat 1%, % от банка)")
        ax.set_ylabel("% банка")
        ax.set_xlabel("Месяц")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "live_pool.png"), dpi=130)
        plt.close(fig)
