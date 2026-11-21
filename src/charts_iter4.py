"""Графики итерации 4: форвард-curve полного сезона, сравнение подходов."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

plt.rcParams.update({"figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
                     "axes.edgecolor": "#30363d", "axes.labelcolor": "#c9d1d9",
                     "xtick.color": "#c9d1d9", "ytick.color": "#c9d1d9",
                     "text.color": "#c9d1d9", "grid.color": "#21262d",
                     "font.size": 10, "savefig.dpi": 130})


def make_charts_iter4(out_dir=OUT):
    res = json.load(open(os.path.join(out_dir, "iter4.json")))
    curve = np.load(os.path.join(out_dir, "forward_curve.npy"))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))
    ax1.plot(curve, color="#58a6ff", lw=1.8)
    ax1.axhline(10000, color="#8b949e", ls="--", lw=1)
    ax1.set_title("Live-пул 2025/26: банк (полный сезон, 325 ставок)")
    ax1.set_xlabel("Ставка"); ax1.set_ylabel("Банк"); ax1.grid(True)
    mon = res["forward"]["monthly"]
    ks = list(mon.keys()); vs = list(mon.values())
    ax2.bar(ks, vs, color=["#3fb950" if v >= 0 else "#f85149" for v in vs])
    ax2.set_title("PnL по месяцам (10 000 банк, %)")
    ax2.tick_params(axis="x", rotation=45); ax2.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "forward_full.png"))
    plt.close()

    fwd = res["forward"]
    items = [
        ("XGBoost v1 (test 24/25)", -0.3609, "ROI"),
        ("Dixon-Coles (test 24/25)", res["dixon_coles"]["backtest"]["roi"], "ROI"),
        ("Стэкинг модель×рынок", res["stacking"]["backtest"]["roi"], "ROI"),
        ("Перекалибровка рынка", res["recalibration"]["backtest"]["roi"], "ROI"),
        ("Базлайн «фаворит»", res["dixon_coles"]["baseline_roi"], "ROI"),
        ("Live-пул 25/26 (форвард)", fwd["roi"], "ROI"),
        ("Steam chase 2% (yield)", res["steam"]["chase_0.02"]["yield"], "yield"),
        ("Steam fade 2% (yield)", res["steam"]["fade_0.02"]["yield"], "yield"),
    ]
    labels = [i[0] for i in items]
    vals = [i[1] * 100 for i in items]
    colors = ["#3fb950" if v > 0 else "#f85149" for v in vals]
    y = np.arange(len(items))[::-1]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.barh(y, vals, color=colors, height=0.62)
    for yi, v in zip(y, vals):
        ax.text(v + (1.2 if v >= 0 else -1.2), yi, f"{v:+.1f}%",
                va="center", ha="left" if v >= 0 else "right", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(0, color="#8b949e", lw=1)
    ax.set_title("Итерация 4: все подходы (ROI / yield, %)")
    ax.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "iter4_variants.png"))
    plt.close()
    print("charts_iter4 done")


if __name__ == "__main__":
    make_charts_iter4()
