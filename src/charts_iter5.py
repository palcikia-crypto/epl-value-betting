"""Графики итерации 5: точность по видам спорта, кривая обучаемости, калибровка."""
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


def make_charts_iter5(out_dir=OUT):
    res = json.load(open(os.path.join(out_dir, "iter5.json")))

    items = []
    for r in res["football"]:
        items.append((r["league"] + " (футбол)", r["accuracy"], r["market"]["accuracy"]))
    items.append(("ATP теннис (Elo)", res["tennis"]["atp"]["elo_accuracy"],
                  res["tennis"]["atp"]["market_accuracy"]))
    items.append(("WTA теннис (Elo)", res["tennis"]["wta"]["elo_accuracy"],
                  res["tennis"]["wta"]["market_accuracy"]))
    if res.get("nhl"):
        items.append(("НХЛ (Elo)", res["nhl"]["elo_accuracy"], None))
    labels = [i[0] for i in items]
    acc = [i[1] * 100 for i in items]
    mk = [i[2] * 100 if i[2] else np.nan for i in items]
    y = np.arange(len(items))[::-1]
    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.barh(y + 0.2, acc, height=0.38, color="#58a6ff", label="наша модель")
    ax.barh(y - 0.2, mk, height=0.38, color="#8b949e", label="рынок (фаворит/devig)")
    for yi, v in zip(y, acc):
        ax.text(v + 0.4, yi + 0.2, f"{v:.1f}%", va="center", fontsize=9)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(33.3, color="#8b949e", ls=":", lw=1)
    ax.text(34, len(items) - 0.8, "случайность (33%)", fontsize=8, color="#8b949e")
    ax.set_xlabel("Точность прогноза исхода, % (test)")
    ax.set_title("Итерация 5: точность по видам спорта и лигам")
    ax.legend(loc="lower right"); ax.grid(True, axis="x")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cross_sport.png"))
    plt.close()

    lc = res["learning_curve"]
    ks = [r["n_train_seasons"] for r in lc["curve"]]
    accs = [r["accuracy"] * 100 for r in lc["curve"]]
    lls = [r["logloss"] for r in lc["curve"]]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4))
    a1.plot(ks, accs, "o-", color="#3fb950", lw=2)
    a1.axhline(lc["market"]["accuracy"] * 100, color="#8b949e", ls="--", lw=1,
               label=f"рынок {lc['market']['accuracy']*100:.1f}%")
    a1.set_xlabel("Сезонов истории в train"); a1.set_ylabel("Accuracy, %")
    a1.set_title("Точность EPL vs объём истории (test 2024/25)")
    a1.legend(); a1.grid(True)
    a2.plot(ks, lls, "o-", color="#f85149", lw=2)
    a2.axhline(lc["market"]["logloss"], color="#8b949e", ls="--", lw=1,
               label=f"рынок {lc['market']['logloss']:.3f}")
    a2.set_xlabel("Сезонов истории в train"); a2.set_ylabel("Logloss")
    a2.set_title("Logloss vs объём истории"); a2.legend(); a2.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "learning_curve.png"))
    plt.close()
    print("charts_iter5 done")


if __name__ == "__main__":
    make_charts_iter5()
