"""Графики итерации 6: live-кривая НХЛ, back-to-back эффект, портфель + Монте-Карло."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
j = json.load(open(os.path.join(OUT, "iter6.json")))

# 1. LIVE-кривая НХЛ
fig, ax = plt.subplots(figsize=(9, 5))
ig = j["nhl_ingame"]
xs = [10, 20, 30, 40, 50, 58, 62]
ys = [c["accuracy"] * 100 for c in ig["curve"]]
ax.plot(xs, ys, "o-", lw=2.5, color="#1f77b4", label="Модель (счёт + время + броски)")
ax.axhline(56.0, ls="--", lw=1.2, color="green", label="Пре-матч Elo 56.0%")
ax.axhline(50.0, ls=":", lw=1.2, color="gray", label="База 50%")
ax.set_xlabel("Минут матча прошло (дебют → финал)")
ax.set_ylabel("Точность предсказания победителя, %")
ax.set_title(f"НХЛ LIVE: предсказуемость матча по ходу игры (n_states={ig['n_states']}, games={ig['n_test_games']} test)")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "iter6_ingame.png"), dpi=130)
plt.close(fig)

# 2. Back-to-back эффект
rc = j["nhl_deep"]["rest_causality"]
fig, ax = plt.subplots(figsize=(9, 5))
names = ["Хозяин B2B\n(гость свежий)", "Гость B2B\n(хозяин свежий)", "Оба B2B", "Оба свежие"]
keymap = ["home_b2b_away_rested", "away_b2b_home_rested", "both_b2b", "both_rested"]
vals = [next(r["home_win_rate"] for r in rc if r["bucket"] == k) * 100 for k in keymap]
cis = [next(r["ci95"] for r in rc if r["bucket"] == k) for k in keymap]
ns = [next(r["n"] for r in rc if r["bucket"] == k) for k in keymap]
cols = ["#d62728", "#2ca02c", "#7f7f7f", "#1f77b4"]
bars = ax.bar(names, vals, color=cols, alpha=0.85)
base = j["nhl_deep"]["home_win_all"] * 100
ax.axhline(base * 1, ls="--", color="black", lw=1.2, label=f"Средняя доля побед хозяев {base:.1f}%")
for b, v, ci, n in zip(bars, vals, cis, ns):
    ax.errorbar(b.get_x() + b.get_width() / 2, v,
                yerr=[[max(v - ci[0] * 100, 0)], [max(ci[1] * 100 - v, 0)]],
                fmt="none", ecolor="black", capsize=4)
    ax.text(b.get_x() + b.get_width() / 2, ci[1] * 100 + 0.7, f"n={n}", ha="center", fontsize=9)
ax.set_ylabel("Доля побед хозяев, %")
ax.set_ylim(40, 75)
ax.set_title("НХЛ: эффект расписания (back-to-back = игра вторые сутки подряд), 95% ДИ")
ax.legend()
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "iter6_rest.png"), dpi=130)
plt.close(fig)

# 3. Портфель: кривые банка + Монте-Карло
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5))
pf = j["portfolio"]
ax1.plot(pf["curve_all_flat"], lw=1.8, color="#d62728", label=f"Все value-ставки ({pf['all_flat']['bets']}, ROI {pf['all_flat']['roi'] * 100:+.1f}%)")
ax1.plot(pf["curve_top3"], lw=1.8, color="#ff7f0e", label=f"Топ-3 по эджу в день ({pf['top3_per_day']['bets']}, ROI {pf['top3_per_day']['roi'] * 100:+.1f}%)")
ax1.plot(pf["curve_top1"], lw=1.8, color="#2ca02c", label=f"1 ставка в день ({pf['top1_per_day']['bets']}, ROI {pf['top1_per_day']['roi'] * 100:+.1f}%)")
ax1.axhline(10000, color="black", ls="--", lw=1)
ax1.set_title("Портфельные стратегии, EPL 2024/25 (старт 10 000)")
ax1.set_xlabel("Ставка №")
ax1.set_ylabel("Банк")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

h = pf["mc_hist"]
centers = [(h["edges"][i] + h["edges"][i + 1]) / 2 for i in range(len(h["counts"]))]
ax2.bar(centers, h["counts"], width=(h["edges"][1] - h["edges"][0]) * 0.92, color="#1f77b4", alpha=0.8)
mc = pf["mc_no_edge"]
ax2.axvline(10000, color="black", ls="--", lw=1.2, label="Стартовый банк 10 000")
ax2.set_title(f"Монте-Карло БЕЗ эджа: {mc['sims']} сим. по {mc['n_bets']} ставок\n"
              f"P(плюс)={mc['p_positive'] * 100:.0f}%, медиана={mc['median_final']:,.0f}".replace(",", " "))
ax2.set_xlabel("Финальный банк")
ax2.set_ylabel("Число симуляций")
ax2.legend()
ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "iter6_portfolio.png"), dpi=130)
plt.close(fig)

print("charts saved")
