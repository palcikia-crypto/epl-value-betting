"""Графики итерации 9: Polymarket vs Elo и кросс-спортивная усталость."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
j = json.load(open(os.path.join(OUT, "iter9.json")))

# 1. Polymarket vs tuned Elo на одинаковой выборке NHL
pm = j["polymarket"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
accs = [pm["polymarket"]["accuracy"] * 100, pm["tuned_elo_same_subset"]["accuracy"] * 100]
lls = [pm["polymarket"]["logloss"], pm["tuned_elo_same_subset"]["logloss"]]
b = axes[0].bar(["Polymarket\n(толпа)", "Tuned Elo\n(наша модель)"], accs,
                color=["#2ca02c", "#1f77b4"], alpha=0.85, width=0.55)
for r, v in zip(b, accs):
    axes[0].text(r.get_x() + r.get_width() / 2, v + 0.15, "%.2f%%" % v, ha="center", fontsize=11)
axes[0].set_ylim(52, 58)
axes[0].set_title("Точность на %d играх NHL (25/26 + 26/27)" % pm["n_matched_games"])
axes[0].grid(alpha=0.3, axis="y")
axes[0].set_ylabel("Точность, %")
b = axes[1].bar(["Polymarket\n(толпа)", "Tuned Elo\n(наша модель)"], lls,
                color=["#2ca02c", "#1f77b4"], alpha=0.85, width=0.55)
for r, v in zip(b, lls):
    axes[1].text(r.get_x() + r.get_width() / 2, v + 0.002, "%.4f" % v, ha="center", fontsize=11)
axes[1].set_ylim(0.66, 0.70)
axes[1].set_title("Logloss (ниже = лучше)")
axes[1].grid(alpha=0.3, axis="y")
fig.suptitle("Polymarket — первый внешний рынок, который побил нашу модель: logloss −0.011, точность +1.2 пп",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "iter9_poly.png"), dpi=130)
plt.close(fig)

# 2. Кросс-спортивная усталость: сдвиг против базы
fig, ax = plt.subplots(figsize=(11, 5))
n = j["nhl_ot_load"]
o = j["nfl_fatigue"]
c = j["ncaab_fatigue"]
a = j["atp_fatigue"]
rows = [
    ("NHL: гость B2B\n(против rested)", (n["away_b2b_no_ot"]["home_win"] - n["away_rested_no_ot"]["home_win"]) * 100, int(n["away_b2b_no_ot"]["n"])),
    ("NHL: гость B2B после OT\n(против B2B без OT)", (n["away_b2b_ot_last"]["home_win"] - n["away_b2b_no_ot"]["home_win"]) * 100, int(n["away_b2b_ot_last"]["n"])),
    ("NFL: дома после бая\n(против базы)", (o["home_bye_away_norm"]["home_win"] - o["base_home_win"]) * 100, int(o["home_bye_away_norm"]["n"])),
    ("NCAAB: гость B2B\n(против базы)", (c["away_b2b_home_rested"]["home_win"] - c["both_rested"]["home_win"]) * 100, int(c["away_b2b_home_rested"]["n"])),
    ("ATP: игрок после плотного\nматча (факт − ожидание)", -a["fatigue_delta"] * 100, int(a["n_instances_with_fat_state"])),
]
xs = np.arange(len(rows))
vals = [r[1] for r in rows]
cols = ["#d62728" if v > 0 else "#7f7f7f" for v in vals]
bars = ax.bar(xs, vals, color=cols, alpha=0.85, width=0.6)
for x, (label, v, nn) in zip(xs, rows):
    ax.text(x, v + (0.25 if v >= 0 else -0.55), "%+.1f пп" % v, ha="center", fontsize=11)
    ax.text(x, -2.5, "n=%d" % nn, ha="center", fontsize=9, color="gray")
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(xs)
ax.set_xticklabels([r[0] for r in rows], fontsize=9)
ax.set_ylabel("Сдвиг победы «свежей» стороны, п.п.")
ax.set_title("Усталость по спортам: знак неожиданный у OT и ATP — проверяем гипотезы, не подгоняем")
ax.set_ylim(-13, 13)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "iter9_fatigue.png"), dpi=130)
plt.close(fig)
print("charts_iter9: saved iter9_poly.png, iter9_fatigue.png")
