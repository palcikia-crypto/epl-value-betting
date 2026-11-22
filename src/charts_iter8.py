"""Графики итерации 8: 10-сезонный пул B2B, окна правила, честный Elo-трек."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
j = json.load(open(os.path.join(OUT, "iter8.json")))

# 1. Частотный B2B-эффект: 10 сезонов + свежая 2025/26 + live 2026/27
stab = j["season_stability_10y"]
ff = j["fresh_freq_2526"]
pp = j["paper_2627"]
fig, ax = plt.subplots(figsize=(11, 5))
xs = np.arange(len(stab))
w = 0.38
b1 = ax.bar(xs - w / 2, [r["home_win_away_b2b"] * 100 for r in stab], w,
            color="#d62728", alpha=0.85, label="Гость B2B (хозяин свежий)")
b2 = ax.bar(xs + w / 2, [r["home_win_rested"] * 100 for r in stab], w,
            color="#1f77b4", alpha=0.85, label="Оба свежие")
ax.axhline(j["frequency_pooled_10y"]["away_rested"]["home_win"] * 100, ls=":", lw=1.3, color="gray",
           label=f"База {j['frequency_pooled_10y']['games']} игр: "
                 f"{j['frequency_pooled_10y']['away_rested']['home_win']*100:.1f}%")
ax.axvline(len(stab) - 0.5, color="green", lw=1.2, ls="--")
fp = j["frequency_pooled_10y"]
ax.set_title(f"B2B на 10 сезонах: +{(fp['away_b2b']['home_win']-fp['away_rested']['home_win'])*100:.1f} п.п., "
             f"z={fp['z_away_b2b_vs_rested']}, p={fp['p_away_b2b_vs_rested']:.5f} — эффект устойчив")
ax.set_xticks(xs)
ax.set_xticklabels([s["season"] for s in stab], rotation=45, ha="right")
ax.set_ylabel("Победы хозяина, %")
ax.set_ylim(42, 72)
ax.legend(loc="upper left", fontsize=9)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "iter8_freq.png"), dpi=130)
plt.close(fig)

# 2. Окна правила с точными ценами + prior
we = j["rule_exact"]["windows"]
pr = j["rule_favside_prior"]["windows"]
fig, ax = plt.subplots(figsize=(11, 5))
names = [f"prior {t}" for t in pr] + [t for t in we] + ["POOLED exact"]
vals = [pr[t]["yield"] * 100 for t in pr] + [we[t]["rule"]["yield"] * 100 for t in we] + \
       [j["rule_exact"]["pooled"]["yield"] * 100]
ns = [pr[t]["n"] for t in pr] + [we[t]["rule"]["n"] for t in we] + [j["rule_exact"]["pooled"]["n"]]
cols = ["#7f7f7f"] * len(pr) + ["#1f77b4"] * len(we) + ["#ff7f0e"]
bars = ax.bar(range(len(vals)), vals, color=cols, alpha=0.85)
for b, v, n in zip(bars, vals, ns):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.6 if v >= 0 else -1.6),
            f"{v:+.1f}%\n(n={n})", ha="center", fontsize=8.5)
ax.axhline(0, color="black", lw=1)
ax.set_xticks(range(len(vals)))
ax.set_xticklabels([n.replace("prior ", "p") for n in names], rotation=45, ha="right", fontsize=9)
ax.set_ylabel("Yield правила B2B, %")
ax.set_title("B2B-правило по окнам\nСерые = 2012-2017 (цена фаворита), синие = 2018-2023 (обе стороны), пул exact +3.75%, t=1.31", fontsize=11)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "iter8_rule.png"), dpi=130)
plt.close(fig)

# 3. Честный Elo-трек: калибровка -> тест -> свежий сезон
ef = j["elo_fresh_2526"]
pa = j["paper_2627"]
fig, ax = plt.subplots(figsize=(9, 5))
pts = [("Val 2023\n(настройка)", 59.47, "#aec7e8"),
       ("Test 2024/25\n(n=2295)", 58.78, "#1f77b4"),
       ("Свежий 2025/26\n(n=1312)", ef["accuracy"] * 100, "#ff7f0e"),
       ("Live 2026/27 до даты\n(n=%d)" % pa["games"], pa["elo_accuracy_live"] * 100, "#d62728")]
for i, (name, v, c) in enumerate(pts):
    b = ax.bar(i, v, color=c, alpha=0.9, width=0.55)
    ax.text(i, v + 0.4, f"{v:.1f}%", ha="center", fontsize=11)
ax.axhline(50, ls=":", lw=1.2, color="gray")
ax.set_xticks(range(len(pts)))
ax.set_xticklabels([p[0] for p in pts], fontsize=9)
ax.set_ylim(48, 62)
ax.set_ylabel("Точность Elo, %")
ax.set_title("Эволюция точности НХЛ-Elo во времени: настройка -> тест -> свежие сезоны\n"
             "На 2025/26 все варианты параметров падают до ~52% — сезон сам по себе другой")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "iter8_elo.png"), dpi=130)
plt.close(fig)

print("charts iter8 saved")
