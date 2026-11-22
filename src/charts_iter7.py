"""Графики итерации 7: тюнинг Elo, B2B-стабильность по сезонам, кривая банка правила."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
sys.path.insert(0, ROOT)
from iter7.run_iter7 import add_rest, load_nhl_ext, load_odds

j = json.load(open(os.path.join(OUT, "iter7.json")))

# 1. Elo: базовый -> тюнингованный
e = j["elo"]
fig, ax = plt.subplots(figsize=(9, 5))
labels = ["Val 2023", "Test 2024 -> 2025-04", "Календарный 2024"]
vals = [e["val"]["accuracy"] * 100,
        e["test_2024_2025"]["accuracy"] * 100,
        e["test_cal2024"]["accuracy"] * 100]
ns = [1347, e["n_test"], e["n_test_cal2024"]]
cols = ["#aec7e8", "#1f77b4", "#ff7f0e"]
bars = ax.bar(labels, vals, color=cols, alpha=0.9, width=0.55)
for b, v, n in zip(bars, vals, ns):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.15, f"{v:.2f}%\n(n={n})",
            ha="center", va="bottom", fontsize=10)
ax.axhline(55.95, ls="--", lw=1.4, color="red",
           label="Плоский Elo (итер. 5): 55.95%")
ax.axhline(58.94, ls="--", lw=1.4, color="green",
           label="Рынок (matched 2022/23): 58.94%")
ax.set_ylim(53, 61)
ax.set_ylabel("Точность предсказания победителя, %")
ax.set_title(f"НХЛ Elo после тюнинга (k=30, home=40, MOV=0.3, rtm=0.2) — {e['n_test']} test-игр")
ax.legend(loc="lower right")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "iter7_elo.png"), dpi=130)
plt.close(fig)

# 2. Стабильность B2B-эффекта по 7 сезонам
stab = j["b2b"]["b2b_stability"]
fp = j["b2b"]["b2b_frequency_pooled"]
fig, ax = plt.subplots(figsize=(10, 5))
xs = np.arange(len(stab))
w = 0.38
b1 = ax.bar(xs - w / 2, [r["home_win_away_b2b"] * 100 for r in stab], w,
            color="#d62728", alpha=0.85, label="Гость B2B, хозяин свежий")
b2 = ax.bar(xs + w / 2, [r["home_win_away_rested"] * 100 for r in stab], w,
            color="#1f77b4", alpha=0.85, label="Оба свежие")
for b, r in zip(b1, stab):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.4,
            f"{b.get_height():.1f}\n(n={r['n']})", ha="center", fontsize=8.5)
ax.axhline(fp["away_rested"]["home_win"] * 100, ls=":", lw=1.3, color="gray",
           label=f"База 7 сезонов: {fp['away_rested']['home_win']*100:.2f}%")
ax.set_xticks(xs)
ax.set_xticklabels([f"сезон {r['season'][-2:]}/{str(int(r['season'])+1)[-2:]}" for r in stab])
ax.set_ylabel("Победы хозяина, %")
diff = fp["away_b2b"]["home_win"] - fp["away_rested"]["home_win"]
ax.set_title(f"B2B-эффект: пул 7 сезонов +{diff*100:.1f} п.п. "
             f"(n={fp['away_b2b']['n']} vs {fp['away_rested']['n']}, "
             f"z={fp['z_away_b2b_vs_rested']}, p={fp['p_away_b2b_vs_rested']})")
ax.set_ylim(42, 68)
ax.legend(loc="upper left")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "iter7_b2b.png"), dpi=130)
plt.close(fig)

# 3. Кривая банка: B2B-правило vs фаворит
df_all = add_rest(load_nhl_ext())
odds = load_odds()
o = odds.merge(df_all[["date", "home", "away", "hg", "ag"]],
               on=["date", "home", "away"], how="inner").sort_values("date")
o = add_rest(o)
y = (o["hg"] > o["ag"]).astype(int).values
ab2b = ((o["b2b_away"] == 1) & (o["b2b_home"] == 0)).values
hb2b = ((o["b2b_home"] == 1) & (o["b2b_away"] == 0)).values
rule = ([(int(i), float(o.iloc[i]["close_home"]), bool(y[i]))
         for i in np.where(ab2b)[0]] +
        [(int(i), float(o.iloc[i]["close_away"]), bool(1 - y[i]))
         for i in np.where(hb2b)[0]])
rule.sort(key=lambda b: b[0])
ph = 1 / o["close_home"] / (1 / o["close_home"] + 1 / o["close_away"])
pf = (ph >= 0.5).astype(int).values
fav = [(int(i), float(o.iloc[i]["close_home"] if pf[i] else o.iloc[i]["close_away"]),
        bool(y[i] == pf[i])) for i in range(len(o))]
fav.sort(key=lambda b: b[0])


def equity(bets, init=10000.0, frac=0.01):
    bank = init
    curve = [bank]
    for _, odds_i, won in bets:
        bank += bank * frac * (odds_i - 1.0 if won else -1.0)
        curve.append(bank)
    return np.array(curve)


pr = j["b2b"]["pooled_rule"]
fig, ax = plt.subplots(figsize=(10, 5))
cr = equity(rule)
cf = equity(fav)
ax.plot(np.arange(len(cr)), cr, lw=2.2, color="#d62728",
        label=f"B2B-правило: {pr['n']} ставок, "
              f"ROI {pr['roi']*100:+.1f}%, yield {pr['yield']*100:+.2f}%")
ax.plot(np.arange(len(cf)), cf, lw=1.6, color="#1f77b4", alpha=0.8,
        label=f"Всегда фаворит (matched): {len(fav)} игр")
ax.set_xlabel("Ставки по ходу 2 сезонов (SBR, закрытые линии)")
ax.set_ylabel("Банк, $ (старт 10,000, флэт 1%)")
ax.set_title(f"B2B-правило vs фаворит: pooled t={pr['yield_t']}, p={pr['p_value']} — "+
             "тренд положительный, но статистики пока мало")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "iter7_rule_equity.png"), dpi=130)
plt.close(fig)

print("charts iter7 saved",
      os.path.exists(os.path.join(OUT, "iter7_elo.png")),
      os.path.exists(os.path.join(OUT, "iter7_b2b.png")),
      os.path.exists(os.path.join(OUT, "iter7_rule_equity.png")))
