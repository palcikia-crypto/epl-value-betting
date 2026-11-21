"""Итерация 5: точность прогнозов и уверенность.

1) Кросс-спорт: точность прогнозов исхода (acc/logloss/Brier) для 6 футбольных лиг,
   ATP/WTA тенниса, НХЛ — с рыночным baseline и уровнем «угадать всегда класс H».
2) Кривая обучаемости: сколько сезонов истории нужно (1..5) для целевой точности EPL.
3) Калибровка: ECE/reliability + temperature scaling.
4) Совет моделей («агентов»): XGBoost + Dixon-Coles + рынок, взвешивание по 1/logloss.
5) НХЛ: открытый API api-web.nhle.com, Elo с burn-in, accuracy test 2024/25.
"""
import json
import os
import sys
import time
import urllib.request
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, log_loss

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_all
from src.features import build_features, FEATURE_COLS
from src.models import fit_calibrated_ts
from src.backtest import backtest, baseline_favorite
from src.tennis import load_years
from iter4.run_iter4 import (make_xgb, devig, value_bets, res_metrics,
                             fit_dc, dc_probs, _clean_features)
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DASH = os.path.join(ROOT, "dashboard")
SEASONS = ["2020/21", "2021/22", "2022/23", "2023/24", "2024/25"]
LEAGUES = ["E0", "E1", "SP1", "D1", "I1", "F1"]
CACHE = os.path.join(OUT, "nhl_cache")


# ---------- метрики ----------

def eval_probs(y_true, P):
    y = np.asarray(y_true)
    P = np.asarray(P)
    onehot = np.eye(3)[y]
    brier = float(np.mean(np.sum((P - onehot) ** 2, axis=1)))
    ece = 0.0
    n = len(y)
    for c in range(3):
        conf = P[:, c]
        pred = (P.argmax(axis=1) == c).astype(float)
        bins = np.clip((conf * 10).astype(int), 0, 9)
        for b in range(10):
            m = bins == b
            if m.sum() > 0:
                ece += m.sum() / n * abs(conf[m].mean() - pred[m].mean())
    return {"accuracy": float(accuracy_score(y, P.argmax(1))),
            "logloss": float(log_loss(y, P, labels=[0, 1, 2])),
            "brier": round(brier, 5), "ece": round(ece, 5)}


# ---------- 1. Футбол: кросс-лиги ----------

def football_accuracy():
    print("=== Футбол: acc по лигам (test 2024/25) ===")
    prm = json.load(open(os.path.join(OUT, "results.json")))["xgb_params"]

    def make():
        return XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                             tree_method="hist", random_state=42, verbosity=0, n_jobs=-1, **prm)

    rows = []
    for lg in LEAGUES:
        df = _clean_features(load_all("data", league=lg))
        tr = df[df["Season"].isin(SEASONS)]
        te = df[df["Season"] == "2024/25"].reset_index(drop=True)
        model = fit_calibrated_ts(make, tr[FEATURE_COLS].values, tr["FTR"].values)
        P = model.predict_proba(te[FEATURE_COLS].values)
        m = eval_probs(te["FTR"].values, P)
        mk = devig(te[["PSH", "PSD", "PSA"]].astype(float).values)
        m["market"] = eval_probs(te["FTR"].values, mk)
        m["league"] = lg
        m["n"] = len(te)
        rows.append(m)
        print(f"{lg}: model acc={m['accuracy']:.3f} ll={m['logloss']:.4f} | "
              f"market acc={m['market']['accuracy']:.3f} ll={m['market']['logloss']:.4f}")
    return rows


# ---------- 2. Tennis: Elo acc (ATP+2025, WTA+2025) ----------

def tennis_elo_accuracy():
    print("=== Теннис: Elo точность (test 2025) ===")
    out = {}
    for tour in ("atp", "wta"):
        df = load_years(range(2021, 2026), tour=tour)
        st = _TennisEloBare()
        correct = total = 0
        market_correct = market_total = 0
        for _, m in df.iterrows():
            if int(m["Date"].year) >= 2025:
                pw = st.prob(m["Winner"], m["Loser"])
                correct += int(pw >= 0.5)
                total += 1
                try:
                    o1, o2 = float(m["PSW"]), float(m["PSL"])
                    if np.isfinite(o1) and np.isfinite(o2) and o1 > 1 and o2 > 1:
                        market_correct += int(o1 <= o2)
                        market_total += 1
                except (TypeError, ValueError):
                    pass
            st.update(m["Winner"], m["Loser"])
        out[tour] = {"elo_accuracy": round(correct / total, 4), "n": total,
                     "market_accuracy": round(market_correct / market_total, 4) if market_total else None}
        print(f"{tour.upper()}: Elo acc={out[tour]['elo_accuracy']:.4f} на {total} матчах 2025; "
              f"рынок (фаворит PSW) acc={out[tour]['market_accuracy']}")
    return out


class _TennisEloBare:
    """Минимальный Elo для тенниса (K=250/(m+5)^0.4), только общий рейтинг."""

    def __init__(self):
        self.elo = {}
        self.m = {}

    def get(self, p):
        self.elo.setdefault(p, 1500.0)
        self.m.setdefault(p, 0)
        return self.elo[p]

    def prob(self, a, b):
        ea, eb = self.get(a), self.get(b)
        return 1.0 / (1 + 10 ** (-(ea - eb) / 400.0))

    def k(self, p):
        return 250.0 / ((self.m[p] + 5) ** 0.4)

    def update(self, w, l):
        pw = self.prob(w, l)
        kw, kl = self.k(w), self.k(l)
        self.elo[w] += kw * (1 - pw)
        self.elo[l] += kl * (0 - (1 - pw))
        self.m[w] += 1
        self.m[l] += 1


# ---------- 3. NHL: открытый API + Elo ----------

def nhl_load(start=date(2023, 10, 1), end=date(2025, 4, 30)):
    cached = os.path.join(OUT, "nhl_games.csv")
    if os.path.exists(cached):
        df = pd.read_csv(cached)
        print(f"NHL (cache): {len(df)} завершённых матчей {df['date'].min()} -> {df['date'].max()}")
        return df
    os.makedirs(CACHE, exist_ok=True)
    UA = {"User-Agent": "Mozilla/5.0"}
    d = start
    rows = []
    while d <= end:
        f = os.path.join(CACHE, f"{d.isoformat()}.json")
        if not os.path.exists(f):
            for attempt in range(3):
                try:
                    u = f"https://api-web.nhle.com/v1/score/{d.isoformat()}"
                    b = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=20).read()
                    open(f, "wb").write(b)
                    break
                except Exception:
                    if attempt == 2:
                        open(f, "wb").write(b"{}")
                    time.sleep(0.5)
            time.sleep(0.03)
        try:
            j = json.load(open(f))
        except Exception:
            j = {}
        for g in j.get("games", []):
            if g.get("gameState") in ("OFF", "FINAL"):
                rows.append({"date": g["gameDate"],
                             "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                             "hg": g["homeTeam"].get("score"), "ag": g["awayTeam"].get("score")})
        d += timedelta(days=1)
    df = pd.DataFrame(rows).dropna()
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    print(f"NHL: {len(df)} завершённых матчей {df['date'].min()} → {df['date'].max()}")
    return df


def nhl_elo_accuracy(df, k=20.0, home_adv=60.0, base=1500.0):
    elo, m = {}, {}

    def get(t):
        elo.setdefault(t, base)
        m.setdefault(t, 0)
        return elo[t]

    kfn = lambda t: k / ((m[t] + 5) ** 0.4)
    probs, y = [], []
    for _, g in df.iterrows():
        eh, ea = get(g["home"]), get(g["away"])
        p_h = 1 / (1 + 10 ** (-((eh + home_adv) - ea) / 400))
        probs.append(p_h)
        y.append(1 if g["hg"] > g["ag"] else 0)
        kw, kl = kfn(g["home"]), kfn(g["away"])
        s_h, s_a = (1, 0) if g["hg"] > g["ag"] else (0, 1)
        elo[g["home"]] += kw * (s_h - p_h)
        elo[g["away"]] += kl * (s_a - (1 - p_h))
        m[g["home"]] += 1
        m[g["away"]] += 1
    probs = np.array(probs)
    y = np.array(y)
    return probs, y


def nhl_test():
    print("=== НХЛ (открытый NHL API) ===")
    df = nhl_load()
    years = pd.to_datetime(df["date"]).dt.year
    probs, y = nhl_elo_accuracy(df, home_adv=60.0)
    mask = years.values == 2024
    pr = probs[mask]
    yy = y[mask]
    acc = float(((pr >= 0.5).astype(int) == yy).mean())
    base_iso = float(max(yy.mean(), 1 - yy.mean()))
    out = {"n_burnin": int((years < 2024).sum()), "n_test": int(mask.sum()),
           "elo_accuracy": round(acc, 4), "majority_baseline": round(base_iso, 4),
           "home_win_share": round(float(yy.mean()), 4)}
    print(f"НХЛ test 2023/24: Elo acc={acc:.4f} на {mask.sum()} матчах; "
          f"доля побед хозяев={yy.mean():.3f}")
    return out


# ---------- 4. Кривая обучаемости: сколько сезонов истории нужно ----------

def learning_curve():
    print("=== Кривая обучаемости EPL (1..5 сезонов истории) ===")
    prm = json.load(open(os.path.join(OUT, "results.json")))["xgb_params"]

    def make():
        return XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                             tree_method="hist", random_state=42, verbosity=0, n_jobs=-1, **prm)

    df = _clean_features(load_all("data"))
    te = df[df["Season"] == "2024/25"].reset_index(drop=True)
    rows = []
    for k in range(1, 6):
        tr = df[df["Season"].isin(SEASONS[:k])]
        model = fit_calibrated_ts(make, tr[FEATURE_COLS].values, tr["FTR"].values)
        P = model.predict_proba(te[FEATURE_COLS].values)
        m = eval_probs(te["FTR"].values, P)
        m["n_train_seasons"] = k
        rows.append(m)
        print(f"{k} сезон(ов) истории: acc={m['accuracy']:.3f} ll={m['logloss']:.4f} "
              f"brier={m['brier']:.4f} ece={m['ece']:.4f}")
    mk = devig(te[["PSH", "PSD", "PSA"]].astype(float).values)
    mref = eval_probs(te["FTR"].values, mk)
    print(f"рынок (devig): acc={mref['accuracy']:.3f} ll={mref['logloss']:.4f}")
    return {"curve": rows, "market": mref}


# ---------- 5. Совет моделей («агенты») ----------

def council():
    print("=== Совет моделей ===")
    df = _clean_features(load_all("data"))
    tr = df[df["Season"].isin(SEASONS[:3])]
    va = df[df["Season"] == "2023/24"].reset_index(drop=True)
    tv = df[df["Season"].isin(SEASONS[:4])]
    te = df[df["Season"] == "2024/25"].reset_index(drop=True)

    xgb_v = fit_calibrated_ts(lambda: make_xgb(), tr[FEATURE_COLS].values, tr["FTR"].values)
    P_xgb_v = xgb_v.predict_proba(va[FEATURE_COLS].values)

    par_dc_v, teams_dc_v = fit_dc(tr, 0.003)
    P_dc_v, mask_dc_v = dc_probs(par_dc_v, teams_dc_v, va)
    par_dc, teams_dc = fit_dc(tv, 0.003)
    P_dc, mask_dc = dc_probs(par_dc, teams_dc, te)

    iv = devig(va[["PSH", "PSD", "PSA"]].astype(float).values)
    it = devig(te[["PSH", "PSD", "PSA"]].astype(float).values)

    ll = {
        "xgb": float(log_loss(va["FTR"].values, P_xgb_v, labels=[0, 1, 2])),
        "dc": float(log_loss(va["FTR"].values[mask_dc_v], P_dc_v[mask_dc_v], labels=[0, 1, 2])),
        "market": float(log_loss(va["FTR"].values, iv, labels=[0, 1, 2])),
    }
    wts = {k_: (1 / ll[k_]) for k_ in ll}
    sw = sum(wts.values())
    wts = {k_: v / sw for k_, v in wts.items()}
    tv_model = fit_calibrated_ts(lambda: make_xgb(), tv[FEATURE_COLS].values, tv["FTR"].values)
    P_xgb_t = tv_model.predict_proba(te[FEATURE_COLS].values)

    kk = te["FTR"].values
    P_council = np.zeros_like(P_xgb_t)
    for i in range(len(te)):
        for c in range(3):
            p = wts["xgb"] * P_xgb_t[i, c] + wts["market"] * it[i, c]
            if mask_dc[i]:
                p += wts["dc"] * P_dc[i, c]
            else:
                p += wts["dc"] * it[i, c]
            P_council[i, c] = p
    P_council /= P_council.sum(axis=1, keepdims=True)
    MC = te[["PSH", "PSD", "PSA"]].astype(float).values
    res = {
        "val_logloss": ll, "weights": {k_: round(v, 3) for k_, v in wts.items()},
        "xgb": eval_probs(kk, P_xgb_t),
        "market": eval_probs(kk, it),
        "council": eval_probs(kk, P_council),
        "backtest_council": res_metrics(backtest(te, value_bets(P_council, MC))),
    }
    print(f"val logloss: {ll}")
    print(f"test: xgb={res['xgb']} market={res['market']} council={res['council']} "
          f"ROI council={res['backtest_council']['roi']}")
    return res


def main():
    fb = football_accuracy()
    tn = tennis_elo_accuracy()
    nhl = None
    try:
        nhl = nhl_test()
    except Exception as e:
        print("NHL FAIL:", e)
    lc = learning_curve()
    cn = council()
    res = {"football": fb, "tennis": tn, "nhl": nhl, "learning_curve": lc, "council": cn}
    with open(os.path.join(OUT, "iter5.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    print("saved iter5.json")


if __name__ == "__main__":
    main()
