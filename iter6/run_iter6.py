"""
Итерация 6: live-данные НХЛ, стекинг против рынка, портфельные стратегии.

1) НХЛ углублённо: пре-матч модель (Elo + отдых + back-to-back + форма) на открытом API,
   календарный 2024 в тесте; абляция блоков фичей.
2) Причинность расписания: back-to-back / отдых — винрейт с 95% ДИ, z-тест долей.
3) LIVE play-by-play (api-web.nhle.com/v1/gamecenter/{id}/play-by-play): реконструкция
   состояний матча каждые ~4 минуты; кривая «точность предсказания исхода за X минут
   до конца»; абляция «только счёт vs счёт+броски».
4) Футбол: divergence-анализ — где модель расходится с рынком и права ли она там.
5) Портфельные стратегии ставок + Монте-Карло «ставить на много матчей без эджа».
"""
import json
import math
import os
import sys
import time
import urllib.request
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_all
from src.features import FEATURE_COLS
from src.models import fit_calibrated_ts
from src.backtest import backtest
from iter4.run_iter4 import make_xgb, devig, value_bets, res_metrics, _clean_features
from iter5.run_iter5 import nhl_load, nhl_elo_accuracy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
SEASONS = ["2020/21", "2021/22", "2022/23", "2023/24", "2024/25"]
PBP = os.path.join(OUT, "nhl_pbp")
UA = {"User-Agent": "Mozilla/5.0"}
CK = list(range(0, 66, 4))


# ---------- 1. НХЛ: пре-матч фичи (Elo, отдых, B2B, форма) ----------

def nhl_features(df):
    """Хронологический проход; признаки только из матчей, сыгранных ДО текущего."""
    elo, hist = {}, {}
    rows = []
    for _, g in df.iterrows():
        h, a = g["home"], g["away"]
        d = g["date"]
        eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)

        def agg(t):
            q = hist.setdefault(t, [])
            last10 = q[-10:]
            gd10 = float(np.mean([x["gd"] for x in last10])) if last10 else 0.0
            w10 = float(np.mean([1.0 if x["w"] else 0.0 for x in last10])) if last10 else 0.5
            n = len(q)
            if n:
                gds = float(np.mean([x["gd"] for x in q]))
                ws = float(np.mean([1.0 if x["w"] else 0.0 for x in q]))
                rest = min(max((d - q[-1]["date"]).days, 0), 7)
            else:
                gds, ws, rest = 0.0, 0.5, 7
            return gd10, w10, gds, ws, rest, n

        gd10h, w10h, gdsh, wsh, resth, nh = agg(h)
        gd10a, w10a, gdsa, wsa, resta, na = agg(a)
        s_h = 1.0 if g["hg"] > g["ag"] else 0.0
        rows.append({
            "date": d, "home": h, "away": a,
            "elo_diff": eh - ea,
            "gd10_diff": gd10h - gd10a, "w10_diff": w10h - w10a,
            "season_gd_diff": gdsh - gdsa, "season_w_diff": wsh - wsa,
            "rest_home": float(resth), "rest_away": float(resta),
            "b2b_home": float(resth <= 1), "b2b_away": float(resta <= 1),
            "n_home": float(min(nh, 82)), "n_away": float(min(na, 82)),
            "y": s_h,
        })
        ng = int(g["hg"]) - int(g["ag"])
        hist[h].append({"date": d, "gd": ng, "w": bool(s_h > 0.5)})
        hist[a].append({"date": d, "gd": -ng, "w": bool(1 - s_h > 0.5)})
        p_h = 1 / (1 + 10 ** (-((eh + 60.0) - ea) / 400))
        elo[h] = eh + 20.0 * (s_h - p_h)
        elo[a] = ea + 20.0 * ((1 - s_h) - (1 - p_h))
    return pd.DataFrame(rows)


NHL_FEAT = ["elo_diff", "gd10_diff", "w10_diff", "season_gd_diff", "season_w_diff",
            "rest_home", "rest_away", "b2b_home", "b2b_away", "n_home", "n_away"]
NHL_CORE = [c for c in NHL_FEAT if not c.startswith(("rest_", "b2b_"))]


def wilson(p, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 4), round(c + h, 4))


def two_prop_z(w1, n1, w2, n2):
    p1, p2 = w1 / max(n1, 1), w2 / max(n2, 1)
    p = (w1 + w2) / max(n1 + n2, 1)
    se = math.sqrt(p * (1 - p) * (1 / max(n1, 1) + 1 / max(n2, 1)))
    return (p1 - p2) / se if se > 0 else 0.0


def nhl_deep_model():
    print("=== НХЛ: пре-матч модель (тест: календарный 2024) ===")
    df = nhl_load()
    df["date"] = pd.to_datetime(df["date"])
    F = nhl_features(df)
    yrs = F["date"].dt.year
    tr, te = F[yrs < 2024], F[yrs == 2024]
    ytr, yte = tr["y"].values.astype(int), te["y"].values.astype(int)
    out = {"n_train": int(len(tr)), "n_test": int(len(te))}

    def run(cols):
        m = LogisticRegression(max_iter=3000, C=0.5)
        m.fit(tr[cols].values, ytr)
        P = m.predict_proba(te[cols].values)[:, 1]
        return {"accuracy": round(float(((P >= 0.5) == yte).mean()), 4),
                "logloss": round(float(log_loss(yte, np.clip(P, 1e-6, 1 - 1e-6))), 4),
                "brier": round(float(np.mean((P - yte) ** 2)), 4)}

    out["model_full"] = run(NHL_FEAT)
    out["model_core"] = run(NHL_CORE)
    ep, ey = nhl_elo_accuracy(df)
    m2024 = (pd.to_datetime(df["date"]).dt.year == 2024).values
    epp = (np.array(ep) >= 0.5).astype(int)
    out["elo_only"] = {"accuracy": round(float((epp[m2024] == np.array(ey)[m2024]).mean()), 4)}
    out["majority"] = round(float(max(yte.mean(), 1 - yte.mean())), 4)
    out["home_win_share"] = round(float(yte.mean()), 4)

    base = float(yte.mean())
    w_home = te["y"].values
    bb_h, bb_a = te["b2b_home"].values, te["b2b_away"].values
    buckets = {
        "home_b2b_away_rested": (bb_h == 1) & (bb_a == 0),
        "away_b2b_home_rested": (bb_a == 1) & (bb_h == 0),
        "both_b2b": (bb_h == 1) & (bb_a == 1),
        "both_rested": (bb_h == 0) & (bb_a == 0),
    }
    rows = []
    for name, m in buckets.items():
        n = int(m.sum())
        p = float(w_home[m].mean()) if n else 0.0
        rows.append({"bucket": name, "n": n, "home_win_rate": round(p, 4),
                     "ci95": list(wilson(p, n)),
                     "diff_vs_all": round(p - base, 4)})
    b1 = buckets["away_b2b_home_rested"]
    b2 = buckets["home_b2b_away_rested"]
    z = two_prop_z(int((1 - w_home[b1]).sum()), int(b1.sum()),
                   int(w_home[b2].sum()), int(b2.sum()))
    out["rest_causality"] = rows
    out["z_away_b2b_vs_home_b2b"] = round(float(z), 3)
    out["home_win_all"] = round(base, 4)
    for r in rows:
        print(f"  {r['bucket']}: n={r['n']} home win={r['home_win_rate']} CI{r['ci95']}")
    print(f"  model acc={out['model_full']['accuracy']} (core {out['model_core']['accuracy']}) | "
          f"Elo-only {out['elo_only']['accuracy']} | majority {out['majority']} | z={out['z_away_b2b_vs_home_b2b']}")
    return out


# ---------- 2. LIVE play-by-play: кривая накопления информации ----------

def _secs(t):
    m, s = t.split(":")
    return int(m) * 60 + int(s)


def pbp_fetch(sample_rows):
    os.makedirs(PBP, exist_ok=True)
    out = {}
    for date_s, gid in sample_rows:
        f = os.path.join(PBP, f"{gid}.json")
        if os.path.exists(f) and os.path.getsize(f) > 200:
            try:
                out[gid] = json.load(open(f))
                continue
            except Exception:
                pass
        for attempt in range(3):
            try:
                u = f"https://api-web.nhle.com/v1/gamecenter/{gid}/play-by-play"
                b = urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=25).read()
                j = json.loads(b)
                open(f, "wb").write(b)
                out[gid] = j
                break
            except Exception:
                if attempt == 2:
                    out[gid] = None
                time.sleep(0.6)
        time.sleep(0.04)
    return {k: v for k, v in out.items() if v}


def pbp_extract(j):
    """Состояния по чекпойнтам CK: (прошло_мин, разница счёта, разница бросков в створ)."""
    plays = j.get("plays", [])
    if not plays:
        return None
    states = []
    i = 0
    hs, as_, hsog, asog = 0, 0, 0, 0
    for p in plays:
        pdd = p.get("periodDescriptor") or {}
        num = pdd.get("number", 1)
        if pdd.get("periodType") == "SO":
            continue
        tr_ = p.get("timeRemaining", "20:00")
        try:
            rem = _secs(tr_)
        except Exception:
            continue
        if num <= 3:
            elapsed = (num - 1) * 20 + (1200 - rem) / 60.0
        else:
            elapsed = 60.0 + (300 - rem) / 60.0
        dd = p.get("details") or {}
        if p.get("typeDescKey") == "goal":
            hs, as_ = int(dd.get("homeScore", hs)), int(dd.get("awayScore", as_))
        elif p.get("typeDescKey") == "shot-on-goal":
            hsog, asog = int(dd.get("homeSOG", hsog)), int(dd.get("awaySOG", asog))
        while i < len(CK) and elapsed >= CK[i]:
            states.append((max(min(CK[i], 63.0), 0.0), hs - as_, hsog - asog))
            i += 1
        if i >= len(CK):
            break
    while i < len(CK):
        states.append((max(min(CK[i], 63.0), 0.0), hs - as_, hsog - asog))
        i += 1
    return states


def nhl_ingame():
    print("=== НХЛ LIVE: play-by-play, кривая предсказуемости по ходу матча ===")
    df = nhl_load()
    df["date"] = pd.to_datetime(df["date"])

    import glob
    id_map = {}
    for f in glob.glob(os.path.join(OUT, "nhl_cache", "*.json")):
        try:
            j = json.load(open(f))
        except Exception:
            continue
        for g in j.get("games", []):
            if g.get("gameState") in ("OFF", "FINAL"):
                id_map[(g["gameDate"], g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"])] = g["id"]

    ymap = {}
    for _, r in df.iterrows():
        ymap[(r["date"].strftime("%Y-%m-%d"), r["home"], r["away"])] = 1 if r["hg"] > r["ag"] else 0

    sample = {}
    for idx, (_, r) in enumerate(df.iterrows()):
        d = r["date"].strftime("%Y-%m-%d")
        k = (d, r["home"], r["away"])
        if k not in id_map:
            continue
        take = (idx % 2 == 0) if d <= "2023-12-31" else (idx % 6 == 0)
        if take:
            sample[(d, id_map[k])] = (r["home"], r["away"])
    sample_rows = sorted(sample.keys())
    print(f"Выборка PBP: {len(sample_rows)} игр")
    got = pbp_fetch(sample_rows)
    print(f"Скачано PBP: {len(got)} игр")

    X, T, M, G = [], [], [], []
    gcnt = 0
    for (d, gid), (h, a) in sample.items():
        j = got.get(gid)
        y = ymap.get((d, h, a))
        if not j or y is None:
            continue
        st = pbp_extract(j)
        for elapsed, sd, sgd in st:
            tl = max(1.0 - elapsed / 63.0, 0.001)
            X.append([sd, tl, sgd, sd * (1 - tl), sgd * (1 - tl)])
            T.append(y)
            M.append(elapsed)
            G.append(d)

    X, T, M = np.array(X), np.array(T), np.array(M)
    G = np.array(G)
    print(f"Состояний: {len(T)}")

    tr_m = G <= "2023-12-31"
    te_m = ~tr_m
    n_train_g = tr_m.sum() / len(CK)
    n_test_g = te_m.sum() / len(CK)
    print(f"Train-игр: {n_train_g:.0f} (состояний {tr_m.sum()}), Test-игр: {n_test_g:.0f} (состояний {te_m.sum()})")

    m1 = LogisticRegression(max_iter=3000, C=0.5)
    m1.fit(X[tr_m], T[tr_m])
    P = m1.predict_proba(X[te_m])[:, 1]
    Tt, Mt = T[te_m], M[te_m]

    curve = []
    for lo, hi in [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 60), (60, 64)]:
        s = (Mt >= lo) & (Mt < hi)
        if s.sum() == 0:
            continue
        curve.append({"band": f"{lo}-{hi}", "n": int(s.sum()),
                      "accuracy": round(float(((P[s] >= 0.5).astype(int) == Tt[s]).mean()), 4),
                      "home_lead_share": round(float((X[te_m][s][:, 0] > 0).mean()), 4)})

    Xi = X[:, [0, 1, 3]]
    m0 = LogisticRegression(max_iter=3000, C=0.5)
    m0.fit(Xi[tr_m], T[tr_m])
    P0 = m0.predict_proba(Xi[te_m])[:, 1]
    ovr = ((P >= 0.5).astype(int) == Tt)
    out = {"n_states": int(len(T)), "n_train": int(tr_m.sum()), "n_test": int(te_m.sum()),
           "curve": curve,
           "overall_accuracy": round(float(ovr.mean()), 4),
           "auc": round(float(roc_auc_score(Tt, P)), 4),
           "logloss": round(float(log_loss(Tt, np.clip(P, 1e-6, 1 - 1e-6))), 4),
           "accuracy_score_only": round(float(((P0 >= 0.5).astype(int) == Tt).mean()), 4),
           "auc_score_only": round(float(roc_auc_score(Tt, P0)), 4),
           "n_train_games": int(n_train_g),
           "n_test_games": int(n_test_g)}
    print(f"  Overall: acc={out['overall_accuracy']} (AUC {out['auc']}) | счёт-only acc={out['accuracy_score_only']}")
    for c in curve:
        print(f"  t={c['band']} мин прошло: acc={c['accuracy']} (n={c['n']})")
    return out


# ---------- 3. Футбол: divergence-анализ (модель vs рынок) ----------

def market_divergence():
    print("=== Футбол E0: расхождение модели с рынком (тест 2024/25) ===")
    df = _clean_features(load_all("data", league="E0"))
    tr = df[df["Season"].isin(SEASONS)]
    te = df[df["Season"] == "2024/25"].reset_index(drop=True)
    model = fit_calibrated_ts(make_xgb, tr[FEATURE_COLS].values, tr["FTR"].values)
    P = model.predict_proba(te[FEATURE_COLS].values)
    O = te[["PSH", "PSD", "PSA"]].astype(float).values
    k = devig(O)
    e = P * O - 1.0
    side = e.argmax(axis=1)
    edge = e[np.arange(len(e)), side]
    d = P[np.arange(len(P)), side] - k[np.arange(len(k)), side]
    yhit = (te["FTR"].values == side).astype(int)
    mk_claim = k[np.arange(len(k)), side]

    def buckets(mask):
        rows = []
        for lo, hi in [(-9.0, 0.0), (0.0, 0.02), (0.02, 0.05), (0.05, 0.09), (0.09, 9.0)]:
            m = mask & (d >= lo) & (d < hi)
            if m.sum() == 0:
                continue
            rows.append({"bucket": f"[{lo}, {hi})", "n": int(m.sum()),
                         "hit_rate": round(float(yhit[m].mean()), 4),
                         "market_claim": round(float(mk_claim[m].mean()), 4),
                         "edge_gap": round(float(yhit[m].mean() - mk_claim[m].mean()), 4)})
        return rows

    vm = edge > 0.05
    out = {"all": buckets(np.ones(len(d), bool)), "value_only": buckets(vm),
           "corr_d_hit": round(float(np.corrcoef(d, yhit)[0, 1]), 4)}
    for tag in ("all", "value_only"):
        for r in out[tag]:
            print(f"  {tag} d{r['bucket']}: n={r['n']} факт={r['hit_rate']} рынок={r['market_claim']} gap={r['edge_gap']:+.4f}")
    print(f"  corr(d, попадание)={out['corr_d_hit']}")
    return out


# ---------- 4. Портфельные стратегии ставок ----------

def portfolio():
    print("=== Футбол E0: портфельные стратегии (тест 2024/25) ===")
    df = _clean_features(load_all("data", league="E0"))
    tr = df[df["Season"].isin(SEASONS)]
    te = df[df["Season"] == "2024/25"].reset_index(drop=True)
    model = fit_calibrated_ts(make_xgb, tr[FEATURE_COLS].values, tr["FTR"].values)
    P = model.predict_proba(te[FEATURE_COLS].values)
    O = te[["PSH", "PSD", "PSA"]].astype(float).values
    bets = value_bets(P, O, 0.05)

    def run_bt(bs):
        r = backtest(te, bs)
        return res_metrics(r), [round(x, 1) for x in r["curve"]]

    m_all, c_all = run_bt(bets)
    res = {"n_value_bets": len(bets), "all_flat": m_all, "curve_all_flat": c_all}

    by_day = {}
    for b in bets:
        by_day.setdefault(te.iloc[b[0]]["Date"], []).append(b)
    top3 = [b for d, bs in sorted(by_day.items()) for b in sorted(bs, key=lambda x: -x[4])[:3]]
    res["top3_per_day"], res["curve_top3"] = run_bt(top3)

    top1 = [sorted(bs, key=lambda x: -x[4])[0] for d, bs in sorted(by_day.items())]
    res["top1_per_day"], res["curve_top1"] = run_bt(top1)

    k = devig(O)
    idx = np.array([b[0] for b in bets])
    side = np.array([b[1] for b in bets])
    d = P[idx, side] - k[idx, side]
    thr = float(np.percentile(d, 50))
    res["high_divergence_half"] = res_metrics(backtest(te, [b for b, dd in zip(bets, d) if dd >= thr]))
    res["low_divergence_half"] = res_metrics(backtest(te, [b for b, dd in zip(bets, d) if dd < thr]))

    rng = np.random.default_rng(7)
    odds_pool = np.array([b[2] for b in bets])
    sims, n = 4000, len(bets)
    finals = np.empty(sims)
    for s in range(sims):
        bank = 10000.0
        for o in rng.choice(odds_pool, size=n, replace=True):
            if rng.random() < 1.0 / o:
                bank += bank * 0.01 * (o - 1)
            else:
                bank -= bank * 0.01
            if bank < 1:
                bank = 1.0
        finals[s] = bank
    hist_c, hist_e = np.histogram(finals, bins=30)
    res["mc_hist"] = {"edges": [round(float(x), 0) for x in hist_e], "counts": [int(c) for c in hist_c]}
    res["mc_no_edge"] = {"n_bets": int(n), "sims": sims,
                         "median_final": round(float(np.median(finals)), 0),
                         "mean_final": round(float(finals.mean()), 0),
                         "p_positive": round(float((finals > 10000).mean()), 4),
                         "p5_final": round(float(np.percentile(finals, 5)), 0),
                         "p95_final": round(float(np.percentile(finals, 95)), 0)}
    for kk, vv in res.items():
        print(f"  {kk}: {vv}")
    return res


def main():
    out = {}
    out["nhl_deep"] = nhl_deep_model()
    out["nhl_ingame"] = nhl_ingame()
    out["divergence"] = market_divergence()
    out["portfolio"] = portfolio()
    with open(os.path.join(OUT, "iter6.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    print("saved iter6.json")


if __name__ == "__main__":
    main()
