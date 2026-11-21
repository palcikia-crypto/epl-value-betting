"""Итерация 4: углубление и нестандартные подходы.

1) Полный форвард-тест live-пула EPL 2025/26 сезона (правила зафиксированы ДО сезона,
   итерация 2) + CLV против closing-линии Pinnacle/B365.
2) Нестандартный подход: Dixon-Coles — Пуассон-модель силы команд с затуханием времени.
3) Стэкинг: модель x рынок (LogReg поверх лог-отношений) против market-only.
4) Steam-фактор: движение opening -> closing линии Pinnacle.
5) Перекалибровка рынка (isotonic) как самостоятельная стратегия.
"""
import json
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_all
from src.features import build_features, FEATURE_COLS
from src.models import fit_calibrated_ts
from src.backtest import backtest, baseline_favorite

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DASH = os.path.join(ROOT, "dashboard")
SEASONS = ["2020/21", "2021/22", "2022/23", "2023/24", "2024/25"]
THRESH = 0.05


def make_xgb():
    prm = json.load(open(os.path.join(OUT, "results.json")))["xgb_params"]
    return XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss",
                         tree_method="hist", random_state=42, verbosity=0, n_jobs=-1, **prm)


def devig(o):
    inv = 1.0 / np.asarray(o, dtype=float)
    return inv / inv.sum(axis=1, keepdims=True)


def value_bets(prob, o, thresh=THRESH):
    o = np.asarray(o, dtype=float)
    e = prob * o - 1.0
    s = e.argmax(axis=1)
    return [(i, int(s[i]), float(o[i, s[i]]), float(prob[i, s[i]]), float(e[i, s[i]]))
            for i in range(len(o))
            if np.isfinite(e[i]).all() and (o[i] > 1).all() and e[i, s[i]] > thresh]


def res_metrics(r):
    return {"bets": int(r["n_bets"]), "roi": round(float(r["roi"]), 4),
            "yield": round(float(r["yield_pct"]), 4), "wr": round(float(r["win_rate"]), 4),
            "dd": round(float(r["max_drawdown"]), 4)}


# ---------- 1. Полный форвард-тест live-пула 2025/26 ----------

def _clean_features(df):
    df = build_features(df)
    for c in FEATURE_COLS:
        df[c] = df[c].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return df


def live_load_2526():
    p = os.path.join(ROOT, "data", "E0_2526.csv")
    if not os.path.exists(p):
        import urllib.request
        urllib.request.urlretrieve("https://www.football-data.co.uk/mmz4281/2526/E0.csv", p)
    raw = pd.read_csv(p, comment="#")
    raw.columns = [c.strip() for c in raw.columns]
    raw["Date"] = pd.to_datetime(raw["Date"], dayfirst=True, errors="coerce")
    cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "PSH", "PSD", "PSA", "B365H", "B365D", "B365A",
            "PSCH", "PSCD", "PSCA", "B365CH", "B365CD", "B365CA"]
    keep = raw[[c for c in cols if c in raw.columns]].copy()
    keep = keep.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    for c in ("FTHG", "FTAG"):
        keep[c] = pd.to_numeric(keep[c], errors="coerce")
    keep["FTR"] = keep["FTR"].map({"H": 0, "D": 1, "A": 2})
    keep["Season"] = "2025/26"
    return keep


def full_season_forward():
    print("=== Live-пул 2025/26: ПОЛНЫЙ сезон (форвард-тест) ===")
    dfb = build_features(load_all("data"))
    tr = dfb[dfb["Season"].isin(SEASONS)]
    model = fit_calibrated_ts(lambda: make_xgb(), tr[FEATURE_COLS].values, tr["FTR"].values)

    live = live_load_2526()
    cols = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
            "PSH", "PSD", "PSA", "B365H", "B365D", "B365A",
            "PSCH", "PSCD", "PSCA", "B365CH", "B365CD", "B365CA", "Season"]
    full = pd.concat([dfb, live[[c for c in cols if c in live.columns]]],
                     ignore_index=True).sort_values("Date").reset_index(drop=True)
    feats = _clean_features(full)
    lf = feats[feats["Season"] == "2025/26"].reset_index(drop=True)
    prob = model.predict_proba(lf[FEATURE_COLS].values)

    pin = lf[["PSH", "PSD", "PSA"]].astype(float).values
    b365 = lf[["B365H", "B365D", "B365A"]].astype(float).values
    pin_close = lf[["PSCH", "PSCD", "PSCA"]].astype(float).values
    b365_close = lf[["B365CH", "B365CD", "B365CA"]].astype(float).values
    k = lf["FTR"].values
    dates = pd.to_datetime(lf["Date"])

    bets = []
    for i in range(len(lf)):
        has_ps = np.isfinite(pin[i]).all() and (pin[i] > 1).all()
        o = pin[i] if has_ps else b365[i]
        cls = pin_close[i] if has_ps else b365_close[i]
        src = "PS" if has_ps else "B365"
        e = prob[i] * o - 1.0
        s = int(np.argmax(e))
        if np.isfinite(e).all() and (o > 1).all() and e[s] > THRESH:
            bets.append({"i": i, "bet": ["H", "D", "A"][s], "odds": float(o[s]),
                         "prob": float(prob[i, s]), "edge": float(e[s]), "src": src,
                         "close_odds": float(cls[s]) if np.isfinite(cls[s]) and cls[s] > 1 else np.nan,
                         "won": bool(int(k[i]) == s) if pd.notna(k[i]) else None,
                         "date": str(dates.iloc[i])[:10],
                         "home": lf.iloc[i]["HomeTeam"], "away": lf.iloc[i]["AwayTeam"]})
    bd = pd.DataFrame(bets)
    pl = bd[bd["won"].notna()].copy().reset_index(drop=True)
    ret = np.where(pl["won"].astype(bool), pl["odds"] - 1.0, -1.0)
    n = len(ret)
    t = float(ret.mean() / ret.std() * np.sqrt(n)) if n > 2 else float("nan")
    bank = 10000 * np.cumprod(1 + ret * 0.01)
    peak = np.maximum.accumulate(bank)

    cs = pl["close_odds"].astype(float).values
    clv = np.where(np.isfinite(cs) & (cs > 1), pl["odds"].values / cs - 1.0, np.nan)
    clv_v = pd.Series(clv).dropna()

    monthly = pd.Series(ret * 100).groupby(pl["date"].str[:7]).sum()
    roi_by_src = {s: float(np.where(pl[pl.src == s]["won"].astype(bool),
                                    pl[pl.src == s]["odds"] - 1, -1.0).mean())
                  for s in pl["src"].unique()}
    res = {
        "count": int(len(bd)), "played": int(n), "wins": int(pl["won"].sum()),
        "roi": float(bank[-1] / 10000 - 1), "yield_pct": float(ret.mean()),
        "t_stat": t, "max_drawdown": float(np.max(1 - bank / peak)),
        "mean_clv": float(clv_v.mean()) if len(clv_v) else None,
        "clv_positive_share": float((clv_v > 0).mean()) if len(clv_v) else None,
        "clv_n": int(len(clv_v)),
        "monthly": {kk: round(float(v), 1) for kk, v in monthly.items()},
        "src_counts": {kk: int(v) for kk, v in bd.groupby("src").size().items()},
        "roi_by_src": roi_by_src,
    }
    print(f"bets={len(bd)} played={n} wins={int(pl['won'].sum())} "
          f"ROI={res['roi']*100:.2f}% yield={res['yield_pct']*100:.2f}% t={t:.2f} "
          f"DD={res['max_drawdown']*100:.1f}% CLV={res['mean_clv']}% (n={res['clv_n']})")
    return res, bd, bank.tolist()


# ---------- 2. Dixon-Coles ----------

GRID = 11
TRI_H = np.tril(np.ones((GRID, GRID)), -1)
TRI_A = np.triu(np.ones((GRID, GRID)), 1)


def dc_base(lam, mu, rho):
    s = np.arange(GRID)
    ph = poisson.pmf(s[None, :], lam[:, None])
    pa = poisson.pmf(s[None, :], mu[:, None])
    base = ph[:, :, None] * pa[:, None, :]
    base[:, 0, 0] *= np.clip(1 - lam * mu * rho, 0, None)
    base[:, 0, 1] *= np.clip(1 + lam * rho, 0, None)
    base[:, 1, 0] *= np.clip(1 + mu * rho, 0, None)
    base[:, 1, 1] *= np.clip(1 - rho, 0, None)
    base = np.clip(base, 0, None)
    return base / base.sum(axis=(1, 2), keepdims=True)


def dc_probs(par, teams, df):
    tm = {t: i for i, t in enumerate(teams)}
    hi = df["HomeTeam"].map(tm).map(lambda x: x if pd.notna(x) else -1).fillna(-1).astype(int).values
    ai = df["AwayTeam"].map(tm).map(lambda x: x if pd.notna(x) else -1).fillna(-1).astype(int).values
    hi = np.asarray(hi, dtype=int)
    ai = np.asarray(ai, dtype=int)
    lam = np.exp(par["att"][hi] - par["dfn"][ai] + par["gamma"])
    mu = np.exp(par["att"][ai] - par["dfn"][hi])
    base = dc_base(np.clip(lam, 0.02, 12), np.clip(mu, 0.02, 12), par["rho"])
    diag = base.diagonal(axis1=1, axis2=2).sum(1)
    return (np.stack([(base * TRI_H).sum((1, 2)), diag, (base * TRI_A).sum((1, 2))], axis=1),
            (hi >= 0) & (ai >= 0))


def fit_dc(df, xi):
    df = df.dropna(subset=["FTHG", "FTAG", "FTR"]).reset_index(drop=True)
    teams = sorted(set(df["HomeTeam"]) | set(df["AwayTeam"]))
    tid = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h = df["HomeTeam"].map(tid).values
    a = df["AwayTeam"].map(tid).values
    k = np.where(df["FTHG"] > df["FTAG"], 0, np.where(df["FTHG"] == df["FTAG"], 1, 2))
    dates = pd.to_datetime(df["Date"])
    w = np.exp(-xi * (dates.max() - dates).dt.days.values / 365.0)

    def obj(p):
        att = np.concatenate([p[:n - 1], [0.0]])
        dfn = np.concatenate([p[n - 1:2 * n - 2], [0.0]])
        P = dc_probs({"att": att, "dfn": dfn, "gamma": p[-2], "rho": p[-1]}, teams, df)[0]
        P = np.clip(P, 1e-12, None)
        return -(w * np.log(P[np.arange(len(df)), k])).sum() + 1e-3 * np.sum(p[:-2] ** 2)

    p0 = np.concatenate([np.zeros(2 * n - 2), [0.15, 0.0]])
    r = minimize(obj, p0, method="L-BFGS-B",
                 bounds=[(-3, 3)] * (2 * n - 2) + [(-0.5, 1.5), (-0.2, 0.15)],
                 options={"maxiter": 300})
    return {"att": np.concatenate([r.x[:n - 1], [0.0]]),
            "dfn": np.concatenate([r.x[n - 1:2 * n - 2], [0.0]]),
            "gamma": r.x[-2], "rho": r.x[-1]}, teams


def dixon_coles():
    print("=== Dixon-Coles (xi тюнинг на val) ===")
    df = load_all("data")
    tr = df[df["Season"].isin(SEASONS[:3])]
    va = df[df["Season"] == "2023/24"].reset_index(drop=True)
    best = None
    for xi in [0.0, 0.0005, 0.0008, 0.0015, 0.003]:
        par, teams = fit_dc(tr, xi)
        P, mask = dc_probs(par, teams, va)
        kk = va["FTR"].values[mask]
        ll = float(log_loss(kk, P[mask], labels=[0, 1, 2]))
        print(f"xi={xi}: val logloss={ll:.5f}")
        if best is None or ll < best[0]:
            best = (ll, xi)
    xi = best[1]
    trv = df[df["Season"].isin(SEASONS[:4])]
    te = df[df["Season"] == "2024/25"].reset_index(drop=True)
    par, teams = fit_dc(trv, xi)
    P, mask = dc_probs(par, teams, te)
    kk = te["FTR"].values[mask]
    o = te[["PSH", "PSD", "PSA"]].astype(float).values
    k = te["FTR"].values
    res = backtest(te, value_bets(P, o))
    bl = baseline_favorite(te, o)
    out = {
        "best_xi": xi, "val_logloss": best[0],
        "test_logloss": float(log_loss(kk, P[mask], labels=[0, 1, 2])),
        "test_accuracy": float(accuracy_score(kk, P[mask].argmax(1))),
        "backtest": res_metrics(res), "baseline_roi": round(float(bl["roi"]), 4),
    }
    print(f"DC test logloss={out['test_logloss']:.5f} acc={out['test_accuracy']:.3f} "
          f"ROI={res['roi']*100:.2f}% bets={res['n_bets']} baseline={bl['roi']*100:.2f}%")
    return out


# ---------- 3. Стэкинг модель x рынок ----------

def stacking():
    print("=== Стэкинг модель x рынок ===")
    df = _clean_features(load_all("data"))
    tr = df[df["Season"].isin(SEASONS[:3])]
    va = df[df["Season"] == "2023/24"].reset_index(drop=True)
    tv = df[df["Season"].isin(SEASONS[:4])]
    te = df[df["Season"] == "2024/25"].reset_index(drop=True)

    mA = fit_calibrated_ts(lambda: make_xgb(), tr[FEATURE_COLS].values, tr["FTR"].values)
    pv = mA.predict_proba(va[FEATURE_COLS].values)
    iv = devig(va[["PSH", "PSD", "PSA"]].values)
    stack = LogisticRegression(C=1.0, max_iter=2000).fit(
        np.log(np.clip(pv, 1e-6, None) / np.clip(iv, 1e-6, None)), va["FTR"].values)
    pv_st = stack.predict_proba(np.log(np.clip(pv, 1e-6, None) / np.clip(iv, 1e-6, None)))

    mB = fit_calibrated_ts(lambda: make_xgb(), tv[FEATURE_COLS].values, tv["FTR"].values)
    pt = mB.predict_proba(te[FEATURE_COLS].values)
    o = te[["PSH", "PSD", "PSA"]].astype(float).values
    it = devig(o)
    pt_st = stack.predict_proba(np.log(np.clip(pt, 1e-6, None) / np.clip(it, 1e-6, None)))

    kv, kt = va["FTR"].values, te["FTR"].values
    res_st = backtest(te, value_bets(pt_st, o))
    out = {
        "val_logloss": {"market": float(log_loss(kv, iv, labels=[0, 1, 2])),
                        "model": float(log_loss(kv, pv, labels=[0, 1, 2])),
                        "stack": float(log_loss(kv, pv_st, labels=[0, 1, 2]))},
        "test_logloss": {"market": float(log_loss(kt, it, labels=[0, 1, 2])),
                         "stack": float(log_loss(kt, pt_st, labels=[0, 1, 2]))},
        "backtest": res_metrics(res_st),
    }
    print(f"val: market={out['val_logloss']['market']:.5f} model={out['val_logloss']['model']:.5f} "
          f"stack={out['val_logloss']['stack']:.5f}")
    print(f"test: market={out['test_logloss']['market']:.5f} stack={out['test_logloss']['stack']:.5f} "
          f"ROI={res_st['roi']*100:.2f}% bets={res_st['n_bets']}")
    return out


# ---------- 4. Steam (opening -> closing Pinnacle) ----------

def steam():
    print("=== Steam: движение линии Pinnacle ===")
    parts = []
    for s in ["2021", "2122", "2223", "2324", "2425"]:
        d = pd.read_csv(os.path.join(ROOT, "data", f"E0_{s}.csv"), comment="#")
        d.columns = [c.strip() for c in d.columns]
        parts.append(d)
    e0 = pd.concat(parts, ignore_index=True).reset_index(drop=True)
    po = devig(e0[["PSH", "PSD", "PSA"]].values)
    pc = devig(e0[["PSCH", "PSCD", "PSCA"]].values)
    st = (pc - po) / po
    y = np.where(e0["FTHG"] > e0["FTAG"], 0, np.where(e0["FTHG"] == e0["FTAG"], 1, 2))
    close = e0[["PSCH", "PSCD", "PSCA"]].astype(float).values
    out = {}
    for name, ths in (("chase", (0.02, 0.04)), ("fade", (0.02,))):
        for th in ths:
            rets = []
            for i in range(len(e0)):
                j = int(np.argmax(st[i])) if name == "chase" else int(np.argmin(st[i]))
                cond = st[i, j] > th if name == "chase" else -st[i, j] > th
                if cond:
                    oc = float(close[i, j])
                    rets.append((oc - 1.0) if y[i] == j else -1.0)
            rets = np.array(rets)
            tt = float(rets.mean() / rets.std() * np.sqrt(len(rets))) if len(rets) > 2 else None
            out[f"{name}_{th}"] = {"bets": int(len(rets)),
                                   "yield": round(float(rets.mean()), 4),
                                   "t_stat": round(tt, 3) if tt is not None else None}
            print(f"{name} th={th}: bets={len(rets)} yield={rets.mean()*100:.2f}% t={tt:.2f}")
    out["mean_abs_steam"] = float(np.abs(st).mean())
    return out


# ---------- 5. Перекалибровка рынка ----------

def market_recalibration():
    print("=== Перекалибровка рынка (isotonic) ===")
    df = _clean_features(load_all("data"))
    hist = df[df["Season"].isin(SEASONS[:3])].reset_index(drop=True)
    ih = devig(hist[["PSH", "PSD", "PSA"]].values)
    yh = hist["FTR"].values
    iso = [IsotonicRegression(y_min=0, y_max=1, out_of_bounds="clip").fit(ih[:, c], (yh == c).astype(int))
           for c in range(3)]

    def recal(X):
        R = np.stack([iso[c].predict(X[:, c]) for c in range(3)], axis=1)
        return R / R.sum(axis=1, keepdims=True)

    va = df[df["Season"] == "2023/24"].reset_index(drop=True)
    te = df[df["Season"] == "2024/25"].reset_index(drop=True)
    iv = devig(va[["PSH", "PSD", "PSA"]].values)
    it = devig(te[["PSH", "PSD", "PSA"]].values)
    o = te[["PSH", "PSD", "PSA"]].astype(float).values
    rv = recal(iv)
    rt = recal(it)
    res = backtest(te, value_bets(rt, o))
    out = {
        "val_logloss": {"market": float(log_loss(va["FTR"], iv, labels=[0, 1, 2])),
                        "recal": float(log_loss(va["FTR"], rv, labels=[0, 1, 2]))},
        "test_logloss": {"market": float(log_loss(te["FTR"], it, labels=[0, 1, 2])),
                         "recal": float(log_loss(te["FTR"], rt, labels=[0, 1, 2]))},
        "backtest": res_metrics(res),
    }
    print(f"val: market={out['val_logloss']['market']:.5f} recal={out['val_logloss']['recal']:.5f} | "
          f"test: market={out['test_logloss']['market']:.5f} recal={out['test_logloss']['recal']:.5f} | "
          f"ROI={res['roi']*100:.2f}% bets={res['n_bets']} yield={res['yield_pct']*100:.2f}%")
    return out


def main():
    fwd, bets, curve = full_season_forward()
    dc = dixon_coles()
    st = stacking()
    sm = steam()
    mr = market_recalibration()
    res = {"forward": fwd, "dixon_coles": dc, "stacking": st, "steam": sm, "recalibration": mr}
    with open(os.path.join(OUT, "iter4.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2, default=float)
    bets.to_csv(os.path.join(OUT, "live_2526_full.csv"), index=False)
    np.save(os.path.join(OUT, "forward_curve.npy"), np.array(curve))
    print("saved iter4.json, live_2526_full.csv, forward_curve.npy")


if __name__ == "__main__":
    main()
