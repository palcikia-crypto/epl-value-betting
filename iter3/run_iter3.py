"""Итерация 3: мульти-лиговый бэктест, причинная аблация фич-блоков
(включая удары как xG-proxy из xgabora), WTA-тур."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_all
from src.features import build_features, FEATURE_COLS
from src.models import fit_calibrated_ts
from src.models import argmax_pick
from src.backtest import backtest, baseline_favorite
from src.staking import run_backtest, kelly_fraction
from src.tennis import load_years, build_tennis_features, TENNIS_FEATURES
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DASH = os.path.join(ROOT, "dashboard")
SEASONS = ["2020/21", "2021/22", "2022/23", "2023/24", "2024/25"]
LEAGUES = ["E0", "E1", "SP1", "D1", "I1", "F1"]

tree = ("elo_home", "elo_away", "elo_diff")
FORM = [c for c in FEATURE_COLS if c.startswith(("home_gf", "home_ga", "home_gd", "home_pts",
                                                 "away_gf", "away_ga", "away_gd", "away_pts"))]
SEASON_M = [c for c in FEATURE_COLS if "season" in c]
H2H = ["h2h_gd", "h2h_wr", "h2h_n"]


def make_xgb(prm):
    return XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        tree_method="hist", random_state=42, verbosity=0, n_jobs=-1, **prm)


def trim(res):
    out = {}
    for k, v in res.items():
        if k in ("curve", "records"):
            continue
        try:
            out[k] = round(float(v), 6)
        except (TypeError, ValueError):
            continue
    return out


# ---------- Часть 1: мульти-лиговый бэктест ----------

def multi_league(params):
    print("=== Мульти-лига ===")
    rows = []
    pooled_bets = []
    for lg in LEAGUES:
        try:
            df = load_all("data", league=lg)
        except Exception as e:
            print(lg, "load FAIL", e)
            continue
        if df["Season"].nunique() < 5 or df["Season"].max() != "2024/25":
            print(lg, "не хватает сезонов:", sorted(df["Season"].unique()))
            continue
        df = build_features(df)
        for c in FEATURE_COLS:
            df[c] = df[c].replace([np.inf, -np.inf], 0.0).fillna(0.0)
        tr = df[df["Season"].isin(SEASONS[:4])]
        te = df[df["Season"] == "2024/25"].reset_index(drop=True)
        model = fit_calibrated_ts(lambda: make_xgb(params), tr[FEATURE_COLS].values, tr["FTR"].values)
        prob = model.predict_proba(te[FEATURE_COLS].values)
        o = te[["PSH", "PSD", "PSA"]].astype(float).values
        bets = [b for b in argmax_pick(prob, o) if b[4] > 0.05]
        res = backtest(te, bets)
        bl = baseline_favorite(te, o)
        rows.append({"league": lg, "n_matches": len(te),
                     "bets": res["n_bets"], "roi": res["roi"], "yield": res["yield_pct"],
                     "wr": res["win_rate"], "dd": res["max_drawdown"],
                     "baseline_roi": bl["roi"], "beats_baseline": res["roi"] > bl["roi"]})
        for b in bets:
            pooled_bets.append((bool(int(te.iloc[b[0]]["FTR"]) == b[1]), b[2], b[3]))
        print(f"{lg}: bets={res['n_bets']} ROI={res['roi']*100:.2f}% "
              f"baseline={bl['roi']*100:.2f}%")
    pooled = run_backtest(pooled_bets, "flat")
    rows.append({"league": "POOLED", "n_matches": sum(r["n_matches"] for r in rows),
                 "bets": pooled["n_bets"], "roi": pooled["roi"], "yield": pooled["yield_pct"],
                 "wr": pooled["win_rate"], "dd": pooled["max_drawdown"],
                 "baseline_roi": np.nan, "beats_baseline": None})
    print(f"POOLED: bets={pooled['n_bets']} ROI={pooled['roi']*100:.2f}% "
          f"yield={pooled['yield_pct']*100:.2f}%")
    return rows, pooled


# ---------- Часть 2: причинная аблация (EPL, val 2023/24) ----------

SHOT_MAP = {"HomeShots": "hs", "AwayShots": "as", "HomeTarget": "hst", "AwayTarget": "ast"}


def shots_features_epl(df_fd):
    """Rolling 5-match удары/в цель до матча (xgabora как xG-proxy)."""
    cols_add = ["home_sot5", "away_sot5", "home_sh5", "away_sh5", "sh_diff", "sot_diff"]
    p = os.path.join(ROOT, "data", "club_matches.csv")
    if not os.path.exists(p):
        return df_fd.assign(**{c: 0.0 for c in cols_add})
    raw = pd.read_csv(p, usecols=["Division", "MatchDate", "HomeTeam", "AwayTeam",
                                  "HomeShots", "AwayShots", "HomeTarget", "AwayTarget"])
    raw = raw[raw["Division"] == "E0"]
    raw["MatchDate"] = pd.to_datetime(raw["MatchDate"])
    raw = raw.dropna(subset=["HomeShots"]).sort_values("MatchDate").reset_index(drop=True)

    hist = {}
    rows = []
    for _, m in raw.iterrows():
        dt = m["MatchDate"]
        for team, opp, sot_f, sh_f, sot_a, sh_a in (
            (m["HomeTeam"], m["AwayTeam"], m["HomeTarget"], m["HomeShots"],
             m["AwayTarget"], m["AwayShots"]),
            (m["AwayTeam"], m["HomeTeam"], m["AwayTarget"], m["AwayShots"],
             m["HomeTarget"], m["HomeShots"]),
        ):
            h = hist.get(team, [])
            last5 = h[-5:]
            rows.append({
                "MatchDate": dt, "HomeTeam": m["HomeTeam"], "AwayTeam": m["AwayTeam"],
                "is_home": team == m["HomeTeam"],
                "sot5": np.mean([x[0] for x in last5]) if last5 else np.nan,
                "sot5_conc": np.mean([x[1] for x in last5]) if last5 else np.nan,
                "sh5": np.mean([x[2] for x in last5]) if last5 else np.nan,
                "sh5_conc": np.mean([x[3] for x in last5]) if last5 else np.nan,
            })
            h.append((sot_f, sot_a, sh_f, sh_a))
            hist[team] = h

    agg = pd.DataFrame(rows)
    home_f = agg[agg["is_home"]].set_index(["MatchDate", "HomeTeam", "AwayTeam"])
    away_f = agg[~agg["is_home"]].set_index(["MatchDate", "HomeTeam", "AwayTeam"])
    df_fd = df_fd.copy()
    df_fd["Date_d"] = pd.to_datetime(df_fd["Date"]).dt.normalize()
    df_fd = df_fd.set_index(["Date_d", "HomeTeam", "AwayTeam"])
    df_fd = df_fd.join(home_f[["sot5", "sot5_conc", "sh5", "sh5_conc"]].rename(
        columns={"sot5": "home_sot5", "sot5_conc": "home_sot5c", "sh5": "home_sh5",
                 "sh5_conc": "home_sh5c"}), how="left")
    df_fd = df_fd.join(away_f[["sot5", "sot5_conc", "sh5", "sh5_conc"]].rename(
        columns={"sot5": "away_sot5", "sot5_conc": "away_sot5c", "sh5": "away_sh5",
                 "sh5_conc": "away_sh5c"}), how="left")
    df_fd = df_fd.fillna({c: 0.0 for c in cols_add})
    df_fd["sh_diff"] = df_fd["home_sh5"] - df_fd["away_sh5"]
    df_fd["sot_diff"] = df_fd["home_sot5"] - df_fd["away_sot5"]
    return df_fd.reset_index().rename(columns={"Date_d": "Date"})


def causality_ablation(df, params):
    print("=== Аблация фич-блоков (EPL val 2023/24) ===")
    from sklearn.metrics import log_loss
    df = shots_features_epl(df)
    SH = ["home_sot5", "away_sot5", "home_sh5", "away_sh5", "sh_diff", "sot_diff"]
    tr = df[df["Season"].isin(SEASONS[:3])]
    val = df[df["Season"] == "2023/24"].reset_index(drop=True)
    Xv = val
    market = val[["PSH", "PSD", "PSA"]].astype(float).values
    imp = 1.0 / market
    imp = imp / imp.sum(axis=1, keepdims=True)
    yv = val["FTR"].values
    blocks = {
        "market_only": None,
        "elo": tree, "form": FORM, "season_cum": SEASON_M, "h2h": H2H,
        "shots_xgproxy": SH,
        "elo+form": list(tree) + list(FORM),
        "all": FEATURE_COLS,
        "all+shots": FEATURE_COLS + SH,
    }
    res = {}
    res["market_only"] = {"logloss": log_loss(yv, imp, labels=[0, 1, 2])}
    for name, cols in blocks.items():
        if cols is None:
            continue
        cols = list(cols)
        model = fit_calibrated_ts(lambda: make_xgb(params),
                                  tr[cols].values, tr["FTR"].values)
        p = model.predict_proba(Xv[cols].values)
        res[name] = {"logloss": log_loss(yv, p, labels=[0, 1, 2])}
    for k, v in res.items():
        print(f"  {k}: logloss={v['logloss']:.4f}")
    return res


# ---------- Часть 3: WTA ----------

def wta_part():
    print("=== WTA ===")
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, log_loss
    try:
        df = load_years(range(2021, 2026), tour="wta")
    except Exception as e:
        print("WTA load FAIL", e)
        return {}
    feats = build_tennis_features(df)
    for c in TENNIS_FEATURES:
        feats[c] = pd.to_numeric(feats[c], errors="coerce").replace([np.inf, -np.inf], 0).fillna(0)
    trv = feats[feats["year"] <= 2023]
    val = feats[feats["year"] == 2024]
    tv = feats[feats["year"] <= 2024]
    te = feats[feats["year"] == 2025]
    cands = {"LogReg": lambda: LogisticRegression(max_iter=2000),
             "HistGB": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                              random_state=42)}
    scores = {}
    for name, mk in cands.items():
        m = fit_calibrated_ts(mk, trv[TENNIS_FEATURES].values, trv["player1_wins"].values)
        scores[name] = log_loss(val["player1_wins"].values,
                                m.predict_proba(val[TENNIS_FEATURES].values)[:, 1])
    best = min(scores, key=scores.get)
    final = fit_calibrated_ts(cands[best], tv[TENNIS_FEATURES].values, tv["player1_wins"].values)
    p25 = final.predict_proba(te[TENNIS_FEATURES].values)[:, 1]
    out = {"model": min(scores, key=scores.get), "val_logloss": scores,
           "test_accuracy_2025": accuracy_score(te["player1_wins"].values, (p25 > 0.5).astype(int))}
    bets25 = []
    o1 = te["odds_p1"].values
    o2 = te["odds_p2"].values
    e1 = p25 * o1 - 1
    e2 = (1 - p25) * o2 - 1
    p1w = te["player1_wins"].values
    for i in range(len(te)):
        side = 1 if e1[i] >= e2[i] else 2
        odds = o1[i] if side == 1 else o2[i]
        prob = p25[i] if side == 1 else 1 - p25[i]
        edge = e1[i] if side == 1 else e2[i]
        won = bool((int(p1w[i]) == 1) if side == 1 else (int(p1w[i]) == 0))
        if edge > 0.02 and np.isfinite(odds) and odds > 1:
            bets25.append((won, float(odds), float(prob)))
    out["2025_flat_2"] = trim(run_backtest(bets25, "flat"))
    out["n_matches"] = {"2024": len(val), "2025": len(te)}
    print(f"WTA модель={out['model']} acc25={out['test_accuracy_2025']:.3f} "
          f"bets={out['2025_flat_2']['n_bets']} yield={out['2025_flat_2']['yield_pct']*100:.2f}%")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    params = json.load(open(os.path.join(OUT, "results.json")))["xgb_params"]
    ml_rows, pooled = multi_league(params)
    df0 = build_features(load_all("data"))
    ablation = causality_ablation(df0, params)
    wta = wta_part()
    pool_csv()
    iter3 = {"multi_league": ml_rows, "pooled": trim(pooled),
             "ablation_val2023/24": ablation, "wta": wta}
    with open(os.path.join(OUT, "iter3.json"), "w", encoding="utf-8") as f:
        json.dump(iter3, f, ensure_ascii=False, indent=2, default=float)
    from src.charts_iter3 import make_charts_iter3
    with open(os.path.join(OUT, "iter2.json"), encoding="utf-8") as f:
        played = [r for r in json.load(f)["live_2526"]["recommendations"] if r["won"] is not None]
    ld = pd.DataFrame(played)
    ld["m"] = ld["date"].str[:7]
    ld["pnl"] = ld.apply(lambda r: 0.01 * (r["odds"] - 1) if r["won"] else -0.01, axis=1)
    make_charts_iter3(OUT, ml_rows, ablation, ld.groupby("m")["pnl"].sum().reset_index())
    import shutil
    for fn in ("iter3.json", "multi_league.png", "ablation.png", "live_pool.png",
               "iter2.json", "kelly_curve.png", "tennis_curve.png", "live_2526.csv"):
        src = os.path.join(OUT, fn)
        if os.path.exists(src):
            shutil.copy(src, DASH)
    print("saved iter3.json")


def pool_csv():
    import importlib, shutil
    try:
        it2 = importlib.import_module("iter2.run_iter2")
        live = it2.epl_live(*_live_model())
        pd.DataFrame(live["recommendations"]).to_csv(os.path.join(OUT, "live_2526.csv"), index=False)
        with open(os.path.join(OUT, "iter2.json"), encoding="utf-8") as f:
            prev = json.load(f)
        prev["live_2526"] = live
        with open(os.path.join(OUT, "iter2.json"), "w", encoding="utf-8") as f:
            json.dump(prev, f, ensure_ascii=False, indent=2, default=float)
        print(f"live pool: {live['played']} played, roi={live['roi_so_far']*100:.2f}%")
    except Exception as e:
        print("live refresh FAIL:", e)


def _live_model():
    from src.data import load_all as _la
    params = json.load(open(os.path.join(OUT, "results.json")))["xgb_params"]
    df = build_features(_la("data"))
    tr = df[df["Season"].isin(SEASONS[:5])]
    model = fit_calibrated_ts(lambda: make_xgb(params), tr[FEATURE_COLS].values, tr["FTR"].values)
    return model, df


if __name__ == "__main__":
    main()
