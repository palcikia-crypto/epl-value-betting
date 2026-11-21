"""Итерация 2: Kelly vs flat, pre-close vs closing odds (CLV), tennis Elo-модель,
live-прогнозы EPL 2025/26. Отбор конфигураций — только на валидации."""
import json
import os
import sys
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.features import build_features, FEATURE_COLS
from src.models import fit_calibrated_ts
from src.backtest import baseline_favorite
from src.staking import run_backtest
from src.tennis import load_years, build_tennis_features, TENNIS_FEATURES
from xgboost import XGBClassifier

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DASH = os.path.join(ROOT, "dashboard")
EPL_SEASONS = ["2020/21", "2021/22", "2022/23", "2023/24", "2024/25"]
THRESH = 0.05


def make_xgb(prm):
    return XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        tree_method="hist", random_state=42, verbosity=0, n_jobs=-1, **prm)


def trim_vals(res):
    out = {}
    for k, v in res.items():
        if k in ("curve", "records"):
            continue
        try:
            out[k] = round(float(v), 6)
        except (TypeError, ValueError):
            continue
    return out


def load_all():
    from src.data import load_all as _la
    return _la("data")


def load_features():
    d = build_features(load_all())
    d.to_csv(os.path.join(OUT, "features_full.csv"), index=False)
    return d


def epl_part():
    print("=== EPL iter2 ===")
    df = load_features()
    trainval = df[df["Season"].isin(EPL_SEASONS[:4])].reset_index(drop=True)
    test = df[df["Season"] == "2024/25"].reset_index(drop=True)

    params = json.load(open(os.path.join(OUT, "results.json")))["xgb_params"]
    model = fit_calibrated_ts(lambda: make_xgb(params),
                              trainval[FEATURE_COLS].values, trainval["FTR"].values)
    prob = model.predict_proba(test[FEATURE_COLS].values)

    op = test[["PSH", "PSD", "PSA"]].astype(float).values
    cl = test[["PSCH", "PSCD", "PSCA"]].astype(float).values
    bets = []
    for i in range(len(test)):
        o = op[i]
        e = prob[i] * o - 1.0
        o_ = int(np.argmax(e))
        if e[o_] > THRESH and np.isfinite(o[o_]) and o[o_] > 1:
            clv = (o[o_] / cl[i, o_] - 1.0) if np.isfinite(cl[i, o_]) and cl[i, o_] > 1 else np.nan
            bets.append({
                "i": i, "outcome": ["H", "D", "A"][o_],
                "odds_preclose": float(o[o_]),
                "odds_closing": float(cl[i, o_]) if np.isfinite(cl[i, o_]) else np.nan,
                "prob": float(prob[i, o_]), "edge": float(e[o_]), "clv": clv,
                "won": bool(int(test.iloc[i]["FTR"]) == o_),
                "date": str(test.iloc[i]["Date"])[:10],
                "home": test.iloc[i]["HomeTeam"], "away": test.iloc[i]["AwayTeam"]})

    experiments = {}
    for th in (0.05, 0.10):
        sel = [b for b in bets if b["edge"] > th]
        for scheme, name, kw in (("flat", "flat", {}),
                                 ("kelly_q", "kelly_quarter", {"kelly_mult": 0.25}),
                                 ("kelly_h", "kelly_half", {"kelly_mult": 0.5}),
                                 ("kelly_full", "kelly_full", {"kelly_mult": 1.0})):
            res = run_backtest([(b["won"], b["odds_preclose"], b["prob"]) for b in sel], scheme, **kw)
            experiments[f"preclose_{name}_{int(th*100)}"] = trim_vals(res)
        res = run_backtest([(b["won"], b["odds_closing"], b["prob"]) for b in sel], "flat")
        experiments[f"closing_flat_{int(th*100)}"] = trim_vals(res)

    clv = [b["clv"] for b in bets if np.isfinite(b.get("clv", np.nan))]
    clv_stats = {"mean_clv": float(np.mean(clv)),
                 "positive_share": float(np.mean([c > 0 for c in clv])), "n": len(clv)} if clv else {}

    bl = baseline_favorite(test, cl)
    live = epl_live(model, df)
    return experiments, clv_stats, bl, live, bets


def epl_live(model, df_all):
    print("=== EPL live 2025/26 ===")
    p = os.path.join(ROOT, "data", "E0_2526.csv")
    if not os.path.exists(p):
        urllib.request.urlretrieve("https://www.football-data.co.uk/mmz4281/2526/E0.csv", p)
    raw = pd.read_csv(p, comment="#")
    raw.columns = [c.strip() for c in raw.columns]
    keep = raw[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR", "PSH", "PSD", "PSA"]].copy()
    keep["Date"] = pd.to_datetime(keep["Date"], dayfirst=True, errors="coerce")
    keep = keep.dropna(subset=["Date", "HomeTeam", "AwayTeam"])
    keep["FTHG"] = pd.to_numeric(keep["FTHG"], errors="coerce")
    keep["FTAG"] = pd.to_numeric(keep["FTAG"], errors="coerce")
    keep["FTR"] = keep["FTR"].map({"H": 0, "D": 1, "A": 2})
    keep["Season"] = "2025/26"
    full = pd.concat([df_all, keep], ignore_index=True).sort_values("Date").reset_index(drop=True)
    feats = build_features(full)
    for c in FEATURE_COLS:
        feats[c] = feats[c].replace([np.inf, -np.inf], 0.0).fillna(0.0)
    live_df = feats[feats["Season"] == "2025/26"].reset_index(drop=True)
    probs = model.predict_proba(live_df[FEATURE_COLS].values)
    odds = live_df[["PSH", "PSD", "PSA"]].astype(float).values
    rows = []
    for i in range(len(live_df)):
        o = odds[i]
        if not np.isfinite(o).all() or (o <= 1).any():
            continue
        e = probs[i] * o - 1.0
        o_ = int(np.argmax(e))
        ftr = live_df.iloc[i]["FTR"]
        rec = {
            "date": str(live_df.iloc[i]["Date"])[:10],
            "home": live_df.iloc[i]["HomeTeam"], "away": live_df.iloc[i]["AwayTeam"],
            "bet": ["H", "D", "A"][o_], "odds": float(o[o_]),
            "prob": float(probs[i, o_]), "edge": float(e[o_]),
            "played": bool(pd.notna(ftr)),
            "won": (int(ftr) == o_) if pd.notna(ftr) else None,
        }
        if rec["edge"] > THRESH:
            rows.append(rec)
    played = [r for r in rows if r["won"] is not None]
    pnl = sum(0.01 * (r["odds"] - 1) if r["won"] else -0.01 for r in played)
    return {"count": len(rows), "played": len(played),
            "wins": sum(1 for r in played if r["won"]),
            "roi_so_far": pnl if played else 0.0, "recommendations": rows}


def tennis_part():
    print("=== Tennis (ATP) ===")
    df = load_years(range(2021, 2026))
    print(f"Матчей: {len(df)}")
    feats = build_tennis_features(df)
    for c in TENNIS_FEATURES:
        feats[c] = pd.to_numeric(feats[c], errors="coerce").replace([np.inf, -np.inf], 0.0).fillna(0.0)

    tr = feats[feats["year"] <= 2023]
    val = feats[feats["year"] == 2024]
    tv = feats[feats["year"] <= 2024]
    te = feats[feats["year"] == 2025]

    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import accuracy_score, log_loss

    cands = {
        "LogReg": lambda: LogisticRegression(max_iter=2000),
        "HistGB": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                         random_state=42),
    }
    scores = {}
    for name, mk in cands.items():
        m = fit_calibrated_ts(mk, tr[TENNIS_FEATURES].values, tr["player1_wins"].values)
        p = m.predict_proba(val[TENNIS_FEATURES].values)[:, 1]
        scores[name] = log_loss(val["player1_wins"].values, p)
        print(f"Val {name} logloss: {scores[name]:.4f}")
    best = min(scores, key=scores.get)
    print(f"Выбрана модель (val logloss): {best}")

    final = fit_calibrated_ts(cands[best], tv[TENNIS_FEATURES].values, tv["player1_wins"].values)
    p2025 = final.predict_proba(te[TENNIS_FEATURES].values)[:, 1]
    acc25 = accuracy_score(te["player1_wins"].values, (p2025 > 0.5).astype(int))

    def bet_rows(part, probs):
        rows = []
        p1w = part["player1_wins"].values
        o1s = part["odds_p1"].values
        o2s = part["odds_p2"].values
        for i, pw in enumerate(probs):
            p2m = 1.0 - pw
            e1 = pw * o1s[i] - 1.0
            e2 = p2m * o2s[i] - 1.0
            o_ = 1 if e1 >= e2 else 2
            edge = e1 if o_ == 1 else e2
            odds = o1s[i] if o_ == 1 else o2s[i]
            prob = pw if o_ == 1 else p2m
            won = bool((int(p1w[i]) == 1) if o_ == 1 else (int(p1w[i]) == 0))
            if edge > 0.0 and np.isfinite(odds) and odds > 1:
                rows.append({"won": won, "odds": float(odds), "prob": float(prob), "edge": float(edge)})
        return rows

    bets_val = bet_rows(val, final.predict_proba(val[TENNIS_FEATURES].values)[:, 1])
    bets_2025 = bet_rows(te, p2025)

    out = {"model": best, "test_accuracy_2025": float(acc25),
           "n_matches": {"2021": len(feats[feats["year"] == 2021]),
                         "2024": len(val), "2025": len(te)}}
    res25_holder = {}
    for th in (0.02, 0.05):
        for scheme, name, kw in (("flat", "flat", {}), ("kelly_q", "kelly_quarter", {"kelly_mult": 0.25})):
            res24 = run_backtest([(b["won"], b["odds"], b["prob"]) for b in bets_val if b["edge"] > th], scheme, **kw)
            res25 = run_backtest([(b["won"], b["odds"], b["prob"]) for b in bets_2025 if b["edge"] > th], scheme, **kw)
            out[f"2024_{name}_{int(th*100)}"] = trim_vals(res24)
            out[f"2025_{name}_{int(th*100)}"] = trim_vals(res25)
            res25_holder[(scheme, th)] = res25
    last25 = res25_holder[("kelly_q", 0.05)]

    fb = []
    for i in range(len(te)):
        o1, o2 = te["odds_p1"].values[i], te["odds_p2"].values[i]
        fav = 1 if o1 <= o2 else 2
        odds = o1 if fav == 1 else o2
        if not np.isfinite(odds) or odds <= 1:
            continue
        won = bool((int(te["player1_wins"].values[i]) == 1) if fav == 1
                   else (int(te["player1_wins"].values[i]) == 0))
        fb.append((won, float(odds), 1.0 / float(odds)))
    out["2025_baseline_favorite"] = trim_vals(run_backtest(fb, "flat"))
    return out, bets_2025, last25


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(DASH, exist_ok=True)
    experiments, clv_stats, bl, live, bets = epl_part()
    tennis, tennis_bets_2025, last25 = tennis_part()

    iter2 = {
        "epl": {"test_season": "2024/25", "experiments": experiments, "clv": clv_stats,
                "baseline_closing": trim_vals(bl), "bets": bets},
        "tennis": {k: v for k, v in tennis.items()},
        "live_2526": live,
    }
    with open(os.path.join(OUT, "iter2.json"), "w", encoding="utf-8") as f:
        json.dump(iter2, f, ensure_ascii=False, indent=2, default=float)
    pd.DataFrame(tennis_bets_2025).to_csv(os.path.join(OUT, "tennis_bets_2025.csv"), index=False)
    pd.DataFrame(bets).to_csv(os.path.join(OUT, "iter2_epl_bets.csv"), index=False)
    pd.DataFrame(live["recommendations"]).to_csv(os.path.join(OUT, "live_2526.csv"), index=False)

    print("\n--- Эксперименты EPL (test 2024/25) ---")
    for k, v in experiments.items():
        print(f"{k}: ROI={v['roi']*100:.2f}% yield={v['yield_pct']*100:.2f}% "
              f"bets={v['n_bets']} DD={v['max_drawdown']*100:.1f}%")
    print("CLV:", clv_stats)
    print("\n--- Tennis ---")
    for k, v in tennis.items():
        print(k, v)
    print("\n--- Live 25/26 ---")
    print({k: v for k, v in live.items() if k != "recommendations"})

    from src.charts_iter2 import make_charts_iter2
    import shutil
    make_charts_iter2(OUT, experiments, tennis_bets_2025, last25)
    for fn in ("iter2.json", "kelly_curve.png", "tennis_curve.png",
               "results.json", "bank_curve.png", "tennis_bets_2025.csv",
               "iter2_epl_bets.csv", "live_2526.csv"):
        src = os.path.join(OUT, fn)
        if os.path.exists(src):
            shutil.copy(src, DASH)


if __name__ == "__main__":
    main()
