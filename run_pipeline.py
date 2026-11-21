"""Orchestrator: железный пайплайн от данных до вердикта. Без подгонки под Test."""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_all
from src.features import build_features, FEATURE_COLS
from src.models import tune_xgb, fit_logreg, fit_calibrated_ts, argmax_pick
from xgboost import XGBClassifier


def make_xgb(prm):
    return XGBClassifier(
        objective="multi:softprob", num_class=3, eval_metric="mlogloss",
        tree_method="hist", random_state=42, verbosity=0, n_jobs=-1, **prm)
from src.backtest import backtest, baseline_favorite
from src.charts import make_charts

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
SEASONS = ["2020/21", "2021/22", "2022/23", "2023/24", "2024/25"]
THRESHOLDS = [0.05, 0.10]


def odds_matrix(df, prefix=("PS",)):
    o = df[[f"{prefix[0]}H", f"{prefix[0]}D", f"{prefix[0]}A"]].astype(float).values
    return o


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=== ШАГ 1: данные ===")
    df = load_all("data")
    print(f"Матчей: {len(df)}, сезонов: {df['Season'].nunique()} ({df['Season'].unique().tolist()})")

    print("=== ШАГ 2: фичи ( leak-free ) ===")
    df = build_features(df)
    for col in FEATURE_COLS:
        df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    train = df[df["Season"].isin(SEASONS[:3])].reset_index(drop=True)
    val = df[df["Season"] == "2023/24"].reset_index(drop=True)
    test = df[df["Season"] == "2024/25"].reset_index(drop=True)
    trainval = df[df["Season"].isin(SEASONS[:4])].reset_index(drop=True)
    print(f"Train={len(train)}, Val={len(val)}, Test={len(test)}")

    X_tr, y_tr = train[FEATURE_COLS].values, train["FTR"].values
    X_tv, y_tv = trainval[FEATURE_COLS].values, trainval["FTR"].values
    X_v, y_v = val[FEATURE_COLS].values, val["FTR"].values
    X_t, y_t = test[FEATURE_COLS].values, test["FTR"].values
    odds_v = odds_matrix(val)
    odds_t = odds_matrix(test)

    print("=== ШАГ 3-4: модели, GridSearch + TimeSeriesSplit, isotonic-калибровка ===")
    xgb_best, params, best_score = tune_xgb(X_tr, y_tr)
    print(f"XGBoost best params: {params} (train logloss={-best_score:.4f})")
    logreg = fit_logreg(X_tr, y_tr)
    xgb_cal = fit_calibrated_ts(lambda: xgb_best, X_tr, y_tr)
    logreg_cal = fit_calibrated_ts(lambda: logreg, X_tr, y_tr)

    # отбор модели СТРОГО на validation
    from sklearn.metrics import accuracy_score, log_loss
    val_scores = {}
    preds_v = {}
    for name, m in [("LogReg", logreg_cal), ("XGBoost", xgb_cal)]:
        p = m.predict_proba(X_v)
        preds_v[name] = p
        val_scores[name] = {
            "acc": accuracy_score(y_v, np.argmax(p, axis=1)),
            "logloss": log_loss(y_v, p, labels=[0, 1, 2]),
        }
        print(f"Val {name}: acc={val_scores[name]['acc']:.3f}, logloss={val_scores[name]['logloss']:.4f}")
    best_model_name = min(val_scores, key=lambda k: val_scores[k]["logloss"])
    print(f"Выбрана модель (по validation logloss): {best_model_name}")

    # без утечки: GridSearch на train (TimeSeriesSplit), калибровка кросс-валидирована
    # внутри train. Финальная модель для Test — то же на train+val.
    if best_model_name == "XGBoost":
        model = fit_calibrated_ts(lambda: make_xgb(params), X_tv, y_tv)
    else:
        model = fit_calibrated_ts(lambda: fit_logreg(X_tv, y_tv), X_tv, y_tv)
    prob_v = model.predict_proba(X_v)
    # порог и структура стратегии выбираются СТРОГО на validation
    arg_v = argmax_pick(prob_v, odds_v)
    cand = {th: backtest(val, [b for b in arg_v if b[4] > th]) for th in THRESHOLDS}
    valid_ths = [th for th in THRESHOLDS if cand[th]["n_bets"] >= 50]
    chosen_th = max(valid_ths, key=lambda th: cand[th]["yield_pct"]) if valid_ths else 0.05
    for th in THRESHOLDS:
        print(f"Val edge>{th}: yield={cand[th]['yield_pct']*100:.2f}% bets={cand[th]['n_bets']}")
    print(f"Выбранный порог edge (по validation): {chosen_th}")

    print("=== ШАГ 5: TEST (финальный, не трогается при подгонке) ===")
    prob_t = model.predict_proba(X_t)
    test_bets = [b for b in argmax_pick(prob_t, odds_t) if b[4] > chosen_th]
    res_test = backtest(test, test_bets)
    res_baseline = baseline_favorite(test, odds_t)
    acc_test = accuracy_score(y_t, np.argmax(prob_t, axis=1))
    # сравнительные прогоны (не для выбора — просто отчёт)
    for th in THRESHOLDS:
        r = backtest(test, [b for b in argmax_pick(prob_t, odds_t) if b[4] > th])
        print(f"Test edge>{th}: yield={r['yield_pct']*100:.2f}% ROI={r['roi']*100:.2f}% "
              f"bets={r['n_bets']} wr={r['win_rate']*100:.1f}%")

    print("=== ШАГ 6: Walk-Forward (окно 2 сезона -> следующий, сдвиг 1 сезон) ===")
    wf_rows = []
    for k in range(1, 4):
        tr_seas = SEASONS[k - 1:k + 1]
        te_seas = SEASONS[k + 1]
        tr = df[df["Season"].isin(tr_seas)].reset_index(drop=True)
        te = df[df["Season"] == te_seas].reset_index(drop=True)
        if best_model_name == "XGBoost":
            wcal = fit_calibrated_ts(lambda: make_xgb(params),
                                     tr[FEATURE_COLS].values, tr["FTR"].values)
        else:
            wcal = fit_calibrated_ts(lambda: fit_logreg(tr[FEATURE_COLS].values, tr["FTR"].values),
                                     tr[FEATURE_COLS].values, tr["FTR"].values)
        pk = wcal.predict_proba(te[FEATURE_COLS].values)
        o_w = odds_matrix(te)
        bets_w = [b for b in argmax_pick(pk, o_w) if b[4] > chosen_th]
        r_w = backtest(te, bets_w)
        wf_rows.append({"train": f"{tr_seas[0]}+{tr_seas[1]}", "test": te_seas,
                        "roi": r_w["roi"], "yield": r_w["yield_pct"],
                        "bets": r_w["n_bets"], "win_rate": r_w["win_rate"]})
        print(f"WF {tr_seas[0]}+{tr_seas[1]} -> {te_seas}: ROI={r_w['roi']*100:.2f}% "
              f"yield={r_w['yield_pct']*100:.2f}% bets={r_w['n_bets']}")

    # reliability diagram данные
    rel = _reliability_bins(prob_v, y_v)
    rel_t = _reliability_bins(prob_t, y_t)

    red_flags = []
    if res_test["n_bets"] < 50:
        red_flags.append(f"Всего {res_test['n_bets']} ставок (< 50) — статистическая незначимость")
    if res_test["roi"] > 0.20 and res_test["n_bets"] < 100:
        red_flags.append("ROI > 20% при < 100 ставках — подозрение на переобучение/утечку")
    if acc_test > 0.70:
        red_flags.append(f"Test accuracy {acc_test*100:.1f}% > 70% — почти наверняка утечка")
    if res_test["roi"] <= res_baseline["roi"]:
        red_flags.append("Модель НЕ бьёт baseline 'всегда на фаворита'")

    results = {
        "best_model": best_model_name,
        "xgb_params": {k: (int(v) if isinstance(v, (int, np.integer)) else v) for k, v in params.items()},
        "val_scores": val_scores,
        "chosen_threshold": chosen_th,
        "test": {k: res_test[k] for k in ("n_bets", "roi", "yield_pct", "win_rate",
                                          "max_drawdown", "final_bank")},
        "test_by_threshold": {
            f"{int(th*100)}%": {k: backtest(test, [b for b in argmax_pick(prob_t, odds_t) if b[4] > th])[k]
                                for k in ("n_bets", "roi", "yield_pct", "win_rate", "max_drawdown")}
            for th in THRESHOLDS},
        "baseline": {k: res_baseline[k] for k in ("n_bets", "roi", "yield_pct", "win_rate",
                                                  "max_drawdown", "final_bank")},
        "test_accuracy": float(acc_test),
        "walk_forward": wf_rows,
        "red_flags": red_flags,
    }
    res_test_bets_df = pd.DataFrame(res_test["records"])
    res_test_bets_df.to_csv(os.path.join(OUT, "test_bets.csv"), index=False)
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=float)
    df.to_parquet if False else df.to_csv(os.path.join(OUT, "features_full.csv"), index=False)

    make_charts(OUT, res_test, res_baseline, wf_rows, rel, rel_t, best_model_name)

    import shutil
    dash = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dashboard")
    os.makedirs(dash, exist_ok=True)
    shutil.copy(os.path.join(OUT, "results.json"), dash)
    for png in ("bank_curve.png", "reliability.png", "reliability_val.png", "walkforward.png"):
        p = os.path.join(OUT, png)
        if os.path.exists(p):
            shutil.copy(p, dash)
    for csv in ("test_bets.csv", "features_full.csv"):
        p = os.path.join(OUT, csv)
        if os.path.exists(p):
            shutil.copy(p, dash)

    print("\n=== ВЕРДИКТ ===")
    print(f"Модель: {best_model_name}")
    print(f"Test season: 2024/25")
    print(f"ROI: {res_test['roi']*100:.2f}%")
    print(f"Yield: {res_test['yield_pct']*100:.2f}%")
    print(f"Количество ставок: {res_test['n_bets']}")
    print(f"Max Drawdown: {res_test['max_drawdown']*100:.2f}%")
    print(f"Baseline ROI (фаворит): {res_baseline['roi']*100:.2f}%")
    print(f"Test accuracy: {acc_test*100:.1f}%")
    print("\n=== RED FLAGS ===")
    for rf in red_flags:
        print("- " + rf)
    if not red_flags:
        print("- нет")
    print("\n=== ЗАКЛЮЧЕНИЕ ===")
    if res_test["roi"] < 0 or res_test["n_bets"] < 50:
        print("Модель не демонстрирует устойчивого преимущества. "
              "Не рекомендуется для реальных ставок.")
    elif any("baseline" in f for f in red_flags):
        print("Модель не бьёт baseline. Вердикт отрицательный.")
    else:
        print("Модель показывает преимущество на исторических данных. "
              "Следующий шаг — paper trading 3 месяца.")

    return results


def _reliability_bins(probs, y_true, n_bins=10):
    conf = probs.max(axis=1)
    pred = np.argmax(probs, axis=1)
    correct = (pred == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    out = []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        m = (conf > b0) & (conf <= b1)
        if m.sum() > 0:
            out.append({"bin": float((b0 + b1) / 2),
                        "mean_conf": float(conf[m].mean()),
                        "accuracy": float(correct[m].mean()),
                        "count": int(m.sum())})
    return out


if __name__ == "__main__":
    main()
