# -*- coding: utf-8 -*-
"""
Итерация 8: расширение пула до t-теста + свежий out-of-sample + paper-трекер.

1) SBR-архивы 2012/13 - 2022/23 (Wayback Machine, 10 сезонов, ~18k игр с линиями):
   - частотный B2B-эффект по всем сезонам (без коэффициентов, чистая статистика);
   - ROI правила с ТОЧНЫМИ ценами на 2018/19 - 2022/23 (5 окон, обе стороны);
   - на 2012/13 - 2016/17 у SBR только цена фаворита -> там правило проверяем
     только когда ставим на фавориистскую сторону (недвусмысленный prior-блок);
   - пул всех окон + t-тест.
2) Свежий out-of-sample Elo: сезон 2025/26 (завершён) - параметры никогда его не видели.
3) Paper-трекер live-пула NHL 2026/27: Elo-прогнозы + B2B-сигналы по текущему сезону.
"""
import json
import math
import os
import sys
import time
import urllib.request
from datetime import date, timedelta

import numpy as np
import pandas as pd
import openpyxl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "data", "nhl_odds")

from iter7.run_iter7 import am2dec, bin_metrics, elo_run, roi_sim, two_prop_z, wilson

SEASONS = [
    ("2012-13", "2012"), ("2013-14", "2013"), ("2014-15", "2014"),
    ("2015-16", "2015"), ("2016-17", "2016"), ("2018-19", "2018"),
    ("2019-20", "2019"), ("2020-21", "2020"), ("2021-22", "2021"),
    ("2022-23", "2022"),
]
EXACT_SEASONS = ["2018-19", "2019-20", "2020-21", "2021-22", "2022-23"]


def is_ml(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and abs(float(v)) >= 95


def sbr_parse_v2(tag, start_year):
    rows = list(openpyxl.load_workbook(os.path.join(DATA, f"nhl_odds_{tag}.xlsx"))
                .active.iter_rows(values_only=True))[1:]
    games = []
    i = 0
    while i < len(rows) - 1:
        a, b = rows[i], rows[i + 1]
        if not (a[2] in ("V", "N") and b[2] == "H" and abs(int(a[1]) - int(b[1])) == 1):
            i += 1
            continue
        dd = str(a[0]).zfill(4)
        mo, day = int(dd[:2]), int(dd[2:])
        year = int(start_year) if mo >= 7 else int(start_year) + 1
        dstr = f"{year}-{mo:02d}-{day:02d}"
        try:
            hg, ag = int(b[7]), int(a[7])
        except (ValueError, TypeError):
            i += 2
            continue
        g = {"season": tag, "date": dstr,
             "home": str(b[3]).strip(), "away": str(a[3]).strip(),
             "hg": hg, "ag": ag}
        for side, row in (("away", a), ("home", b)):
            if is_ml(row[8]) and is_ml(row[9]):
                g[f"open_{side}"] = am2dec(row[8])
                g[f"close_{side}"] = am2dec(row[9])
        if "close_home" in g or "close_away" in g:
            g["ml_exact_both"] = ("close_home" in g) and ("close_away" in g)
            games.append(g)
        i += 2
    return pd.DataFrame(games)


def load_sbr_all():
    frames = [sbr_parse_v2(t, y) for t, y in SEASONS]
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)
    return df


def add_b2b(df):
    df = df.sort_values("date").reset_index(drop=True)
    last = {}
    flags = []
    for _, g in df.iterrows():
        d = pd.Timestamp(g["date"])
        row = {}
        for side, t in (("home", g["home"]), ("away", g["away"])):
            r = int((d - last.get(t, d - timedelta(days=9))).days)
            row[f"rest_{side}"] = min(r, 7)
            row[f"b2b_{side}"] = int(r <= 1)
        flags.append(row)
        last[g["home"]] = d
        last[g["away"]] = d
    return pd.concat([df, pd.DataFrame(flags)], axis=1)


def build_rule_bets(o, exact_only=True):
    bets = []
    for i, g in o.iterrows():
        y_home = bool(g["hg"] > g["ag"])
        if g["b2b_away"] == 1 and g["b2b_home"] == 0:
            c = g.get("close_home")
            if exact_only and not g["ml_exact_both"]:
                continue
            if c is not None and not pd.isna(c):
                bets.append((int(i), float(c), y_home, "home"))
        elif g["b2b_home"] == 1 and g["b2b_away"] == 0:
            c = g.get("close_away")
            if exact_only and not g["ml_exact_both"]:
                continue
            if c is not None and not pd.isna(c):
                bets.append((int(i), float(c), not y_home, "away"))
    bets.sort(key=lambda x: x[0])
    return bets


def payout_stats(bets, init=10000.0, frac=0.01):
    return roi_sim([(o, w) for _, o, w, _ in bets], init=init, frac=frac)


def t_stat(bets):
    pay = [(o - 1.0 if w else -1.0) for _, o, w, _ in bets]
    mu, sd = float(np.mean(pay)), float(np.std(pay, ddof=1))
    from scipy import stats as st
    t = mu / (sd / math.sqrt(len(pay))) if sd > 0 and len(pay) > 1 else 0.0
    return t, float(2 * (1 - st.norm.cdf(abs(t))))


def frequency_block(o):
    y = (o["hg"] > o["ag"]).astype(int).values
    ab2b = ((o["b2b_away"] == 1) & (o["b2b_home"] == 0)).values
    hb2b = ((o["b2b_home"] == 1) & (o["b2b_away"] == 0)).values
    rest = ((o["b2b_away"] == 0) & (o["b2b_home"] == 0)).values
    z1 = two_prop_z(int(y[ab2b].sum()), int(ab2b.sum()), int(y[rest].sum()), int(rest.sum()))
    z2 = two_prop_z(int(y[rest].sum()), int(rest.sum()), int(y[hb2b].sum()), int(hb2b.sum()))
    from scipy import stats as st
    out = {"games": int(len(o)),
           "away_b2b": {"n": int(ab2b.sum()), "home_win": round(float(y[ab2b].mean()), 4),
                        "ci95": list(wilson(float(y[ab2b].mean()), int(ab2b.sum())))},
           "away_rested": {"n": int(rest.sum()), "home_win": round(float(y[rest].mean()), 4)},
           "home_b2b": {"n": int(hb2b.sum()), "home_win": round(float(y[hb2b].mean()), 4),
                        "ci95": list(wilson(float(y[hb2b].mean()), int(hb2b.sum())))},
           "z_away_b2b_vs_rested": round(z1, 3),
           "p_away_b2b_vs_rested": round(float(2 * (1 - st.norm.cdf(abs(z1)))), 6),
           "z_rested_vs_home_b2b": round(z2, 3)}
    return out


def rule_block(o, season_filter=None):
    oo = o if season_filter is None else o[o["season"].isin(season_filter)]
    bets = build_rule_bets(oo)
    t, p = t_stat(bets)
    return {**payout_stats(bets), "yield_t": round(t, 3), "p_value": round(p, 4)}


def fav_side_block(o):
    bets = build_rule_bets(o, exact_only=False)
    t, p = t_stat(bets)
    return {**payout_stats(bets), "yield_t": round(t, 3), "p_value": round(p, 4)}


def favorite_block(o, season_filter=None):
    oo = o if season_filter is None else o[o["season"].isin(season_filter)]
    oo = oo[oo["ml_exact_both"]]
    bets = []
    oo = oo.reset_index(drop=True)
    ph = 1 / oo["close_home"] / (1 / oo["close_home"] + 1 / oo["close_away"])
    y = (oo["hg"] > oo["ag"]).astype(int).values
    pf = (ph >= 0.5).astype(int).values
    k = 0
    for _, g in oo.iterrows():
        if pf[k] == 1:
            bets.append((float(g["close_home"]), bool(y[k])))
        else:
            bets.append((float(g["close_away"]), bool(1 - y[k])))
        k += 1
    return roi_sim(bets)


def market_move_block(o):
    oo = o[o["ml_exact_both"]].copy()
    oo["op"] = 1 / oo["open_home"] / (1 / oo["open_home"] + 1 / oo["open_away"])
    oo["cp"] = 1 / oo["close_home"] / (1 / oo["close_home"] + 1 / oo["close_away"])
    ab2b = ((oo["b2b_away"] == 1) & (oo["b2b_home"] == 0)).values
    return {"p_home_shift_away_b2b": round(float((oo["cp"][ab2b] - oo["op"][ab2b]).mean()), 4),
            "p_home_shift_rest": round(float((oo["cp"][~ab2b] - oo["op"][~ab2b]).mean()), 4),
            "n_away_b2b": int(ab2b.sum())}


def load_api(df_path, jogos_filter=True):
    df = pd.read_csv(os.path.join(OUT, df_path))
    if "gameType" in df.columns and jogos_filter:
        df = df[df["gameType"] == 2].reset_index(drop=True)
        df = df.drop(columns=[c for c in ["gameType"] if c in df.columns])
    for c in ("hg", "ag"):
        df[c] = df[c].astype(int)
    return df


def fetch_api(start, end):
    cache = os.path.join(OUT, "nhl_cache")
    os.makedirs(cache, exist_ok=True)
    UA = {"User-Agent": "Mozilla/5.0"}
    d, rows = start, []
    while d <= end:
        f = os.path.join(cache, f"{d.isoformat()}.json")
        if not os.path.exists(f):
            for attempt in range(3):
                try:
                    b = urllib.request.urlopen(urllib.request.Request(
                        f"https://api-web.nhle.com/v1/score/{d.isoformat()}", headers=UA), timeout=20).read()
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
                rows.append({"date": g["gameDate"], "home": g["homeTeam"]["abbrev"],
                             "away": g["awayTeam"]["abbrev"], "hg": g["homeTeam"].get("score"),
                             "ag": g["awayTeam"].get("score"), "gameType": g.get("gameType")})
        d += timedelta(days=1)
    df = pd.DataFrame(rows).dropna()
    df.to_csv(os.path.join(OUT, "nhl_games_2627.csv"), index=False)
    return df


def main():
    out = {}
    print("=== 1. SBR-архивы 2012-2023: парсинг 10 сезонов ===")
    for t, y in SEASONS:
        df = sbr_parse_v2(t, y)
        print(f"  {t}: {len(df)} игр, exact-линии у обеих сторон: {int(df['ml_exact_both'].sum())}")
    o = add_b2b(load_sbr_all())
    print(f"  Итого пул: {len(o)} игр {o['date'].min()} -> {o['date'].max()}")

    print("=== 2. Частотный B2B-эффект: пул 10 сезонов ===")
    out["frequency_pooled_10y"] = frequency_block(o)
    print("  ", out["frequency_pooled_10y"])
    stab = []
    ycol = (o["hg"] > o["ag"]).astype(int)
    for tag in sorted(o["season"].unique()):
        m = o["season"] == tag
        ab = ((o["b2b_away"] == 1) & (o["b2b_home"] == 0))[m]
        yb = ycol[m]
        stab.append({"season": tag, "n": int(ab.sum()),
                     "home_win_away_b2b": round(float(yb[ab].mean()), 4),
                     "home_win_rested": round(float(yb[(o["b2b_away"] == 0)[m]].mean()), 4)})
        print(f"  {tag}: away-B2B {stab[-1]['home_win_away_b2b']} vs rested {stab[-1]['home_win_rested']} (n={stab[-1]['n']})")
    out["season_stability_10y"] = stab

    print("=== 3. B2B-правило: точные цены (2018/19-2022/23, 5 окон) ===")
    exact = {}
    for tag in EXACT_SEASONS:
        r = rule_block(o, [tag])
        f = favorite_block(o, [tag])
        exact[tag] = {"rule": r, "favorite": f}
        print(f"  {tag}: правило n={r['n']} yield={r['yield']} ROI={r['roi']} | фаворит yield={f['yield']}")
    ex_all = rule_block(o, EXACT_SEASONS)
    print(f"  POOLED exact: {ex_all}")
    out["rule_exact"] = {"windows": exact, "pooled": ex_all}

    print("=== 4. Prior-блок 2012/13-2016/17 (только фаворитская сторона, цены точные) ===")
    prior = {}
    for tag in ["2012-13", "2013-14", "2014-15", "2015-16", "2016-17"]:
        r = fav_side_block(o[o["season"] == tag].copy())
        prior[tag] = r
        print(f"  {tag}: n={r['n']} yield={r['yield']}")
    pr_tags = list(prior.keys())
    prior_pooled = fav_side_block(o[o["season"].isin(pr_tags)].copy())
    print(f"  POOLED prior: {prior_pooled}")
    out["rule_favside_prior"] = {"windows": prior, "pooled": prior_pooled}

    every = fav_side_block(o.copy())
    print(f"=== 4b. Пул ВСЕХ 10 сезонов (fav-side, где цена есть): {every}")
    out["rule_all10_favside"] = every

    print("=== 5. Движение рынка Open->Close ===")
    out["market_move"] = market_move_block(o)
    print("  ", out["market_move"])

    print("=== 6. Свежий out-of-sample: Elo на 2025/26 (регулярка) ===")
    fresh = load_api("nhl_games_2627.csv")
    prm = {"k": 30.0, "home_adv": 40.0, "mov": 0.3, "rtm": 0.2}
    from iter7.run_iter7 import load_nhl_ext
    chain = pd.concat([load_nhl_ext(), fresh], ignore_index=True).drop_duplicates(
        subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)
    p, y, dates = elo_run(chain, **prm)
    d_arr = np.array(dates)
    m26 = (d_arr >= "2025-10-01") & (d_arr <= "2026-04-30")
    out["elo_fresh_2526"] = {"params": prm, "n": int(m26.sum()), **bin_metrics(y[m26], p[m26])}
    print(f"  Сезон 2025/26 (n={int(m26.sum())}): {out['elo_fresh_2526']}")

    print("=== 6b. Диагностика (post-hoc, только для отчёта, НЕ перекрутка): чувств параметров на 2025/26 ===")
    diag = {}
    for tag, alt in (("plain_K20_ha100", {"k": 20.0, "home_adv": 100.0, "mov": 0.0, "rtm": 0.0}),
                     ("k20_ha40_rtm02", {"k": 20.0, "home_adv": 40.0, "mov": 0.0, "rtm": 0.2}),
                     ("k30_ha40_rtm02_nomov", {"k": 30.0, "home_adv": 40.0, "mov": 0.0, "rtm": 0.2})):
        pa, ya, da = elo_run(chain, **alt)
        da = np.array(da)
        m = (da >= "2025-10-01") & (da <= "2026-04-30")
        diag[tag] = {**bin_metrics(ya[m], pa[m]), "n": int(m.sum())}
        print(f"  {tag}: {diag[tag]}")
    out["elo_params_diagnostic_2526"] = {"NOTE": "post-hoc диагностик, не новая настройка",
                                          **diag}

    fr = add_b2b(fresh[fresh["date"] <= "2026-04-30"].reset_index(drop=True))
    yfr = (fr["hg"] > fr["ag"]).astype(int).values
    abfr = ((fr["b2b_away"] == 1) & (fr["b2b_home"] == 0)).values
    restfr = ((fr["b2b_away"] == 0) & (fr["b2b_home"] == 0)).values
    out["fresh_freq_2526"] = {
        "away_b2b": {"n": int(abfr.sum()), "home_win": round(float(yfr[abfr].mean()), 4)} if abfr.sum() else {"n": 0},
        "away_rested": {"n": int(restfr.sum()), "home_win": round(float(yfr[restfr].mean()), 4)},
    }
    print(f"  Частота 2025/26: away-B2B {out['fresh_freq_2526']['away_b2b']} vs rested {out['fresh_freq_2526']['away_rested']}")

    print("=== 7. Paper-трекер live 2026/27 ===")
    m27 = d_arr >= "2026-10-01"
    seg = chain[m27].reset_index(drop=True)
    seg_b2b = add_b2b(seg)
    p27 = p[m27]
    rows_csv = []
    for i, g in seg_b2b.iterrows():
        y_home = bool(g["hg"] > g["ag"])
        pred_home = p27[i] >= 0.5
        signal = ""
        if g["b2b_away"] == 1 and g["b2b_home"] == 0:
            signal = "home"
        elif g["b2b_home"] == 1 and g["b2b_away"] == 0:
            signal = "away"
        rows_csv.append({"date": g["date"], "home": g["home"], "away": g["away"],
                         "p_elo_home": round(float(p27[i]), 3),
                         "pred": "H" if pred_home else "A",
                         "b2b_away": int(g["b2b_away"]), "b2b_home": int(g["b2b_home"]),
                         "rule_signal": signal,
                         "rule_winner": (signal == ("home" if y_home else "away")) if signal else "",
                         "outcome": "H" if y_home else "A",
                         "elo_correct": int((pred_home == y_home))})
    pl = pd.DataFrame(rows_csv)
    pl.to_csv(os.path.join(OUT, "live_nhl_2627.csv"), index=False)
    ab = ((seg_b2b["b2b_away"] == 1) & (seg_b2b["b2b_home"] == 0)).values
    hb = ((seg_b2b["b2b_home"] == 1) & (seg_b2b["b2b_away"] == 0)).values
    out["paper_2627"] = {"games": int(m27.sum()),
                         "elo_accuracy": out["elo_fresh_2526"]["accuracy"],
                         "elo_accuracy_live": round(float(pl["elo_correct"].mean()), 4),
                         "away_b2b_n": int(ab.sum()),
                         "away_b2b_home_win": round(float(y[m27][ab].mean()), 4) if ab.sum() else None,
                         "home_b2b_n": int(hb.sum()),
                         "home_b2b_away_win": round(float((1 - y[m27])[hb].mean()), 4) if hb.sum() else None}
    print("  ", out["paper_2627"])

    with open(os.path.join(OUT, "iter8.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    print("saved iter8.json")


if __name__ == "__main__":
    main()
