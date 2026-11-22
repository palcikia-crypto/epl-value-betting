# -*- coding: utf-8 -*-
"""
Итерация 7: превращение B2B-следа в торгуемое правило против реальных линий.

Источник: архивы SBR (sportsbookreviewsonline.com), снапшоты через web.archive.org:
2021/22 — полный сезон, 2022/23 — октябрь-ноябрь. Закрытые moneyline (Open и Close).

1) Парсинг SBR xlsx -> матчи с Open/Close ML + матчинг с официальным NHL API.
2) Расширенная история НХЛ 2019-2025: тюнинг Elo (K, домашнее преимущество, MOV,
   откат к середине между сезонами) на валидации 2023, тест 2024 -> 2025-04.
3) B2B-правило против закрытых линий: ROI/yield на тесте, контроль - фаворит.
4) Причинный вопрос: учёл ли рынок back-to-back? (движение p(home) Open -> Close).
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

SBR2ABBR = {
    "Anaheim": "ANA", "Arizona": "ARI", "Boston": "BOS", "Buffalo": "BUF",
    "Calgary": "CGY", "Carolina": "CAR", "Chicago": "CHI", "Colorado": "COL",
    "Columbus": "CBJ", "Dallas": "DAL", "Detroit": "DET", "Edmonton": "EDM",
    "Florida": "FLA", "LosAngeles": "LAK", "Minnesota": "MIN", "Montreal": "MTL",
    "Nashville": "NSH", "NewJersey": "NJD", "NYIslanders": "NYI", "NYRangers": "NYR",
    "Ottawa": "OTT", "Philadelphia": "PHI", "Pittsburgh": "PIT",
    "SeattleKraken": "SEA", "Seattle": "SEA", "SanJose": "SJS", "StLouis": "STL",
    "TampaBay": "TBL", "Toronto": "TOR", "Vancouver": "VAN", "Vegas": "VGK",
    "VegasGoldenKnights": "VGK", "Washington": "WSH", "Winnipeg": "WPG",
}


def am2dec(a):
    a = float(a)
    if a > 0:
        return 1.0 + a / 100.0
    return 1.0 + 100.0 / (-a)


def bin_metrics(y, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, int)
    return {"accuracy": round(float(((p >= 0.5).astype(int) == y).mean()), 4),
            "logloss": round(float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))), 4),
            "brier": round(float(np.mean((p - y) ** 2)), 4)}


# ---------- 1. SBR: парсинг линий ----------

def sbr_parse(season):
    start_year = int(season[:4])
    wb = openpyxl.load_workbook(os.path.join(DATA, f"nhl_odds_{season}.xlsx"), read_only=True)
    ws = wb[wb.sheetnames[0]]
    rows = [r for r in ws.iter_rows(values_only=True) if isinstance(r[0], int) and r[1] is not None and r[2] in ("V", "H", "N")]
    games = []
    i = 0
    while i < len(rows) - 1:
        a, b = rows[i], rows[i + 1]
        if int(b[1]) - int(a[1]) == 1 and a[2] in ("V", "N") and b[2] in ("H", "N"):
            def yd(d):
                s = str(d).zfill(4)
                mo = int(s[:2])
                day = int(s[2:])
                y = start_year if mo >= 7 else start_year + 1
                return f"{y}-{mo:02d}-{day:02d}"
            try:
                games.append({
                    "date": yd(a[0]),
                    "away": SBR2ABBR.get(str(a[3]).strip(), "?"),
                    "home": SBR2ABBR.get(str(b[3]).strip(), "?"),
                    "fg_away": int(a[7]), "fg_home": int(b[7]),
                    "open_away": am2dec(a[8]), "open_home": am2dec(b[8]),
                    "close_away": am2dec(a[9]), "close_home": am2dec(b[9]),
                    "neutral": a[2] == "N",
                })
            except Exception:
                pass
            i += 2
        else:
            i += 1
    return pd.DataFrame(games)


def load_odds():
    frames = [sbr_parse(s) for s in ["2021-22", "2022-23"]]
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "home", "away"])
    return df


# ---------- 2. Расширенная история + Elo-тюнинг ----------

CACHE_EXT = os.path.join(OUT, "nhl_cache_ext")


def fetch_nhl_ext(start=date(2019, 10, 2), end=date(2023, 9, 30)):
    cached = os.path.join(OUT, "nhl_games_ext.csv")
    if os.path.exists(cached):
        return pd.read_csv(cached)
    os.makedirs(CACHE_EXT, exist_ok=True)
    UA = {"User-Agent": "Mozilla/5.0"}
    d, rows = start, []
    while d <= end:
        f = os.path.join(CACHE_EXT, f"{d.isoformat()}.json")
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
            if g.get("gameType") == 2 and g.get("gameState") in ("OFF", "FINAL"):
                rows.append({"date": g["gameDate"],
                             "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                             "hg": g["homeTeam"].get("score"), "ag": g["awayTeam"].get("score")})
        d += timedelta(days=1)
    df = pd.DataFrame(rows).dropna()
    df["hg"] = df["hg"].astype(int)
    df["ag"] = df["ag"].astype(int)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(cached, index=False)
    print(f"NHL ext fetched: {len(df)} games -> {cached}")
    return df


def load_nhl_ext():
    base = pd.read_csv(os.path.join(OUT, "nhl_games.csv"))
    p = os.path.join(OUT, "nhl_games_ext.csv")
    if os.path.exists(p):
        ext = pd.read_csv(p)
        df = pd.concat([base, ext], ignore_index=True).drop_duplicates(
            subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)
    else:
        df = base
    print(f"NHL история: {len(df)} игр {df['date'].min()} -> {df['date'].max()}")
    return df


def elo_run(df, k=20.0, home_adv=60.0, mov=0.0, rtm=0.0):
    elo, ngame = {}, {}
    probs, ys, dates = [], [], []
    last_month = None

    def kk_fn(h, a, margin):
        base = k / max(((ngame.get(h, 0) + ngame.get(a, 0)) / 2 + 5) ** 0.4, 1.0)
        return base * (1 + mov * margin)

    for _, g in df.iterrows():
        d = str(g["date"])
        if d[5:7] == "10" and last_month != d[:7] and rtm > 0:
            for t in list(elo):
                elo[t] = 1500.0 + (elo[t] - 1500.0) * (1 - rtm)
        last_month = d[:7]
        h, a = g["home"], g["away"]
        eh, ea = elo.get(h, 1500.0), elo.get(a, 1500.0)
        p_h = 1 / (1 + 10 ** (-((eh + home_adv) - ea) / 400))
        probs.append(p_h)
        s_h = 1.0 if g["hg"] > g["ag"] else 0.0
        ys.append(s_h)
        dates.append(d)
        margin = abs(int(g["hg"]) - int(g["ag"]))
        ngame[h] = ngame.get(h, 0) + 1
        ngame[a] = ngame.get(a, 0) + 1
        upd = kk_fn(h, a, margin)
        elo[h] = eh + upd * (s_h - p_h)
        elo[a] = ea + upd * ((1 - s_h) - (1 - p_h))
    return np.array(probs), np.array(ys), np.array(dates)


def aa_mask(y, p, m):
    return bin_metrics(y[m], p[m])


def tune_elo(df_all):
    print("=== НХЛ: тюнинг Elo (val 2023, test 2024->2025-04) ===")
    yrs = df_all["date"].str[:4].astype(int)
    te_mask_val = (yrs == 2023).values
    te_mask_test = (yrs >= 2024).values

    grid = [(k, ha, mv, rt)
            for k in (10.0, 15.0, 20.0, 30.0)
            for ha in (40.0, 60.0, 80.0)
            for mv in (0.0, 0.15, 0.3)
            for rt in (0.0, 0.2)]
    best = None
    for k, ha, mv, rt in grid:
        p, y, dates = elo_run(df_all[yrs <= 2023], k=k, home_adv=ha, mov=mv, rtm=rt)
        m = (np.array(dates) >= "2023-01-01") & (np.array(dates) <= "2023-12-31")
        if m.sum() < 400:
            continue
        r = bin_metrics(y[m], p[m])
        if best is None or r["logloss"] < best[0]:
            best = (r["logloss"], {"k": k, "home_adv": ha, "mov": mv, "rtm": rt}, r)
    ll_val, prm, val_r = best
    print(f"  Val 2023: {prm} -> logloss={ll_val} acc={val_r['accuracy']}")

    p, y, dates = elo_run(df_all, **prm)
    dates_arr = np.array(dates)
    cal24 = (dates_arr >= "2024-01-01") & (dates_arr <= "2024-12-31")
    out = {"best_params": prm, "val_logloss": round(ll_val, 4), "val": val_r,
           "test_2024_2025": bin_metrics(y[te_mask_test], p[te_mask_test]),
           "test_cal2024": bin_metrics(y[cal24], p[cal24]),
           "n_test": int(te_mask_test.sum()), "n_test_cal2024": int(cal24.sum())}
    print(f"  Тест 2024-2025-04 (n={out['n_test']}): {out['test_2024_2025']}")
    print(f"  Календарный 2024 (n={out['n_test_cal2024']}): {out['test_cal2024']}")
    return out, p, dates_arr, y


# ---------- 3. B2B против линий ----------

def add_rest(df):
    df = df.drop(columns=[c for c in df.columns if c.startswith(("rest_", "b2b_"))], errors="ignore")
    df = df.sort_values("date").reset_index(drop=True)
    last = {}
    flags = []
    for _, g in df.iterrows():
        d = pd.Timestamp(g["date"])
        row = {}
        for side, t in (("home", g["home"]), ("away", g["away"])):
            r = int((d - last.get(t, d - pd.Timedelta(days=9))).days)
            row[f"rest_{side}"] = min(r, 7)
            row[f"b2b_{side}"] = int(r <= 1)
        flags.append(row)
        last[g["home"]] = d
        last[g["away"]] = d
    return pd.concat([df, pd.DataFrame(flags)], axis=1)


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


def roi_sim(bets, init=10000.0, frac=0.01):
    bank = init
    staked = profit = 0.0
    wins = 0
    peak = bank
    dd = 0.0
    for o, won in bets:
        stake = bank * frac
        pnl = stake * (o - 1.0) if won else -stake
        bank += pnl
        staked += stake
        profit += pnl
        wins += int(won)
        peak = max(peak, bank)
        dd = max(dd, (peak - bank) / peak)
    n = len(bets)
    return {"n": n, "roi": round((bank - init) / init, 4),
            "yield": round(profit / staked, 4) if staked else 0.0,
            "wr": round(wins / n, 4) if n else 0.0, "dd": round(dd, 4)}


def b2b_rule(df_all, odds):
    print("=== B2B-правило против закрытых линий ===")
    o = odds.merge(df_all[["date", "home", "away", "hg", "ag"]],
                   on=["date", "home", "away"], how="inner").sort_values("date").reset_index(drop=True)
    print(f"Матчинг SBR<->API: {len(o)} игр из {len(odds)} строк линий")
    o = add_rest(o)
    y = (o["hg"] > o["ag"]).astype(int).values
    out = {"n_matched": int(len(o))}

    ph = 1 / o["close_home"] / (1 / o["close_home"] + 1 / o["close_away"])
    out["market_all"] = bin_metrics(y, ph.values)

    res = {}
    for tag, mask_season in (("VAL_2021_22", o["date"].str[:4] == "2021"),
                             ("TEST_2022_23", o["date"].str[:4] == "2022")):
        seg = o[mask_season.values].reset_index(drop=True)
        ys = (seg["hg"] > seg["ag"]).astype(int).values
        ab2b = ((seg["b2b_away"] == 1) & (seg["b2b_home"] == 0)).values
        hb2b = ((seg["b2b_home"] == 1) & (seg["b2b_away"] == 0)).values

        bets = ([(float(seg.iloc[i]["close_home"]), bool(ys[i])) for i in np.where(ab2b)[0]] +
                [(float(seg.iloc[i]["close_away"]), bool(1 - ys[i])) for i in np.where(hb2b)[0]])
        bets.sort(key=lambda b: 0)
        rule = roi_sim(bets)

        ph_seg = 1 / seg["close_home"].values / (1 / seg["close_home"].values + 1 / seg["close_away"].values)
        pf = (ph_seg >= 0.5).astype(int)
        fav_odds = np.where(pf == 1, seg["close_home"].values, seg["close_away"].values)
        fav_won = (ys.astype(bool) == (pf == 1))
        fav = roi_sim([(float(fav_odds[i]), bool(fav_won[i])) for i in range(len(seg))])

        # стратификация правила по кэфу на уставшую противоположную сторону
        strata = []
        if ab2b.sum() and hb2b.sum():
            odds_fav_side = np.where(ab2b, seg["close_home"].values, seg["close_away"].values)
            for lo, hi in [(1.0, 1.7), (1.7, 2.2), (2.2, 9.0)]:
                m = ((odds_fav_side > lo) & (odds_fav_side <= hi))
                sb = [b for b, mm in zip(bets, np.concatenate([ab2b, hb2b])) if mm and (lo < b[0] <= hi)]
                if sb:
                    strata.append({"odds_bucket": f"({lo},{hi}]", **roi_sim(sb)})

        ph0 = 1 / seg["open_home"] / (1 / seg["open_home"] + 1 / seg["open_away"])
        ph1 = 1 / seg["close_home"] / (1 / seg["close_home"] + 1 / seg["close_away"])
        mv_ab2b = float((ph1[ab2b] - ph0[ab2b]).mean()) if ab2b.sum() else None
        mv_rest = float((ph1[~ab2b] - ph0[~ab2b]).mean())

        eff_home = float(ys.astype(float)[ab2b].mean()) if ab2b.sum() else None
        res[tag] = {"b2b_rule": rule, "favorite": fav, "b2b_strata": strata,
                    "n_away_b2b": int(ab2b.sum()), "n_home_b2b": int(hb2b.sum()),
                    "home_win_rate_away_b2b": round(eff_home, 4) if eff_home is not None else None,
                    "home_win_rate_base": round(float(ys.mean()), 4),
                    "market_move": {"p_home_shift_away_b2b": round(mv_ab2b, 4) if mv_ab2b is not None else None,
                                    "p_home_shift_rest": round(mv_rest, 4)}}
        print(f"  {tag}: правило={rule} | фаворит={fav} | away-B2B n={ab2b.sum()} "
              f"home win={eff_home} | база {res[tag]['home_win_rate_base']}")
        print(f"  {tag}: движение p(home) Open->Close: on away-B2B {mv_ab2b:+.4f} vs rest {mv_rest:+.4f}")

    out["segments"] = res

    bets_all = [(float(o.iloc[i]["close_home"]), bool(o.iloc[i]["hg"] > o.iloc[i]["ag"]))
                for i in np.where(((o["b2b_away"] == 1) & (o["b2b_home"] == 0)).values)[0]] + \
               [(float(o.iloc[i]["close_away"]), bool(o.iloc[i]["hg"] <= o.iloc[i]["ag"]))
                for i in np.where(((o["b2b_home"] == 1) & (o["b2b_away"] == 0)).values)[0]]
    payoffs = [(od - 1.0 if won else -1.0) for od, won in bets_all]
    mu, sd = float(np.mean(payoffs)), float(np.std(payoffs, ddof=1))
    t = mu / (sd / math.sqrt(len(payoffs)))
    from scipy import stats as st
    out["pooled_rule"] = {**roi_sim(bets_all), "yield_t": round(t, 3),
                          "p_value": round(float(2 * (1 - st.norm.cdf(abs(t)))), 4)}
    print(f"  POOLED (val+test): {out['pooled_rule']}")

    big = add_rest(df_all)
    yrs = big["date"].str[:4].astype(int)
    stability = []
    for yr in sorted(yrs.unique()):
        seg = big[yrs == yr]
        ab2b = (seg["b2b_away"] == 1) & (seg["b2b_home"] == 0)
        yb = (seg["hg"] > seg["ag"]).astype(int)
        if ab2b.sum() >= 20:
            stability.append({"season": str(yr), "n": int(ab2b.sum()),
                              "home_win_away_b2b": round(float(yb[ab2b].mean()), 4),
                              "home_win_away_rested": round(float(yb[(seg["b2b_away"] == 0)].mean()), 4)})
    out["b2b_stability"] = stability
    for r in stability:
        print(f"  {r['season']}: away-B2B home win={r['home_win_away_b2b']} vs rested {r['home_win_away_rested']} (n={r['n']})")

    yb_all = (big["hg"] > big["ag"]).astype(int).values
    ab2b_all = ((big["b2b_away"] == 1) & (big["b2b_home"] == 0)).values
    rest_all = ((big["b2b_away"] == 0) & (big["b2b_home"] == 0)).values
    hb2b_all = ((big["b2b_home"] == 1) & (big["b2b_away"] == 0)).values
    z_rest = two_prop_z(int(yb_all[ab2b_all].sum()), int(ab2b_all.sum()),
                        int(yb_all[rest_all].sum()), int(rest_all.sum()))
    z_hb2b = two_prop_z(int(yb_all[rest_all].sum()), int(rest_all.sum()),
                        int(yb_all[hb2b_all].sum()), int(hb2b_all.sum()))
    from scipy import stats as st2
    out["b2b_frequency_pooled"] = {
        "away_b2b": {"n": int(ab2b_all.sum()), "home_win": round(float(yb_all[ab2b_all].mean()), 4),
                     "ci95": list(wilson(float(yb_all[ab2b_all].mean()), int(ab2b_all.sum())))},
        "away_rested": {"n": int(rest_all.sum()), "home_win": round(float(yb_all[rest_all].mean()), 4),
                        "ci95": list(wilson(float(yb_all[rest_all].mean()), int(rest_all.sum())))},
        "home_b2b": {"n": int(hb2b_all.sum()), "home_win": round(float(yb_all[hb2b_all].mean()), 4),
                     "ci95": list(wilson(float(yb_all[hb2b_all].mean()), int(hb2b_all.sum())))},
        "z_away_b2b_vs_rested": round(float(z_rest), 3),
        "p_away_b2b_vs_rested": round(float(2 * (1 - st2.norm.cdf(abs(z_rest)))), 4),
    }
    print("  Пул 7 сезонов:", out["b2b_frequency_pooled"])
    return out


def main():
    fetch_nhl_ext()
    df_all = load_nhl_ext()
    out = {}
    tune, probs, dates, ys = tune_elo(df_all)
    out["elo"] = tune
    df_all = add_rest(df_all)

    odds = load_odds()
    good = odds[(odds["home"] != "?") & (odds["away"] != "?")]
    out["unmatched_teams"] = sorted(set(odds["home"]).union(odds["away"]) - set(df_all["home"]).union(df_all["away"]))
    out["n_odds_rows"] = len(odds)

    res_b2b = b2b_rule(df_all, good)
    out["b2b"] = res_b2b

    mk = good.merge(df_all[["date", "home", "away", "hg", "ag"]], on=["date", "home", "away"]).sort_values("date")
    mk = add_rest(mk)
    m22 = (mk["date"] >= "2022-10-01")
    ph22 = 1 / mk["close_home"] / (1 / mk["close_home"] + 1 / mk["close_away"])
    yh22 = (mk["hg"] > mk["ag"]).astype(int).values
    k_prm = tune["best_params"]
    p_all, y_all, d_all = elo_run(df_all, **k_prm)
    key_map = {(d, h, a): p for d, h, a, p in
               zip(df_all["date"], df_all["home"], df_all["away"], p_all)}
    elo22 = np.array([key_map.get((d, h, a), 0.5) for d, h, a in
                      zip(mk[m22]["date"], mk[m22]["home"], mk[m22]["away"])])
    out["elo_vs_market_2022"] = {
        "market": bin_metrics(yh22[m22.values], ph22[m22.values]),
        "elo": bin_metrics(yh22[m22.values], elo22),
        "n": int(m22.sum())}
    print(f"  TEST 2022/23 (окт-ноя, n={out['elo_vs_market_2022']['n']}): рынок {out['elo_vs_market_2022']['market']} | Elo {out['elo_vs_market_2022']['elo']}")

    yrs2 = df_all["date"].str[:4].astype(int)
    ab2b_cal = ((df_all["b2b_away"] == 1) & (df_all["b2b_home"] == 0) & (yrs2 < 2024)).values
    rest_cal = ((df_all["b2b_away"] == 0) & (df_all["b2b_home"] == 0) & (yrs2 < 2024)).values
    delta = float(y_all[ab2b_cal].mean() - y_all[rest_cal].mean()) if ab2b_cal.sum() > 100 else 0.0
    te_cal = (yrs2 >= 2024)
    p_adj = np.clip(p_all[te_cal] + delta * df_all.loc[te_cal, "b2b_away"].values, 0.01, 0.99)
    out["elo_b2b_adjustment"] = {"delta_train": round(delta, 4),
                                 "elo": bin_metrics(y_all[te_cal], p_all[te_cal]),
                                 "elo_b2b_adj": bin_metrics(y_all[te_cal], p_adj)}
    print(f"  Тест 2024-2025: Elo {out['elo_b2b_adjustment']['elo']} | +B2B-поправка {out['elo_b2b_adjustment']['elo_b2b_adj']} (δ={delta:+.4f})")

    with open(os.path.join(OUT, "iter7.json"), "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False, default=str)
    print("saved iter7.json")


if __name__ == "__main__":
    main()
