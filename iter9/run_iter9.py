# -*- coding: utf-8 -*-
"""
Итерация 9: новые источники данных + причинность усталости по всем спортам.

1) Polymarket (gamma + CLOB): исторические деньги-лайн NHL 2025/26 + 2026/27,
   pre-game crowd-вероятность -> сравнение с нашим tuned Elo на той же выборке,
   + анализ расхождений poly-vs-elo.
2) NFL SBR-архивы (2018/19-2021/22, Wayback): аналог усталости - короткая неделя
   (3-4 дня отдыха vs нормальные 6-8). Частота + правило против реальных ML-закрытий.
3) NCAAB SBR-архивы (2018/19-2021/22): back-to-back в колледже. Частота + правило.
4) ATP/tennis-data 2021-2025: усталость = предыдущий матч <=1 день и 3+ сета /
   много геймов. Проверка против PS-закрытий.
5) NHL extra-load: OT/буллиты накануне (period>=4) поверх B2B - из кэша score-API.
6) Сводный стек всех сигналов проекта.
"""
import json
import math
import os
import sys
from datetime import timedelta

import numpy as np
import pandas as pd
import openpyxl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

from iter7.run_iter7 import am2dec, bin_metrics, elo_run, roi_sim, two_prop_z, wilson
from iter8.run_iter8 import add_b2b, t_stat
from iter7.run_iter7 import load_nhl_ext


def payout_stats(bets, init=10000.0, frac=0.01):
    return roi_sim([(o, w) for _, o, w, _ in bets], init=init, frac=frac)


# ---------- SBR generic parser (NFL / NCAAB) ----------
def find_col(hdr, name):
    for i, h in enumerate(hdr):
        if str(h).strip() == name:
            return i
    raise KeyError(name)


def sbr_parse_generic(path, start_year, sport):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    rows_iter = ws.iter_rows(values_only=True)
    hdr = [c for c in next(rows_iter)]
    ifin, iml = find_col(hdr, "Final"), find_col(hdr, "ML")
    rows = list(rows_iter)
    wb.close()
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
        dstr = "%d-%02d-%02d" % (year, mo, day)
        try:
            hg, ag = int(b[ifin]), int(a[ifin])
        except (ValueError, TypeError):
            i += 2
            continue
        g = {"season": start_year if sport == "nfl" else start_year,
             "date": dstr, "home": str(b[3]).strip(), "away": str(a[3]).strip(),
             "hg": hg, "ag": ag}
        for side, row in (("away", a), ("home", b)):
            v = row[iml]
            if isinstance(v, (int, float)) and not isinstance(v, bool) and abs(float(v)) >= 95:
                g["ml_%s" % side] = am2dec(v)
        games.append(g)
        i += 2
    return pd.DataFrame(games)


def add_rest_raw(df, cap):
    df = df.sort_values("date").reset_index(drop=True)
    last = {}
    flags = []
    for _, g in df.iterrows():
        d = pd.Timestamp(g["date"])
        row = {}
        for side, t in (("home", g["home"]), ("away", g["away"])):
            r = int((d - last.get(t, d - timedelta(days=cap + 2))).days)
            row["rest_%s" % side] = min(r, cap)
        flags.append(row)
        last[g["home"]] = d
        last[g["away"]] = d
    return pd.concat([df, pd.DataFrame(flags)], axis=1)


# ---------- NFL ----------
def nfl_main(out):
    print("=== 2. NFL SBR 2018-2022: короткая неделя как аналог B2B ===")
    frames = []
    for tag in ["2018", "2019", "2020", "2021"]:
        p = os.path.join(ROOT, "data", "nfl_odds", "nfl_odds_%s-%s.xlsx" % (tag, str(int(tag) + 1)[2:]))
        d = sbr_parse_generic(p, tag, "nfl")
        print("  NFL %s/%s: %d игр (ML обе стороны: %d)" % (tag, int(tag) + 1, len(d), int(d["ml_home"].notna().sum() & d["ml_away"].notna().sum())))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)
    df = add_rest_raw(df, cap=14)
    y = (df["hg"] > df["ag"]).astype(int).values
    base = ((df["rest_away"] >= 6) & (df["rest_away"] <= 9) & (df["rest_home"] >= 6) & (df["rest_home"] <= 9)).values
    buckets = {
        "both_normal_6_9": base,
        "both_short_4_5": ((df["rest_away"] <= 5) & (df["rest_home"] <= 5)).values,
        "away_short_home_norm": ((df["rest_away"] <= 5) & (df["rest_home"] >= 6) & (df["rest_home"] <= 9)).values,
        "home_short_away_norm": ((df["rest_home"] <= 5) & (df["rest_away"] >= 6) & (df["rest_away"] <= 9)).values,
        "away_bye_home_norm": ((df["rest_away"] >= 10) & (df["rest_home"] >= 6) & (df["rest_home"] <= 9)).values,
        "home_bye_away_norm": ((df["rest_home"] >= 10) & (df["rest_away"] >= 6) & (df["rest_away"] <= 9)).values,
    }
    res = {}
    for k, m in buckets.items():
        w, n = int(y[m].sum()), int(m.sum())
        res[k] = {"n": n, "home_win": round(w / n, 4) if n else None}
        print("  %-22s n=%5d home_win=%.4f" % (k, n, w / n if n else 0))
    nb = int(base.sum())
    res["base_home_win"] = round(float(y[base].mean()), 4)
    zstats = {}
    for mk in ["away_short_home_norm", "home_short_away_norm", "away_bye_home_norm", "home_bye_away_norm"]:
        m = buckets[mk]
        if int(m.sum()) > 20:
            z = two_prop_z(int(y[m].sum()), int(m.sum()), int(y[base].sum()), nb)
            zstats[mk] = round(z, 3)
    res["z_vs_base"] = zstats

    bets = []
    for i, g in df.iterrows():
        if g["rest_home"] >= 10 and 6 <= g["rest_away"] <= 9 and pd.notna(g["ml_home"]):
            bets.append((int(i), float(g["ml_home"]), bool(g["hg"] > g["ag"]), "home"))
        elif g["rest_away"] >= 10 and 6 <= g["rest_home"] <= 9 and pd.notna(g["ml_away"]):
            bets.append((int(i), float(g["ml_away"]), not bool(g["hg"] > g["ag"]), "away"))
    st = payout_stats(bets)
    t, pv = t_stat(bets)
    res["rule_bet_bye_team"] = {"n": len(bets), **st, "yield_t": round(t, 3), "p_value": round(pv, 5)}
    print("  Правило (бай против норм отдыха): %d ставок yield=%.2f%%" % (len(bets), 100 * st["yield"]))
    out["nfl_fatigue"] = res


# ---------- NCAAB ----------
def ncaab_main(out):
    print("=== 3. NCAAB SBR 2018-2022: back-to-back в колледже ===")
    frames = []
    for tag in ["2018", "2019", "2020", "2021"]:
        p = os.path.join(ROOT, "data", "ncaab_odds", "ncaa_basketball_%s-%s.xlsx" % (tag, str(int(tag) + 1)[2:]))
        d = sbr_parse_generic(p, tag, "ncaab")
        print("  NCAAB %s/%s: %d игр" % (tag, int(tag) + 1, len(d)))
        frames.append(d)
    df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)
    df = add_rest_raw(df, cap=7)
    y = (df["hg"] > df["ag"]).astype(int).values
    base = ((df["rest_away"] >= 2) & (df["rest_home"] >= 2)).values
    buckets = {
        "away_b2b_home_rested": ((df["rest_away"] <= 1) & (df["rest_home"] >= 2)).values,
        "home_b2b_away_rested": ((df["rest_home"] <= 1) & (df["rest_away"] >= 2)).values,
        "both_rested": base,
    }
    res = {}
    nb = int(base.sum())
    res["base_home_win"] = round(float(y[base].mean()), 4)
    for k, m in buckets.items():
        w, n = int(y[m].sum()), int(m.sum())
        res[k] = {"n": n, "home_win": round(w / n, 4) if n else None}
        print("  %-22s n=%5d home_win=%.4f" % (k, n, w / n if n else 0))
    for key, mk in [("away_b2b_vs_base_z", "away_b2b_home_rested"), ("home_b2b_vs_base_z", "home_b2b_away_rested")]:
        m = buckets[mk]
        if int(m.sum()) > 20:
            res[key] = round(two_prop_z(int(y[m].sum()), int(m.sum()), int(y[base].sum()), nb), 3)
    bets = []
    for i, g in df.iterrows():
        if g["rest_away"] <= 1 and g["rest_home"] >= 2 and pd.notna(g["ml_home"]):
            bets.append((int(i), float(g["ml_home"]), bool(g["hg"] > g["ag"]), "home"))
    st = payout_stats(bets)
    t, pv = t_stat(bets)
    res["rule_bet_home_vs_away_b2b"] = {"n": len(bets), **st, "yield_t": round(t, 3), "p_value": round(pv, 5)}
    print("  Правило B2B-гость: %d ставок yield=%.2f%%" % (len(bets), 100 * st["yield"]))
    out["ncaab_fatigue"] = res


# ---------- ATP ----------
def tennis_load(path):
    wb = openpyxl.load_workbook(path, read_only=True)
    ws = wb.active
    it = ws.iter_rows(values_only=True)
    hdr = [str(c) for c in next(it)]
    recs = []
    for r in it:
        d = dict(zip(hdr, r))
        if d.get("Comment") != "Completed":
            continue
        recs.append(d)
    wb.close()
    return recs


def tennis_main(out):
    print("=== 4. ATP 2021-2025: усталость по прошлому матчу (дни + нагрузка) ===")
    recs = []
    for y_ in [2021, 2022, 2023, 2024, 2025]:
        recs += tennis_load(os.path.join(ROOT, "data", "tennis", "atp_%d.xlsx" % y_))
    for r in recs:
        r["date"] = pd.to_datetime(r["Date"])
        games_played = 0
        for i_ in range(1, 6):
            w_, l_ = r.get("W%d" % i_), r.get("L%d" % i_)
            if w_ is not None and l_ is not None:
                games_played += int(w_) + int(l_)
        r["tot_games"] = games_played
    recs.sort(key=lambda r: r["date"])

    def devig(ow, ol):
        iw, il = 1.0 / ow, 1.0 / ol
        return iw / (iw + il)

    seq = {}
    instances = []
    for r in recs:
        d = r["date"]
        w_, l_ = str(r["Winner"]), str(r["Loser"])
        inst = {}

        def state(p):
            prev = seq.get(p)
            if prev is None:
                return None
            gap = (d - prev["date"]).days
            if gap > 2:
                return None
            heavy = prev["tot_games"] >= 22 or prev["sets"] >= 3
            return {"gap": gap, "heavy": heavy and gap <= 1}

        def odds_of(player, is_w):
            o = r["PSW"] if is_w else r["PSL"]
            return float(o) if o is not None and float(o) > 1 else None

        ow, ol = odds_of(w_, True), odds_of(l_, False)
        if not ow or not ol:
            seq[w_] = {"date": d, "sets": r["Wsets"], "tot_games": r["tot_games"]}
            seq[l_] = {"date": d, "sets": r["Lsets"], "tot_games": r["tot_games"]}
            continue
        p_fresh_w = devig(ow, ol)
        sw, sl = state(w_), state(l_)
        instances.append({"date": d, "aw": w_, "al": l_, "ow": ow, "ol": ol,
                          "p_w": p_fresh_w, "sw": sw, "sl": sl,
                          "w_win": True})
        seq[w_] = {"date": d, "sets": r["Wsets"], "tot_games": r["tot_games"]}
        seq[l_] = {"date": d, "sets": r["Lsets"], "tot_games": r["tot_games"]}

    fat_stats = {"n_fat_any": 0, "win_when_fat": 0, "exp_when_fat": 0.0}
    for it in instances:
        for side in ("w", "l"):
            me = it["s" + side]
            opp = it["s" + ("l" if side == "w" else "w")]
            if me is None or opp is None or not me.get("heavy"):
                continue
            if opp.get("heavy"):
                continue
            fat_stats["n_fat_any"] += 1
            fat_stats["win_when_fat"] += int(side == "w")
            p_mine = it["p_w"] if side == "w" else 1 - it["p_w"]
            fat_stats["exp_when_fat"] += p_mine
            o_mine = it["ow"] if side == "w" else it["ol"]
            fat_stats.setdefault("profit", 0.0)
            fat_stats["profit"] += (o_mine * (side == "w") - 1.0)

    n_f = fat_stats["n_fat_any"]
    res = {"n_instances_with_fat_state": n_f,
           "win_rate_when_fatigued": round(fat_stats["win_when_fat"] / n_f, 4) if n_f else None,
           "expected_win_rate_odds": round(fat_stats["exp_when_fat"] / n_f, 4) if n_f else None}
    if n_f:
        res["fatigue_delta"] = round(res["win_rate_when_fatigued"] - res["expected_win_rate_odds"], 4)
        res["bet_fatigued_yield"] = round(fat_stats["profit"] / n_f, 4)
        pb = res["win_rate_when_fatigued"]
        se = math.sqrt(pb * (1 - pb) / n_f)
        res["delta_z"] = round((res["win_rate_when_fatigued"] - res["expected_win_rate_odds"]) / se, 3) if se > 0 else 0.0
    print("  Игрок играет после плотного матча (<=1 день, 3+ сета или 22+ гейма): n=%d" % n_f)
    print("  Факт win=%.4f vs рыночная ожидаемая=%.4f -> delta=%.2f пп" % (
        res["win_rate_when_fatigued"], res["expected_win_rate_odds"], 100 * res.get("fatigue_delta", 0)))
    out["atp_fatigue"] = res


# ---------- NHL extra-load (OT накануне) ----------
def nhl_iter_cache(dname):
    dpath = os.path.join(OUT, dname)
    if not os.path.isdir(dpath):
        return
    for f in sorted(os.listdir(dpath)):
        if not f.endswith(".json"):
            continue
        try:
            j = json.load(open(os.path.join(dpath, f)))
        except Exception:
            continue
        for g in j.get("games", []):
            if g.get("gameType") != 2 or g.get("gameState") not in ("OFF", "FINAL"):
                continue
            yield {"date": g["gameDate"],
                   "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                   "hg": g["homeTeam"].get("score"), "ag": g["awayTeam"].get("score"),
                   "period": g.get("period", 3)}


def nhl_load_main(out):
    print("=== 5. NHL: сверхнагрузка - OT/буллит накануне ===")
    rows = list(nhl_iter_cache("nhl_cache_ext")) + list(nhl_iter_cache("nhl_cache"))
    df = pd.DataFrame(rows).drop_duplicates(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)
    df = add_rest_raw(df, cap=7)
    last_meta = {}
    ot_last = {}
    for _, g in df.iterrows():
        d = pd.Timestamp(g["date"])
        for t in (g["home"], g["away"]):
            ot_last[t] = last_meta.get(t, {}).get("ot", False)
        meta = {"ot": int(g["period"] or 3) >= 4}
        last_meta[g["home"]] = {"d": d, "ot": meta["ot"]}
        last_meta[g["away"]] = {"d": d, "ot": meta["ot"]}
    df["ot_last_away"] = [ot_last[g["away"]] for _, g in df.iterrows()]
    df["ot_last_home"] = [ot_last[g["home"]] for _, g in df.iterrows()]
    y = (df["hg"] > df["ag"]).astype(int).values
    b2b = (df["rest_away"] <= 1)
    res = {}
    combos = {
        "away_b2b_ot_last": (b2b & df["ot_last_away"]),
        "away_b2b_no_ot": (b2b & ~df["ot_last_away"]),
        "away_rested_ot_last": (~b2b & df["ot_last_away"]),
        "away_rested_no_ot": (~b2b & ~df["ot_last_away"]),
    }
    for k, m in combos.items():
        w, n = int(y[m.values].sum()), int(m.sum())
        res[k] = {"n": n, "home_win": round(w / n, 4) if n else None}
        print("  %-22s n=%5d home_win=%.4f" % (k, n, w / n if n else 0))
    zb = two_prop_z(int(y[combos["away_b2b_ot_last"].values].sum()), int(combos["away_b2b_ot_last"].sum()),
                    int(y[combos["away_b2b_no_ot"].values].sum()), int(combos["away_b2b_no_ot"].sum()))
    zr = two_prop_z(int(y[combos["away_rested_ot_last"].values].sum()), int(combos["away_rested_ot_last"].sum()),
                    int(y[combos["away_rested_no_ot"].values].sum()), int(combos["away_rested_no_ot"].sum()))
    res["z_b2b_ot_vs_b2b"] = round(zb, 3)
    res["z_rested_ot_vs_rested"] = round(zr, 3)
    print("  z: B2B+OT vs B2B = %.2f | OT vs no-OT (отдыхшие) = %.2f" % (zb, zr))
    out["nhl_ot_load"] = res


# ---------- Polymarket ----------
TEAM2ABBREV = {
    "Ducks": "ANA", "Bruins": "BOS", "Sabres": "BUF", "Flames": "CGY",
    "Hurricanes": "CAR", "Blackhawks": "CHI", "Avalanche": "COL",
    "Blue Jackets": "CBJ", "Stars": "DAL", "Red Wings": "DET", "Oilers": "EDM",
    "Panthers": "FLA", "Kings": "LAK", "Wild": "MIN", "Canadiens": "MTL",
    "Predators": "NSH", "Devils": "NJD", "Islanders": "NYI", "Rangers": "NYR",
    "Senators": "OTT", "Flyers": "PHI", "Penguins": "PIT", "Sharks": "SJS",
    "Kraken": "SEA", "Blues": "STL", "Lightning": "TBL", "Maple Leafs": "TOR",
    "Utah": "UTA", "Utah HC": "UTA", "Utah Hockey Club": "UTA", "Mammoth": "UTA",
    "Canucks": "VAN", "Golden Knights": "VGK", "Capitals": "WAS", "Jets": "WPN",
}


def polymarket_block(out):
    print("=== 6. Polymarket: crowd-вероятности NHL 2025/26 + 2026/27 ===")
    meta = json.load(open(os.path.join(OUT, "polymarket_nhl", "meta.json")))
    games = pd.read_csv(os.path.join(OUT, "nhl_games_2627.csv"))
    games = games[games["gameType"] == 2].reset_index(drop=True)
    by_pair = {}
    for i, g in games.iterrows():
        by_pair.setdefault(frozenset([g["home"], g["away"]]), []).append(i)

    recs = []
    for g in meta:
        outs, start = g["outs"], g["start"]
        t0 = pd.Timestamp(start).tz_localize(None)
        try:
            hist0 = json.load(open(os.path.join(OUT, "polymarket_nhl", "%s.json" % g["cid"])))["hist0"]
            hist1 = json.load(open(os.path.join(OUT, "polymarket_nhl", "%s.json" % g["cid"])))["hist1"]
        except Exception:
            continue
        if not hist0 or not hist1:
            continue
        cut = int(t0.timestamp()) - 600
        p0 = [h["p"] for h in hist0 if h["t"] <= cut]
        p1 = [h["p"] for h in hist1 if h["t"] <= cut]
        if not p0 or not p1:
            continue
        p0, p1 = p0[-1], p1[-1]
        if p0 <= 0.01 or p1 <= 0.01:
            continue
        abbrevs = [TEAM2ABBREV.get(o.strip()) for o in outs]
        if None in abbrevs:
            continue
        key = frozenset(abbrevs)
        if key not in by_pair:
            continue
        for ii in by_pair[key]:
            row = games.loc[ii]
            pd_date = pd.Timestamp(row["date"])
            if abs((pd.Timestamp(t0.date()) - pd_date).days) > 1:
                continue
            home_tok = 0 if TEAM2ABBREV.get(outs[0].strip()) == row["home"] else (1 if TEAM2ABBREV.get(outs[1].strip()) == row["home"] else None)
            if home_tok is None:
                continue
            p_home = p0 if home_tok == 0 else p1
            p_home = p_home / (p_home + (p1 if home_tok == 0 else p0))
            recs.append({"date": row["date"], "home": row["home"], "away": row["away"],
                         "hg": row["hg"], "ag": row["ag"],
                         "p_poly_home": p_home, "start": start})
            break
    df = pd.DataFrame(recs).drop_duplicates(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)
    print("  Matching poly<->NHL API: %d игр" % len(df))

    chain = pd.concat([load_nhl_ext(),
                       pd.read_csv(os.path.join(OUT, "nhl_games.csv")),
                       pd.read_csv(os.path.join(OUT, "nhl_games_2627.csv"))],
                      ignore_index=True)
    if "gameType" in chain.columns:
        chain = chain[chain["gameType"] == 2]
    chain = chain.drop_duplicates(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)
    chain["rowid"] = chain.index
    prm = {"k": 30.0, "home_adv": 40.0, "mov": 0.3, "rtm": 0.2}
    p, yy, _ = elo_run(chain, **prm)
    rid = dict(zip(chain["rowid"], chain.index))
    lookup = chain.set_index([chain["date"], chain["home"], chain["away"]])["rowid"]
    p_elo, y_true = [], []
    for _, g in df.iterrows():
        try:
            idx = lookup.get((g["date"], g["home"], g["away"]))
        except Exception:
            idx = None
        if idx is None:
            p_elo.append(np.nan)
            y_true.append(np.nan)
        else:
            p_elo.append(p[idx])
            y_true.append(yy[idx])
    df["p_elo_home"] = p_elo
    df["y"] = (df["hg"] > df["ag"]).astype(int)

    m = df["p_elo_home"].notna().values
    yv = df["y"].values
    poly = bin_metrics(yv[m], df["p_poly_home"].values[m])
    elo = bin_metrics(yv[m], df["p_elo_home"].values[m])
    n = int(m.sum())
    res = {"n_matched_games": n,
           "polymarket": poly,
           "tuned_elo_same_subset": elo,
           "poly_minus_elo_accuracy": round(poly["accuracy"] - elo["accuracy"], 4),
           "poly_minus_elo_logloss": round(poly["logloss"] - elo["logloss"], 4)}

    m5 = m & (np.abs(df["p_poly_home"].values - df["p_elo_home"].values) >= 0.05)
    if m5.sum() > 25:
        agree = ((df["p_poly_home"].values >= df["p_elo_home"].values).astype(int)[m5] == yv[m5]).mean()
        res["disagreement_5pp"] = {"n": int(m5.sum()), "poly_side_hit_rate": round(float(agree), 4)}
    d = res
    print("  Polymarket: n=%d acc=%.2f%% ll=%.4f | tuned Elo same subset: acc=%.2f%% ll=%.4f" % (
        n, 100 * poly["accuracy"], poly["logloss"], 100 * elo["accuracy"], elo["logloss"]))
    print("  Расхождения >=5пп: %s" % res.get("disagreement_5pp"))
    out["polymarket"] = d


def main():
    out = {}
    nhl_load_main(out)
    nfl_main(out)
    ncaab_main(out)
    tennis_main(out)
    polymarket_block(out)
    json.dump(out, open(os.path.join(OUT, "iter9.json"), "w"), indent=1, ensure_ascii=False)
    print("\nsaved outputs/iter9.json")


if __name__ == "__main__":
    main()
