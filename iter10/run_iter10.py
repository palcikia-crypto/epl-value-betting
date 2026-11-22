# -*- coding: utf-8 -*-
"""
Итерация 10: монетизация против Polymarket + нагрузка и возраст на уровне игроков.

1) Селтл B2B / B2B+OT сигналов против Polymarket-цен (кэш iter9: gamma + CLOB,
   цена за ~10 мин до вброса). Polymarket здесь не ставка сама по себе, а
   бесплатная цена для честного селтла правил 2026/27, пока у SBR нет линий.
2) shift-TOI-нагрузка: /shifts в новом NHL API не существует (404), поэтому
   нагрузка строится по boxscore (playerByGameStats, TOI каждого игрока).
   Метрики команды: общий TOI форвардов, доля минут топ-8 форвардов и
   топ-4 защитников. Смотрим: влияет ли нагрузка ПРЕДЫДУЩЕЙ игры на
   следующий матч и усиливает ли она B2B.
3) Возраст/выносливость: /player/{id}/landing (дата рождения) -> средний
   возраст состава, взвешенный по TOI. Разница возрастов и взаимодействие
   возраст x B2B ( маркируется как ИССЛЕДОВАТЕЛЬСКАЯ гипотеза: n мал ).

Запуск:
  python3 iter10/run_iter10.py                    # селтл по кэшу + анализ кэшей
  python3 iter10/run_iter10.py --fetch-boxscores  # докачать boxscore 25/26+26/27
  python3 iter10/run_iter10.py --fetch-ages       # докачать landing игроков
Кэши: outputs/boxscore/{id}.json, outputs/player_landing/{id}.json
"""
import argparse
import json
import math
import os
import shutil
import sys
import time
import urllib.request

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

from iter7.run_iter7 import load_nhl_ext, two_prop_z


# ---------------- utils ----------------
def fetch_json(url, cachepath, sleep=0.6):
    if os.path.exists(cachepath):
        try:
            return json.load(open(cachepath))
        except Exception:
            pass
    req = urllib.request.Request(url, headers={"User-Agent": "epl-value-betting research"})
    with urllib.request.urlopen(req, timeout=30) as r:
        j = json.loads(r.read().decode("utf-8"))
    os.makedirs(os.path.dirname(cachepath), exist_ok=True)
    with open(cachepath, "w") as f:
        json.dump(j, f)
    time.sleep(sleep)
    return j


def t_stat(profits):
    a = np.asarray(profits, dtype=float)
    n = len(a)
    if n < 5:
        return 0.0, 1.0
    sd = a.std(ddof=1)
    t = a.mean() / (sd / math.sqrt(n)) if sd > 0 else 0.0
    pv = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))
    return round(t, 3), round(pv, 5)


def flat_sim(bets):
    """bets: [(decimal_odds, win)]; flat 1 юнита на честной (девигованной) цене."""
    prof = [o - 1 if w else -1.0 for o, w in bets]
    n = len(prof)
    st = {"n": n, "roi": round(sum(prof) / n, 4) if n else None,
          "yield": round(sum(prof) / n, 4) if n else None}
    if n >= 5:
        st["yield_t"], st["p_value"] = t_stat(prof)
    bank, peak, dd = 0.0, 0.0, 0.0
    for p in prof:
        bank += p
        peak = max(peak, bank)
        dd = max(dd, peak - bank)
    st["max_dd"] = round(dd, 2)
    return st


# ---------------- rest / OT flags on full chain ----------------
def nhl_from_score_caches():
    rows = []
    for dname in ("nhl_cache_ext", "nhl_cache"):
        dpath = os.path.join(OUT, dname)
        if not os.path.isdir(dpath):
            continue
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
                rows.append({"id": g.get("id"), "date": g["gameDate"],
                             "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"],
                             "hg": g["homeTeam"].get("score"), "ag": g["awayTeam"].get("score"),
                             "period": g.get("period", 3)})
    return pd.DataFrame(rows)


def add_rest_ot(df):
    df = df.sort_values("date").reset_index(drop=True)
    last_date = {}
    last_ot = {}
    flags = []
    for _, g in df.iterrows():
        d = pd.Timestamp(g["date"])
        rows_ = {}
        for side, t in (("home", g["home"]), ("away", g["away"])):
            prev = last_date.get(t)
            rows_["rest_%s" % side] = min(int((d - prev).days), 7) if prev is not None else 8
            rows_["ot_last_%s" % side] = bool(last_ot.get(t, False))
        if int(g.get("period") or 3) >= 4:
            last_ot[g["home"]] = True
            last_ot[g["away"]] = True
        else:
            last_ot[g["home"]] = False
            last_ot[g["away"]] = False
        for side, t in (("home", g["home"]), ("away", g["away"])):
            last_date[t] = d
        flags.append(rows_)
    return pd.concat([df, pd.DataFrame(flags)], axis=1)


def load_chain():
    frames = []
    ext = load_nhl_ext()
    if ext is not None and len(ext):
        frames.append(ext)
    for fname in ("nhl_games.csv", "nhl_games_2627.csv"):
        p = os.path.join(OUT, fname)
        if os.path.exists(p):
            d = pd.read_csv(p)
            if "gameType" in d.columns:
                d = d[d["gameType"] == 2]
            frames.append(d)
    score = nhl_from_score_caches()
    if len(score):
        frames.append(score)
    df = pd.concat(frames, ignore_index=True)
    if "gameType" in df.columns:
        df = df[df["gameType"] == 2]
    df = df.drop_duplicates(subset=["date", "home", "away"]).sort_values("date").reset_index(drop=True)
    if "period" not in df.columns:
        df["period"] = 3
    return add_rest_ot(df)


# ---------------- Polymarket load (схема iter9) ----------------
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


def poly_frame():
    mdir = os.path.join(OUT, "polymarket_nhl")
    meta = json.load(open(os.path.join(mdir, "meta.json")))
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
            j = json.load(open(os.path.join(mdir, "%s.json" % g["cid"])))
            h0, h1 = j["hist0"], j["hist1"]
        except Exception:
            continue
        if not h0 or not h1:
            continue
        cut = int(t0.timestamp()) - 600
        p0s = [h["p"] for h in h0 if h["t"] <= cut]
        p1s = [h["p"] for h in h1 if h["t"] <= cut]
        if not p0s or not p1s:
            continue
        p0, p1 = p0s[-1], p1s[-1]
        if p0 <= 0.01 or p1 <= 0.01:
            continue
        abbr = [TEAM2ABBREV.get(o.strip()) for o in outs]
        if None in abbr:
            continue
        for ii in by_pair.get(frozenset(abbr), []):
            row = games.loc[ii]
            if abs((pd.Timestamp(t0.date()) - pd.Timestamp(row["date"])).days) > 1:
                continue
            home_tok = 0 if abbr[0] == row["home"] else 1
            first = p0 if home_tok == 0 else p1
            recs.append({"date": row["date"], "home": row["home"], "away": row["away"],
                         "hg": row["hg"], "ag": row["ag"],
                         "p_poly_home": first / (first + (p1 if home_tok == 0 else p0)),
                         "start": start})
            break
    return pd.DataFrame(recs).drop_duplicates(subset=["date", "home", "away"]) \
        .sort_values("date").reset_index(drop=True)


# ---------------- Block 1: settle B2B signals vs Polymarket ----------------
def settle_group(df, mask, side):
    m = np.asarray(mask, dtype=bool)
    n = int(m.sum())
    if n < 10:
        return {"n": n}
    ph = df["p_poly_home"].values[m]
    dec_odds_home = 1.0 / ph
    dec_odds_away = 1.0 / (1.0 - ph)
    bets = list(zip(dec_odds_home if side == "home" else dec_odds_away,
                    df["y"].values[m]))
    st = flat_sim(bets)
    st["actual_home_win"] = round(float(df["y"].values[m].mean()), 4)
    st["expected_home_win"] = round(float(ph.mean()), 4)
    return st


def settle_block(out):
    print("=== 1. Селтл B2B / B2B+OT против Polymarket-цен ===")
    chain = load_chain()
    poly = poly_frame()
    print("  Polymarket matched (кэш iter9): %d игр" % len(poly))
    lookup = chain.set_index(["date", "home", "away"])
    rest_a, rest_h, ot_a, ot_h, ys = [], [], [], [], []
    for _, g in poly.iterrows():
        try:
            row = lookup.loc[(g["date"], g["home"], g["away"])]
        except KeyError:
            row = None
        if row is not None and isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if row is None:
            rest_a.append(np.nan); rest_h.append(np.nan)
            ot_a.append(False); ot_h.append(False); ys.append(np.nan)
            continue
        rest_a.append(row["rest_away"]); rest_h.append(row["rest_home"])
        ot_a.append(bool(row["ot_last_away"])); ot_h.append(bool(row["ot_last_home"]))
        ys.append(int(row["hg"] > row["ag"]))
    poly["rest_away"] = rest_a; poly["rest_home"] = rest_h
    poly["ot_last_away"] = ot_a; poly["ot_last_home"] = ot_h
    poly["y"] = ys
    df = poly.dropna(subset=["y"]).reset_index(drop=True)
    res = {
        "away_b2b_home": settle_group(df, (df["rest_away"] <= 1) & (df["rest_home"] >= 2), "home"),
        "away_b2b_ot_home": settle_group(df, (df["rest_away"] <= 1) & (df["rest_home"] >= 2) & df["ot_last_away"], "home"),
        "away_b2b_no_ot_home": settle_group(df, (df["rest_away"] <= 1) & (df["rest_home"] >= 2) & ~df["ot_last_away"], "home"),
        "home_b2b_away": settle_group(df, (df["rest_home"] <= 1) & (df["rest_away"] >= 2), "away"),
        "both_fresh_control": settle_group(df, (df["rest_away"] >= 2) & (df["rest_home"] >= 2), "home"),
    }
    for k, v in res.items():
        print("  %-24s %s" % (k, v))
    out["polymarket_settle"] = res


# ---------------- Block 2: boxscore TOI load ----------------
def toi_minutes(p):
    for key in ("toiSEC", "toiSec", "toiSeconds"):
        v = p.get(key)
        if isinstance(v, (int, float)) and v > 0:
            return v / 60.0
    for key in ("toi", "toiceiling", "timeOnIce"):
        v = p.get(key)
        if isinstance(v, str) and ":" in v:
            try:
                mm, ss = v.split(":")
                return float(mm) + float(ss) / 60.0
            except Exception:
                pass
    return None


def boxscore_iter():
    dpath = os.path.join(OUT, "boxscore")
    if not os.path.isdir(dpath):
        return
    for f in sorted(os.listdir(dpath)):
        if not f.endswith(".json"):
            continue
        try:
            yield int(f.split(".")[0]), json.load(open(os.path.join(dpath, f)))
        except Exception:
            continue


def fetch_boxscores():
    df = nhl_from_score_caches()
    df = df.drop_duplicates("id")
    df = df[df["hg"].notna()]
    bdir = os.path.join(OUT, "boxscore")
    have = {int(f.split(".")[0]) for f in os.listdir(bdir)} if os.path.isdir(bdir) else set()
    todo = [int(i) for i in df["id"] if int(i) not in have]
    print("  boxscore fetch: %d игр к скачиванию" % len(todo))
    for i in todo:
        try:
            fetch_json("https://api-web.nhle.com/gamecenter/%d/boxscore" % i,
                       os.path.join(bdir, "%d.json" % i))
        except Exception as e:
            print("    %d: %s" % (i, e))


def team_load(j, side):
    try:
        players = j["playerByGameStats"][side]["players"]
    except Exception:
        return None
    fw, dfn = [], []
    for p in players:
        t = toi_minutes(p)
        if t is None:
            continue
        pos = str(p.get("positionCode") or p.get("position") or "").upper()
        if pos.startswith("D") or pos == "D":
            dfn.append(t)
        else:
            fw.append(t)
    if len(fw) < 8 or len(dfn) < 4:
        return None
    fw.sort(reverse=True); dfn.sort(reverse=True)
    tot_f, tot_d = sum(fw), sum(dfn)
    return {"f_min": round(tot_f, 1),
            "top8f_share": round(sum(fw[:8]) / tot_f, 4),
            "top4d_share": round(sum(dfn[:4]) / tot_d, 4)}


def toi_block(out, fetch=False):
    print("=== 2. shift-TOI: нагрузка по boxscore (прокси закрытого /shifts) ===")
    if fetch:
        fetch_boxscores()
    rows = []
    prev_top8f = {}
    prev_date = {}
    for gid, j in boxscore_iter():
        try:
            d = j.get("gameDate")
            home, away = j.get("homeTeam", {}).get("abbrev"), j.get("awayTeam", {}).get("abbrev")
            hs = j.get("homeTeamScore", j.get("homeTeam", {}).get("score"))
            as_ = j.get("awayTeamScore", j.get("awayTeam", {}).get("score"))
            lh, la = team_load(j, "homeTeam"), team_load(j, "awayTeam")
            if not d or not home or hs is None or lh is None or la is None:
                continue
            rec = {"gid": gid, "date": d, "home": home, "away": away, "hg": hs, "ag": as_,
                   "home_top8f": lh["top8f_share"], "away_top8f": la["top8f_share"],
                   "home_top4d": lh["top4d_share"], "away_top4d": la["top4d_share"]}
            for side, t in (("home", home), ("away", away)):
                rec["b2b_%s" % side] = int(bool(prev_date.get(t)) and
                                           (pd.Timestamp(d) - prev_date[t]).days <= 1)
                rec["prev_top8f_%s" % side] = prev_top8f.get(t)
                prev_top8f[t] = lh["top8f_share"] if side == "home" else la["top8f_share"]
                prev_date[t] = pd.Timestamp(d)
            rows.append(rec)
        except Exception:
            continue
    if not rows:
        print("  кэш boxscore пуст -- прогони с --fetch-boxscores")
        out["toi"] = {"error": "no boxscore cache"}
        return
    df = pd.DataFrame(rows).drop_duplicates("gid").sort_values("date").reset_index(drop=True)
    df.to_csv(os.path.join(OUT, "iter10_toi.csv"), index=False)
    yv = (df["hg"] > df["ag"]).astype(int)
    res = {"n_games_with_toi": int(len(df)),
           "source": "NHL boxscore API (playerByGameStats TOI)"}
    med = df["prev_top8f_away"].median()
    prev = df["prev_top8f_away"]
    b2b = df["b2b_away"] == 1
    for k, m in {
        "away_rested_prev_light": (~b2b) & (prev < med),
        "away_rested_prev_heavy": (~b2b) & (prev >= med),
        "b2b_prev_light": b2b & (prev < med),
        "b2b_prev_heavy": b2b & (prev >= med),
    }.items():
        n = int(m.sum())
        res[k] = {"n": n, "home_win": round(float(yv[m].mean()), 4) if n else None}
        print("  %-24s n=%4d home_win=%s" % (k, n, res[k]["home_win"]))
    h, l = res["b2b_prev_heavy"], res["b2b_prev_light"]
    if h.get("n", 0) > 10 and l.get("n", 0) > 10:
        z = two_prop_z(round(h["home_win"] * h["n"]), h["n"],
                       round(l["home_win"] * l["n"]), l["n"])
        res["z_b2b_heavy_vs_light"] = round(z, 3)
        print("  z(B2B после тяжёлого распределения TOI vs лёгкого)=%.2f -- ИССЛЕДОВАТЕЛЬСКАЯ гипотеза" % z)
    out["toi"] = res


# ---------------- Block 3: age ----------------
def age_from_birth(birth, game_date):
    try:
        return (pd.Timestamp(game_date) - pd.Timestamp(birth)).days / 365.25
    except Exception:
        return None


def fetch_ages():
    pids = set()
    for _, j in boxscore_iter():
        for side in ("homeTeam", "awayTeam"):
            try:
                for p in j["playerByGameStats"][side]["players"]:
                    if p.get("playerId"):
                        pids.add(int(p["playerId"]))
            except Exception:
                continue
    ldir = os.path.join(OUT, "player_landing")
    have = {int(f.split(".")[0]) for f in os.listdir(ldir)} if os.path.isdir(ldir) else set()
    todo = sorted(pids - have)
    print("  landing fetch: %d из %d игроков к скачиванию" % (len(todo), len(pids)))
    for pid in todo:
        try:
            fetch_json("https://api-web.nhle.com/player/%d/landing" % pid,
                       os.path.join(ldir, "%d.json" % pid))
        except Exception as e:
            print("    %d: %s" % (pid, e))


def age_block(out, fetch=False):
    print("=== 3. Возраст состава (TOI-вес), возраст x B2B ===")
    if fetch:
        fetch_ages()
    ldir = os.path.join(OUT, "player_landing")
    births = {}
    if os.path.isdir(ldir):
        for f in os.listdir(ldir):
            try:
                j = json.load(open(os.path.join(ldir, f)))
                if j.get("birthDate"):
                    births[int(f.split(".")[0])] = j["birthDate"]
            except Exception:
                continue
    rows = []
    prev_age = {}
    prev_date = {}
    for gid, j in boxscore_iter():
        try:
            d = j.get("gameDate")
            home, away = j.get("homeTeam", {}).get("abbrev"), j.get("awayTeam", {}).get("abbrev")
            hs = j.get("homeTeamScore", j.get("homeTeam", {}).get("score"))
            as_ = j.get("awayTeamScore", j.get("awayTeam", {}).get("score"))
            if not d or not home or hs is None:
                continue
            team_age = {}
            for side, t in (("homeTeam", "home"), ("awayTeam", "away")):
                wts = []
                for p in j["playerByGameStats"][side]["players"]:
                    m = toi_minutes(p)
                    pid = p.get("playerId")
                    if m and pid in births:
                        a = age_from_birth(births[pid], d)
                        if a:
                            wts.append((m, a))
                team_age[t] = round(sum(m * a for m, a in wts) / sum(m for m, _ in wts), 2) if wts else None
            if team_age["home"] is None or team_age["away"] is None:
                continue
            rec = {"gid": gid, "date": d, "home": home, "away": away, "hg": hs, "ag": as_,
                   "age_home": team_age["home"], "age_away": team_age["away"],
                   "age_diff": round(team_age["home"] - team_age["away"], 2)}
            for side, t in (("home", home), ("away", away)):
                rec["b2b_%s" % side] = int(bool(prev_date.get(t)) and
                                           (pd.Timestamp(d) - prev_date[t]).days <= 1)
                rec["prev_age_%s" % side] = prev_age.get(t)
                prev_age[t] = team_age[side]
                prev_date[t] = pd.Timestamp(d)
            rows.append(rec)
        except Exception:
            continue
    if not rows:
        print("  нет boxscore/landing-кэша -- прогони с --fetch-boxscores --fetch-ages")
        out["age"] = {"error": "no data"}
        return
    df = pd.DataFrame(rows).drop_duplicates("gid").sort_values("date").reset_index(drop=True)
    df.to_csv(os.path.join(OUT, "iter10_age.csv"), index=False)
    yv = (df["hg"] > df["ag"]).astype(int)
    res = {"n_games": int(len(df)), "players_with_age": len(births),
           "corr_age_diff_home_win": round(float(df["age_diff"].corr(yv)), 3)}
    print("  corr(diff TOI-возрастов, победа хозяина)=%.3f на %d играх" % (res["corr_age_diff_home_win"], len(df)))
    adv = df["age_diff"] >= 0.5
    res["home_older_0p5y"] = {"n": int(adv.sum()), "home_win": round(float(yv[adv].mean()), 4)}
    yng = df["age_diff"] <= -0.5
    res["home_younger_0p5y"] = {"n": int(yng.sum()), "home_win": round(float(yv[yng].mean()), 4)}
    b2b = df["b2b_away"] == 1
    med_a = df["age_away"].median()
    splits = {"away_b2b_old_squad": b2b & (df["age_away"] >= med_a),
              "away_b2b_young_squad": b2b & (df["age_away"] < med_a)}
    for k, m in splits.items():
        n = int(m.sum())
        res[k] = {"n": n, "home_win": round(float(yv[m].mean()), 4) if n else None}
        print("  %-22s n=%3d home_win=%s" % (k, n, res[k]["home_win"]))
    n1, n0 = res["away_b2b_old_squad"]["n"], res["away_b2b_young_squad"]["n"]
    if n1 > 10 and n0 > 10:
        z = two_prop_z(round(res["away_b2b_old_squad"]["home_win"] * n1), n1,
                       round(res["away_b2b_young_squad"]["home_win"] * n0), n0)
        res["z_b2b_age_split"] = round(z, 3)
        print("  z(старый гость на B2B vs молодой)=%.2f -- ИССЛЕДОВАТЕЛЬСКАЯ гипотеза, n мало" % z)
    out["age"] = res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch-boxscores", action="store_true")
    ap.add_argument("--fetch-ages", action="store_true")
    args = ap.parse_args()
    out = {}
    settle_block(out)
    toi_block(out, fetch=args.fetch_boxscores)
    age_block(out, fetch=args.fetch_ages)
    json.dump(out, open(os.path.join(OUT, "iter10.json"), "w"), indent=1, ensure_ascii=False)
    try:
        shutil.copy(os.path.join(OUT, "iter10.json"), os.path.join(ROOT, "dashboard", "iter10.json"))
    except Exception:
        pass
    print("\nsaved outputs/iter10.json (+ iter10_toi.csv, iter10_age.csv); dashboard: dashboard/iter10.html")


if __name__ == "__main__":
    main()
