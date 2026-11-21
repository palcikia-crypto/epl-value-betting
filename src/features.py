"""Feature engineering строго без утечки: только информация, доступная до начала матча."""
import numpy as np
import pandas as pd

INITIAL_ELO = 1500.0
K_FACTOR = 20.0
HOME_ADV = 100.0


def compute_elo_features(df: pd.DataFrame) -> pd.DataFrame:
    """Elo до матча для обеих команд. Обновление после матча: K=20, home adv=100."""
    elo = {}
    elo_home, elo_away = [], []
    for _, row in df.iterrows():
        h, a = row["HomeTeam"], row["AwayTeam"]
        eh = elo.get(h, INITIAL_ELO)
        ea = elo.get(a, INITIAL_ELO)
        elo_home.append(eh)
        elo_away.append(ea)
        expected_h = 1.0 / (1.0 + 10.0 ** ((ea - (eh + HOME_ADV)) / 400.0))
        actual_h = {0: 1.0, 1: 0.5, 2: 0.0}[row["FTR"]]
        delta = K_FACTOR * (actual_h - expected_h)
        elo[h] = eh + delta
        elo[a] = ea - delta
    out = df.copy()
    out["elo_home"] = elo_home
    out["elo_away"] = elo_away
    out["elo_diff"] = out["elo_home"] - out["elo_away"]
    return out


def _team_events(df: pd.DataFrame) -> dict:
    """team -> хронологический список событий (date, season, gf, ga, gd, pts, win)."""
    h = pd.DataFrame({"team": df["HomeTeam"], "date": df["Date"], "season": df["Season"],
                      "gf": df["FTHG"].astype(float), "ga": df["FTAG"].astype(float)})
    a = pd.DataFrame({"team": df["AwayTeam"], "date": df["Date"], "season": df["Season"],
                      "gf": df["FTAG"].astype(float), "ga": df["FTHG"].astype(float)})
    ev = pd.concat([h, a], ignore_index=True)
    ev["gd"] = ev["gf"] - ev["ga"]
    ev["pts"] = np.sign(ev["gf"] - ev["ga"])
    ev["pts"] = ev["pts"].map({-1.0: 0.0, 0.0: 1.0, 1.0: 3.0})
    ev["win"] = (ev["pts"] == 3).astype(float)
    ev = ev.sort_values("date").reset_index(drop=True)
    return {t: g for t, g in ev.groupby("team", sort=False)}


def _prior_matches(g: pd.DataFrame, date, season, n):
    prior = g[g["date"] < date]
    tail_n = prior.tail(n)
    cur = prior[prior["season"] == season]
    return tail_n, cur


def compute_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Форма за N=3 и N=5 матчей + накопленные метрики сезона + H2H (последние 3)."""
    groups = _team_events(df)
    out = df.copy()
    n_rows = len(df)

    for n in (3, 5):
        for side in ("home", "away"):
            lang = "HomeTeam" if side == "home" else "AwayTeam"
            gf = np.zeros(n_rows); ga = np.zeros(n_rows)
            gd = np.zeros(n_rows); pts = np.zeros(n_rows)
            for i, val in enumerate(df[lang]):
                g = groups[val]
                tail, _ = _prior_matches(g, df["Date"].iloc[i], df["Season"].iloc[i], n)
                if len(tail):
                    gf[i] = tail["gf"].mean(); ga[i] = tail["ga"].mean()
                    gd[i] = tail["gd"].mean(); pts[i] = tail["pts"].mean()
            out[f"{side}_gf_{n}"] = gf
            out[f"{side}_ga_{n}"] = ga
            out[f"{side}_gd_{n}"] = gd
            out[f"{side}_pts_{n}"] = pts

    for side in ("home", "away"):
        lang = "HomeTeam" if side == "home" else "AwayTeam"
        sgf = np.zeros(n_rows); sga = np.zeros(n_rows); sgd = np.zeros(n_rows)
        spts = np.zeros(n_rows); swr = np.zeros(n_rows)
        for i in range(n_rows):
            val = df[lang].iloc[i]
            g = groups[val]
            _, cur = _prior_matches(g, df["Date"].iloc[i], df["Season"].iloc[i], 5)
            if len(cur):
                sgf[i] = cur["gf"].mean(); sga[i] = cur["ga"].mean()
                sgd[i] = cur["gd"].mean(); spts[i] = cur["pts"].mean()
                swr[i] = cur["win"].mean()
        out[f"{side}_season_gf"] = sgf
        out[f"{side}_season_ga"] = sga
        out[f"{side}_season_gd"] = sgd
        out[f"{side}_season_pts"] = spts
        out[f"{side}_season_wr"] = swr

    key_pairs = list(zip(df["HomeTeam"], df["AwayTeam"], df["Date"]))
    dates = df["Date"].values
    home_teams = df["HomeTeam"].values
    away_teams = df["AwayTeam"].values
    h2h_gd = np.zeros(n_rows); h2h_wr = np.zeros(n_rows); h2h_n = np.zeros(n_rows)
    for i in range(n_rows):
        ht, at, dt = home_teams[i], away_teams[i], dates[i]
        h2h_gd[i] = 0.0; h2h_wr[i] = 0.0
        past_idx = [j for j in range(n_rows)
                    if dates[j] < dt and
                    ((home_teams[j] == ht and away_teams[j] == at) or
                     (home_teams[j] == at and away_teams[j] == ht))]
        past_idx = past_idx[-3:]
        h2h_n[i] = len(past_idx)
        for j in past_idx:
            team_is_home = (home_teams[j] == ht)
            gd_team = (df["FTHG"].iloc[j] - df["FTAG"].iloc[j]) if team_is_home \
                else (df["FTAG"].iloc[j] - df["FTHG"].iloc[j])
            h2h_gd[i] += gd_team
            if gd_team > 0:
                h2h_wr[i] += 1
        if past_idx:
            h2h_gd[i] /= len(past_idx)
            h2h_wr[i] /= len(past_idx)
    out["h2h_gd"] = h2h_gd
    out["h2h_wr"] = h2h_wr
    out["h2h_n"] = h2h_n
    return out


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = compute_elo_features(df)
    df = compute_form_features(df)
    return df


FEATURE_COLS = [
    "elo_home", "elo_away", "elo_diff",
    "home_gf_3", "home_ga_3", "home_gd_3", "home_pts_3",
    "away_gf_3", "away_ga_3", "away_gd_3", "away_pts_3",
    "home_gf_5", "home_ga_5", "home_gd_5", "home_pts_5",
    "away_gf_5", "away_ga_5", "away_gd_5", "away_pts_5",
    "home_season_gf", "home_season_ga", "home_season_gd", "home_season_pts", "home_season_wr",
    "away_season_gf", "away_season_ga", "away_season_gd", "away_season_pts", "away_season_wr",
    "h2h_gd", "h2h_wr", "h2h_n",
]
