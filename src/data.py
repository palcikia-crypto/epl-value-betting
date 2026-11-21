"""Загрузка и очистка данных football-data.co.uk (EPL, код E0)."""
import urllib.request
import os

import pandas as pd

BASE_URL = "https://www.football-data.co.uk/mmz4281/{season}/E0.csv"
SEASONS = ["2021", "2122", "2223", "2324", "2425"]
SEASON_LABELS = {
    "2021": "2020/21",
    "2122": "2021/22",
    "2223": "2022/23",
    "2324": "2023/24",
    "2425": "2024/25",
}

KEEP_COLS = [
    "Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR",
    "B365H", "B365D", "B365A",
    "PSH", "PSD", "PSA",
    "B365CH", "B365CD", "B365CA",
    "PSCH", "PSCD", "PSCA",
]

LEAKAGE_COLS = {"FTHG", "FTAG", "FTR", "HS", "AS", "HST", "AST", "HF", "AF",
                "HC", "AC", "HY", "AY", "HR", "AR"}


def download(season: str, out_dir: str = "data") -> str:
    out = os.path.join(out_dir, f"E0_{season}.csv")
    if not os.path.exists(out):
        urllib.request.urlretrieve(BASE_URL.format(season=season), out)
    return out


def load_season(season: str, out_dir: str = "data") -> pd.DataFrame:
    path = download(season, out_dir)
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]
    df = df[KEEP_COLS].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
    if df["Date"].isna().all():
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
    df["FTHG"] = df["FTHG"].astype(int)
    df["FTAG"] = df["FTAG"].astype(int)
    df["FTR"] = df["FTR"].str.strip().map({"H": 0, "D": 1, "A": 2})
    df["Season"] = SEASON_LABELS[season]
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def load_all(out_dir: str = "data") -> pd.DataFrame:
    parts = [load_season(s, out_dir) for s in SEASONS]
    df = pd.concat(parts, ignore_index=True)
    df = df.sort_values("Date").reset_index(drop=True)
    return df
