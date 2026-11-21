"""Tennis: загрузка tennis-data.co.uk ATP, Elo (de Graves), leak-free фичи."""
import os
import io
import zipfile
from collections import deque

import numpy as np
import pandas as pd

BASE_TPL = "http://www.tennis-data.co.uk/hrjk-85HytOjkhth76j_ygh4jf7/{y}/{y}.xlsx"
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/120.0"}


def download_year(year: int, out_dir: str = "data/tennis") -> str:
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"atp_{year}.xlsx")
    if not os.path.exists(out):
        req = urllib_request(BASE_TPL.format(y=year))
        open(out, "wb").write(req)
    return out


def urllib_request(url):
    import urllib.request
    req = urllib.request.Request(url, headers=UA)
    return urllib.request.urlopen(req, timeout=60).read()


def load_year(year: int, out_dir: str = "data/tennis") -> pd.DataFrame:
    df = pd.read_excel(download_year(year, out_dir))
    df.columns = [c.strip() for c in df.columns]
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    keep = df.dropna(subset=["Date", "Winner", "Loser", "PSW", "PSL"]).copy()
    keep["WRank"] = pd.to_numeric(keep["WRank"], errors="coerce")
    keep["LRank"] = pd.to_numeric(keep["LRank"], errors="coerce")
    keep["Comment"] = keep.get("Comment", "").astype(str) if "Comment" in keep else ""
    return keep.sort_values("Date").reset_index(drop=True)


def load_years(years, out_dir="data/tennis") -> pd.DataFrame:
    parts = [load_year(y, out_dir) for y in years]
    df = pd.concat(parts, ignore_index=True).sort_values("Date").reset_index(drop=True)
    return df


class EloState:
    """Elo ATP (FiveThirtyEight): K = 250/(m+5)^0.4, отдельные Elo по покрытию."""
    def __init__(self, base=1500.0, k=250.0):
        self.base = base
        self.k = k
        self.elo = {}
        self.elo_surf = {}
        self.matches = {}
        self.matches_surf = {}

    def get(self, p, surf=None):
        if p not in self.elo:
            self.elo[p] = self.base
            self.matches[p] = 0
        if surf is not None:
            key = (p, surf)
            if key not in self.elo_surf:
                self.elo_surf[key] = self.base
                self.matches_surf[key] = 0
        return self.elo[p]

    def k_factor(self, p, surf=None):
        m = self.matches[p] if surf is None else self.matches_surf.get((p, surf), 0)
        mm = self.matches[p] if surf is not None else m
        return self.k / ((m + 5) ** 0.4)

    def prob(self, p1, p2, surf=None):
        e1 = self.get(p1)
        e2 = self.get(p2)
        if surf is not None:
            e1 = 0.7 * e1 + 0.3 * self.get(p1, surf)
            e2 = 0.7 * e2 + 0.3 * self.get(p2, surf)
        return 1.0 / (1.0 + 10.0 ** ((e2 - e1) / 400.0))

    def update(self, winner, loser, surf=None, best_of=3):
        e_w = self.get(winner)
        e_l = self.get(loser)
        exp_w = 1.0 / (1.0 + 10.0 ** ((e_l - e_w) / 400.0))
        kw = self.k_factor(winner)
        kl = self.k_factor(loser)
        self.elo[winner] = e_w + kw * (1.0 - exp_w)
        self.elo[loser] = e_l - kl * (1.0 - exp_w)
        self.matches[winner] = self.matches.get(winner, 0) + 1
        self.matches[loser] = self.matches.get(loser, 0) + 1
        if surf is not None:
            ew_s = self.get(winner, surf)
            el_s = self.get(loser, surf)
            exp_ws = 1.0 / (1.0 + 10.0 ** ((el_s - ew_s) / 400.0))
            kw_s = self.k_factor(winner, surf)
            kl_s = self.k_factor(loser, surf)
            self.elo_surf[(winner, surf)] = ew_s + kw_s * (1.0 - exp_ws)
            self.elo_surf[(loser, surf)] = el_s - kl_s * (1.0 - exp_ws)
            self.matches_surf[(winner, surf)] = self.matches_surf.get((winner, surf), 0) + 1
            self.matches_surf[(loser, surf)] = self.matches_surf.get((loser, surf), 0) + 1


def build_tennis_features(df: pd.DataFrame) -> pd.DataFrame:
    """Хронологический проход: Elo до матча, форма (win% 20), H2H. Без утечки."""
    st = EloState()
    form = {}
    h2h = {}
    rows = []
    for _, m in df.iterrows():
        w, l, surf = m["Winner"], m["Loser"], m.get("Surface")
        p_w = st.prob(w, l, surf)
        f_w = np.mean(form[w]) if (w in form and form[w]) else 0.5
        f_l = np.mean(form[l]) if (l in form and form[l]) else 0.5
        key = (min(w, l), max(w, l))
        wA = key == (w, l)
        a_wins, b_wins = h2h.get(key, (0, 0))
        rows.append({
            "Date": m["Date"],
            "year": m["Date"].year,
            "player1": key[0], "player2": key[1],
            "elo1": st.get(key[0]), "elo2": st.get(key[1]),
            "elo_s1": st.get(key[0], surf), "elo_s2": st.get(key[1], surf),
            "rank1": m["WRank"] if wA else m["LRank"],
            "rank2": m["LRank"] if wA else m["WRank"],
            "form1": f_w if wA else f_l,
            "form2": f_l if wA else f_w,
            "h2h1": a_wins, "h2h2": b_wins,
            "best_of": m.get("Best of", 3),
            "surface": surf,
            "odds_p1": float(m["PSW"] if wA else m["PSL"]),
            "odds_p2": float(m["PSL"] if wA else m["PSW"]),
            "player1_wins": 1 if wA else 0,
        })
        if w not in form:
            form[w] = deque(maxlen=20)
        if l not in form:
            form[l] = deque(maxlen=20)
        form[w].append(1)
        form[l].append(0)
        h2h[key] = (a_wins + 1, b_wins) if wA else (a_wins, b_wins + 1)
        st.update(w, l, surf, m.get("Best of", 3))
    out = pd.DataFrame(rows)
    for c in ("elo1", "elo2", "elo_s1", "elo_s2", "rank1", "rank2"):
        out[c] = pd.to_numeric(out[c], errors="coerce")
    out["elo_diff"] = out["elo1"] - out["elo2"]
    out["elo_sdiff"] = out["elo_s1"] - out["elo_s2"]
    out["log_rank_diff"] = np.log(out["rank1"].clip(lower=1)) - np.log(out["rank2"].clip(lower=1))
    out["form_diff"] = out["form1"] - out["form2"]
    out["h2h_diff"] = out["h2h1"] - out["h2h2"]
    out["implied1"] = 1.0 / out["odds_p1"]
    return out


TENNIS_FEATURES = ["elo_diff", "elo_sdiff", "log_rank_diff", "form_diff", "h2h_diff", "implied1"]
