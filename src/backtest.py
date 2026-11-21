"""Бэктест value betting: Тимinator фикс-ставка 1% банка, Pinnacle closing odds."""
import numpy as np
import pandas as pd

INITIAL_BANK = 10_000.0
STAKE_FRAC = 0.01


def backtest(df_eval: pd.DataFrame, bets: list, initial_bank: float = INITIAL_BANK,
             stake_frac: float = STAKE_FRAC, one_per_match: bool = True) -> dict:
    """bets: [(match_idx, outcome, odds, prob, edge)], df_eval содержит FTR."""
    # одна ставка на матч: максимум edge
    by_match = {}
    for b in bets:
        i = b[0]
        if i not in by_match or b[4] > by_match[i][4]:
            by_match[i] = b
    ordered = sorted(by_match.values(), key=lambda b: (b[0], ))
    if not one_per_match:
        ordered = sorted(bets, key=lambda b: (b[0], -b[4]))

    bank = initial_bank
    peak = initial_bank
    max_dd = 0.0
    staked = 0.0
    profit = 0.0
    wins = 0
    curve = [initial_bank]
    records = []
    for i, outcome, odds, prob, edge in ordered:
        stake = bank * stake_frac
        ftr = int(df_eval.iloc[i]["FTR"])
        won = (ftr == outcome)
        pnl = stake * (odds - 1.0) if won else -stake
        bank += pnl
        staked += stake
        profit += pnl
        wins += int(won)
        peak = max(peak, bank)
        dd = (peak - bank) / peak
        max_dd = max(max_dd, dd)
        curve.append(bank)
        records.append({
            "date": df_eval.iloc[i]["Date"],
            "home": df_eval.iloc[i]["HomeTeam"],
            "away": df_eval.iloc[i]["AwayTeam"],
            "bet": ["H", "D", "A"][outcome],
            "odds": odds, "prob": prob, "edge": edge,
            "ftr": ["H", "D", "A"][ftr], "won": bool(won),
            "stake": stake, "pnl": pnl, "bank": bank,
        })
    n = len(ordered)
    return {
        "n_bets": n,
        "roi": (bank - initial_bank) / initial_bank,
        "yield_pct": profit / staked if staked > 0 else 0.0,
        "win_rate": wins / n if n else 0.0,
        "max_drawdown": max_dd,
        "final_bank": bank,
        "curve": curve,
        "records": records,
    }


def baseline_favorite(df_eval: pd.DataFrame, odds: np.ndarray,
                      initial_bank: float = INITIAL_BANK,
                      stake_frac: float = STAKE_FRAC) -> dict:
    """Всегда ставить 1% банка на фаворита рынка (минимальный коэффициент Pinnacle)."""
    bets = []
    for i in range(len(df_eval)):
        o = odds[i]
        if not np.isfinite(o).all():
            continue
        fav = int(np.argmin(o))
        bets.append((i, fav, float(o[fav]), float(1.0 / o[fav]), 0.0))
    # передаём как value-ставки (prob захардкожен как implied), но модель не используется
    return backtest(df_eval, bets, initial_bank, stake_frac)
