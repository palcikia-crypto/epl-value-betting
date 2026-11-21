"""Стейкинг: flat 1% банка, Kelly (full/half/quarter) с капом на ставку."""


def kelly_fraction(prob, odds):
    b = odds - 1.0
    if b <= 0:
        return 0.0
    return max(0.0, (prob * b - (1.0 - prob)) / b)


def run_backtest(stakes_and_bets, scheme, initial_bank=10_000.0,
                 kelly_mult=0.25, max_stake=0.05):
    """stakes_and_bets: [(won(bool), odds(float), prob(float))] в хронологии ставок."""
    bank = initial_bank
    peak = initial_bank
    max_dd = 0.0
    staked = 0.0
    profit = 0.0
    wins = 0
    n = 0
    curve = [initial_bank]
    amounts = []
    for won, odds, prob in stakes_and_bets:
        won = bool(won)
        if scheme == "flat":
            stake = bank * 0.01
        elif scheme.startswith("kelly"):
            f = kelly_fraction(prob, odds)
            stake = bank * min(f * kelly_mult, max_stake)
        else:
            raise ValueError(scheme)
        if stake <= 0:
            amounts.append(0.0)
            continue
        pnl = stake * (odds - 1.0) if won else -stake
        bank += pnl
        staked += stake
        profit += pnl
        wins += int(won)
        n += 1
        peak = max(peak, bank)
        max_dd = max(max_dd, (peak - bank) / peak)
        curve.append(bank)
        amounts.append(stake)
    return {
        "n_bets": n,
        "roi": (bank - initial_bank) / initial_bank,
        "yield_pct": profit / staked if staked > 0 else 0.0,
        "win_rate": wins / n if n else 0.0,
        "max_drawdown": max_dd,
        "final_bank": bank,
        "curve": curve,
        "avg_stake": (sum(amounts) / n) if n else 0.0,
    }
