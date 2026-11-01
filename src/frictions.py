"""Part B market-friction scenarios on the NIFTY index backtest.

!! These are SCENARIO ASSUMPTIONS, not measurements. !!
The supplied data are daily index levels and India VIX: there are no futures prices, option quotes,
bid-ask spreads, fee schedules or contract specifications in them.  This module therefore asks "how fast
does the frictionless result erode if hedging costs X?" on a round-number grid; it does not claim to know
what a real NIFTY desk paid.  Replace the grid with sourced values once contract specifications and quote data
are available (see README, 'Next steps').

Modelled frictions (all defaults leave the baseline untouched):
  * one-way proportional hedging cost c (bps of traded notional) = half bid-ask spread + fees + slippage lumped
    together; charged on every hedge trade including the initial purchase and the liquidation at maturity;
  * option half-spread h (volatility points): the seller receives the BID, i.e. sells at VIX - h instead of VIX,
    while the hedge ratio is still computed at VIX;
  * trading-size granularity g: hedge ratios rounded to multiples of g (stylised whole-lot constraint);
  * rebalancing interval k trading days.
Hedging instrument = the index itself (futures basis, carry and roll are ignored: no futures data supplied).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import real_data as rd
from .hedge import hedge_pnl

COST_BPS = (0.0, 1.0, 2.0, 5.0, 10.0)
OPT_HALF_SPREAD_VOL = (0.0, 0.5, 1.0)
REBAL_EVERY = (1, 2, 5, 10)
DELTA_STEPS = (0.01, 0.05, 0.10)


def friction_pnl(ws: rd.WindowSet, *, r: float = cfg.RISK_FREE_RATE_B, cost_bps: float = 0.0,
                 opt_half_spread_vol: float = 0.0, rebalance_every: int = 1, delta_step: float | None = None):
    """Hedging PnL (fraction of S0) per window and mean turnover (sum of |delta changes|, in option notionals)."""
    if cost_bps < 0 or opt_half_spread_vol < 0:
        raise ValueError("cost_bps and opt_half_spread_vol must be non-negative")
    W = ws.window
    T = W / cfg.TRADING_DAYS_PER_YEAR
    times = np.arange(W + 1) / cfg.TRADING_DAYS_PER_YEAR
    S0 = ws.paths[:, 0]
    sold_vol = ws.sigma_implied - opt_half_spread_vol / 100.0
    if np.any(sold_vol <= 0):
        raise ValueError("option half-spread leaves a non-positive selling volatility")
    sigma_hedge = np.repeat(ws.sigma_implied[:, None], W, axis=1)        # delta still computed at the VIX
    res = hedge_pnl(ws.paths, S0, sold_vol, r, T, times=times, rebalance_idx=range(0, W, rebalance_every),
                    sigma_hedge=sigma_hedge, cost_rate=cost_bps / 1e4, delta_step=delta_step, return_deltas=True)
    d = res.deltas
    turnover = np.abs(np.diff(np.concatenate([np.zeros((len(d), 1)), d, d[:, -1:] * 0], axis=1), axis=1)).sum(1)
    return res.pnl / S0, turnover, res.total_cost / S0


def friction_grid(ws: rd.WindowSet) -> pd.DataFrame:
    combos = [(k, c, h, None) for k in REBAL_EVERY for c in COST_BPS for h in OPT_HALF_SPREAD_VOL]
    combos += [(1, c, 0.0, g) for g in DELTA_STEPS for c in (0.0, 2.0, 5.0)]
    rows = []
    for k, c, h, g in combos:
        pnl, turn, cost = friction_pnl(ws, cost_bps=c, opt_half_spread_vol=h, rebalance_every=k, delta_step=g)
        p = pnl * 100
        rows.append(dict(rebalance_every=k, one_way_cost_bps=c, option_half_spread_vol_pts=h,
                         delta_step=np.nan if g is None else g, mean_pnl_pct=p.mean(), std_pnl_pct=p.std(ddof=1),
                         frac_profit=float((p > 0).mean()), p05_pnl_pct=float(np.percentile(p, 5)),
                         mean_turnover=float(turn.mean()), mean_cost_pct=float(cost.mean() * 100)))
    return pd.DataFrame(rows)


def breakeven_cost_bps(grid: pd.DataFrame, rebalance_every: int, opt_half_spread: float = 0.0) -> float:
    """One-way cost (bps) at which mean PnL falls to zero, from a straight-line fit of mean PnL on cost.

    Costs enter the PnL linearly (they are a sum of |trade| x rate), so the line is exact up to compounding.
    """
    g = grid[(grid.rebalance_every == rebalance_every) & (grid.option_half_spread_vol_pts == opt_half_spread)
             & grid.delta_step.isna()]
    slope, icpt = np.polyfit(g.one_way_cost_bps, g.mean_pnl_pct, 1)
    return float(-icpt / slope) if slope < 0 else float("inf")


def summarise(grid: pd.DataFrame) -> dict:
    base = grid[(grid.rebalance_every == 1) & (grid.one_way_cost_bps == 0) & (grid.option_half_spread_vol_pts == 0)
                & grid.delta_step.isna()].iloc[0]
    out = dict(baseline_mean_pnl_pct=float(base.mean_pnl_pct),
               breakeven_cost_bps={f"every_{k}d": breakeven_cost_bps(grid, k) for k in REBAL_EVERY},
               breakeven_cost_bps_with_half_vol_spread={f"every_{k}d": breakeven_cost_bps(grid, k, 0.5) for k in REBAL_EVERY})
    # best interval at each cost level (option spread 0)
    best = {}
    for c in COST_BPS:
        g = grid[(grid.one_way_cost_bps == c) & (grid.option_half_spread_vol_pts == 0) & grid.delta_step.isna()]
        row = g.loc[g.mean_pnl_pct.idxmax()]
        best[f"{c:g}bps"] = dict(best_interval_for_mean=int(row.rebalance_every), mean_pnl_pct=float(row.mean_pnl_pct))
    out["best_interval_by_cost"] = best
    out["half_spread_1vol_loss_pct"] = float(base.mean_pnl_pct - grid[
        (grid.rebalance_every == 1) & (grid.one_way_cost_bps == 0) & (grid.option_half_spread_vol_pts == 1.0)
        & grid.delta_step.isna()].iloc[0].mean_pnl_pct)
    return out
