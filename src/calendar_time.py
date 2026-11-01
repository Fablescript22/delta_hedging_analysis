"""Part B robustness: the same backtest on a CALENDAR-time clock instead of the equally spaced trading-day clock.

Baseline (``backtest.run_backtest``): every window is treated as T = 21/252 year with 21 equal steps of 1/252.
But 21 trading days span 28 to 38 calendar days (weekends, holidays), and India VIX is itself an annualised
volatility quoted on calendar time to expiry.  This module re-runs the hedge with the real dates:

  * time stamps    t_i = (date_i - date_0).days / 365       (ACT/365)
  * option expiry  T   = (date_W - date_0).days / 365       (so the premium is priced for the actual horizon)
  * interest       accrues at r over the same calendar time (premium grows at e^{rT}, cash at e^{r dt_i})
  * realised vol   sigma_r^2 = (sum of squared daily log returns) / T,   i.e. annualised over calendar time
  * hedge dates    unchanged: the trading days; weekend/holiday moves are the Friday-to-Monday returns

Because the time stamps differ across windows, the hedge engine is called once per window (a few seconds in
total).  The comparison asks whether the headline conclusions (mean PnL, spread, slope on the variance gap,
R-squared, volatility risk premium) depend on the choice of clock.  Nothing here is tuned: ACT/365 is the
standard calendar day count (NSE documents India VIX as the expected volatility over the next 30 calendar days, with time to
expiry measured in minutes; the day-count basis is not stated in the pages consulted), and no other convention is tried.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as cfg
from . import real_data as rd
from .backtest import Backtest, mean_hac, ols_hac
from .hedge import gamma_pnl_estimate, hedge_pnl

DAYS_PER_YEAR = 365.0


def window_day_offsets(data: rd.RealData, ws: rd.WindowSet) -> np.ndarray:
    """(M, W+1) calendar days elapsed since each window's start, for every observation of the window."""
    dates = data.nifty.index
    pos = pd.Series(np.arange(len(dates)), index=dates)
    start = pos.loc[ws.t0].to_numpy()
    idx = start[:, None] + np.arange(ws.window + 1)[None, :]
    days = (dates.values.astype("datetime64[D]").astype(np.int64))[idx]
    return (days - days[:, :1]).astype(float)


def run_calendar_backtest(data: rd.RealData, ws: rd.WindowSet, r: float = cfg.RISK_FREE_RATE_B) -> Backtest:
    """Daily-rebalanced backtest on the ACT/365 clock (see module docstring)."""
    if not np.isfinite(r):
        raise ValueError("r must be finite")
    off = window_day_offsets(data, ws)
    if not np.array_equal(off[:, -1], ws.calendar_days.astype(float)):
        raise ValueError("calendar offsets inconsistent with WindowSet.calendar_days")
    M = len(ws.t0)
    tt = off / DAYS_PER_YEAR
    pnl, prem, est, wsum = (np.empty(M) for _ in range(4))
    for m in range(M):
        P = ws.paths[m:m + 1]
        S0, sig, T, times = P[:, 0], ws.sigma_implied[m:m + 1], float(tt[m, -1]), tt[m]
        res = hedge_pnl(P, S0, sig, r, T, times=times)
        pnl[m], prem[m] = res.pnl[0] / S0[0], res.premium[0] / S0[0]
        est[m] = gamma_pnl_estimate(P, S0, sig, r, T, times=times)[0] / S0[0]
        wsum[m] = gamma_pnl_estimate(P, S0, sig, r, T, times=times, realised_var=0.0)[0] / sig[0] ** 2 / S0[0]
    lr = np.diff(np.log(ws.paths), axis=1)
    rv = np.sqrt(np.square(lr).sum(axis=1) / tt[:, -1])
    return Backtest(ws=ws, r=r, pnl=pnl, premium=prem, rv=rv, gap=ws.sigma_implied ** 2 - rv ** 2, est=est,
                    slope_theory=wsum)


def _summary(label: str, bt: Backtest, lag: int) -> dict:
    p = bt.pnl * 100
    m, se = mean_hac(p, lag)
    reg = ols_hac(bt.gap, bt.pnl, lag)
    vol_gap = bt.ws.vix_pct - bt.rv * 100
    vg, vse = mean_hac(vol_gap, lag)
    return dict(clock=label, mean_pnl_pct=m, mean_pnl_hac_lo=m - 1.96 * se, mean_pnl_hac_hi=m + 1.96 * se,
                std_pnl_pct=float(p.std(ddof=1)), frac_profit=float((p > 0).mean()),
                mean_premium_pct=float(bt.premium.mean() * 100), mean_rv_pct=float(bt.rv.mean() * 100),
                vrp_vol_points=vg, vrp_hac_lo=vg - 1.96 * vse, vrp_hac_hi=vg + 1.96 * vse,
                slope=reg["slope"], slope_hac_lo=reg["slope"] - 1.96 * reg["se_slope_hac"],
                slope_hac_hi=reg["slope"] + 1.96 * reg["se_slope_hac"], r2=reg["r2"],
                pathwise_estimate_r2=float(np.corrcoef(bt.est, bt.pnl)[0, 1] ** 2))


def calendar_time_comparison(data: rd.RealData, ws: rd.WindowSet, bt: Backtest, lag: int) -> tuple[pd.DataFrame, dict]:
    """Baseline (trading-day clock) versus the ACT/365 calendar clock, with paired differences."""
    cal = run_calendar_backtest(data, ws, bt.r)
    table = pd.DataFrame([_summary("trading-day clock (baseline, T = 21/252)", bt, lag),
                          _summary("calendar clock (ACT/365, actual dates)", cal, lag)])
    d = (cal.pnl - bt.pnl) * 100
    dm, dse = mean_hac(d, lag)
    info = dict(
        calendar_days_min=int(ws.calendar_days.min()), calendar_days_max=int(ws.calendar_days.max()),
        calendar_days_mean=float(ws.calendar_days.mean()), mean_T_calendar=float(ws.calendar_days.mean() / DAYS_PER_YEAR),
        T_trading=float(ws.window / cfg.TRADING_DAYS_PER_YEAR),
        paired_mean_diff_pct=dm, paired_mean_diff_hac_lo=dm - 1.96 * dse, paired_mean_diff_hac_hi=dm + 1.96 * dse,
        corr_pnl=float(np.corrcoef(cal.pnl, bt.pnl)[0, 1]),
        max_abs_diff_pct=float(np.abs(d).max()),
        corr_rv=float(np.corrcoef(cal.rv, bt.rv)[0, 1]),
        mean_rv_ratio=float((cal.rv / bt.rv).mean()),
        slope_ratio=float(table.loc[1, "slope"] / table.loc[0, "slope"]),
    )
    return table, info
