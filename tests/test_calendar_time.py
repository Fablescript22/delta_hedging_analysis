"""Calendar-time (ACT/365) re-run of the Part B hedge."""
import numpy as np
import pandas as pd
import pytest

from src import backtest as bk
from src import bs
from src import calendar_time as ct
from src import config as cfg
from src import real_data as rd
from src.hedge import hedge_pnl


def _daily_calendar_data(days=80, seed=5):
    """Prices on EVERY calendar day, so the calendar clock is exactly uniform: 1/365 per step."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=days, freq="D")
    close = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal(days)))
    nifty = pd.DataFrame(dict(Open=close, High=close * 1.005, Low=close * 0.995, Close=close), index=idx)
    vix = pd.DataFrame(dict(Close=18.0 + rng.standard_normal(days)), index=idx)
    aligned = pd.DataFrame(dict(nifty_close=nifty.Close, vix_pct=vix.Close, sigma_implied=vix.Close / 100), index=idx)
    return rd.RealData(nifty=nifty, vix=vix, aligned=aligned)


@pytest.fixture(scope="module")
def real():
    data = rd.load_all()
    ws = rd.build_windows(data)
    return data, ws, bk.run_backtest(ws)


def test_day_offsets_match_window_calendar_span(real):
    data, ws, _ = real
    off = ct.window_day_offsets(data, ws)
    assert off.shape == (len(ws.t0), ws.window + 1)
    assert (off[:, 0] == 0).all() and (np.diff(off, axis=1) >= 1).all()
    assert np.array_equal(off[:, -1], ws.calendar_days)


def test_uniform_calendar_data_equals_the_engine_with_365_day_steps():
    data = _daily_calendar_data()
    ws = rd.build_windows(data)
    cal = ct.run_calendar_backtest(data, ws, r=0.06)
    W = ws.window
    ref = hedge_pnl(ws.paths, ws.paths[:, 0], ws.sigma_implied, 0.06, W / 365.0, times=np.arange(W + 1) / 365.0)
    assert np.allclose(cal.pnl, ref.pnl / ws.paths[:, 0])
    assert np.allclose(cal.premium, ref.premium / ws.paths[:, 0])
    # realised vol annualised on the same 365-day clock
    lr = np.diff(np.log(ws.paths), axis=1)
    assert np.allclose(cal.rv, np.sqrt((lr ** 2).sum(1) / (W / 365.0)))


def test_premium_uses_the_actual_horizon(real):
    data, ws, _ = real
    cal = ct.run_calendar_backtest(data, ws)
    i = 700
    T = ws.calendar_days[i] / 365.0
    S0 = ws.paths[i, 0]
    want = bs.call_price(S0, S0, T, ws.sigma_implied[i], ws.r[i]) / S0
    assert cal.premium[i] == pytest.approx(want)


def test_calendar_and_trading_clocks_agree_on_the_conclusions(real):
    data, ws, bt = real
    table, info = ct.calendar_time_comparison(data, ws, bt, 20)
    assert list(table.clock) == ["trading-day clock (baseline, T = 21/252)", "calendar clock (ACT/365, actual dates)"]
    base, cal = table.iloc[0], table.iloc[1]
    assert base.mean_pnl_pct == pytest.approx(bt.pnl.mean() * 100)
    assert base.slope == pytest.approx(bk.regression_block(bt)["slope"])
    # same sign and similar size: the clock is not what drives the result
    assert cal.mean_pnl_pct > 0 and cal.slope > 0 and cal.vrp_vol_points > 0
    assert abs(cal.mean_pnl_pct - base.mean_pnl_pct) < 0.1 and abs(cal.r2 - base.r2) < 0.1
    assert info["corr_pnl"] > 0.95 and 28 <= info["calendar_days_min"] <= info["calendar_days_max"] <= 38


def test_calendar_run_rejects_bad_rate(real):
    data, ws, _ = real
    with pytest.raises(ValueError):
        ct.run_calendar_backtest(data, ws, r=float("nan"))
