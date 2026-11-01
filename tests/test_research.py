"""Tests for the extended empirical research (src/research.py) and friction scenarios (src/frictions.py)."""
import math

import numpy as np
import pandas as pd
import pytest

from src import backtest as bk
from src import frictions as fr
from src import real_data as rd
from src import research as rs
from src.gbm_sim import make_rng
from src.hedge import hedge_pnl


@pytest.fixture(scope="module")
def real():
    data = rd.load_all()
    ws = rd.build_windows(data)
    return data, ws, bk.run_backtest(ws)


# ------------------------------------------------------------------ effect-size intervals
def test_effect_size_intervals_structure(real):
    _, _, bt = real
    df = rs.effect_size_cis(bt, 20, make_rng)
    assert len(df) == 6 and df.effect.is_unique
    for _, r in df.iterrows():
        assert r.boot_ci_lo < r.boot_ci_hi
        if not np.isnan(r.hac_se):
            assert r.hac_ci_lo < r.estimate < r.hac_ci_hi
    mean_row = df[df.effect == "mean hedging PnL"].iloc[0]
    assert mean_row.estimate == pytest.approx(bt.pnl.mean() * 100)
    assert mean_row.boot_ci_lo > 0                       # the mean is positive with 95% confidence under block bootstrap


def test_block_bootstrap_ci_covers_known_mean():
    rng = np.random.default_rng(5)
    x = rng.normal(1.0, 1.0, 600)
    lo, hi = rs.block_bootstrap_ci(lambda a: a.mean(), [x], 20, 800, np.random.default_rng(6))
    assert lo < 1.0 < hi and (hi - lo) == pytest.approx(2 * 1.96 / math.sqrt(600), rel=0.3)


def test_block_bootstrap_is_reproducible():
    x = np.random.default_rng(1).normal(size=300)
    a = rs.block_bootstrap_ci(np.mean, [x], 10, 200, make_rng(14, 0))
    b = rs.block_bootstrap_ci(np.mean, [x], 10, 200, make_rng(14, 0))
    assert a == b


# ------------------------------------------------------------------ market states
def test_states_partition_windows_and_match_independent_rule(real):
    data, ws, bt = real
    lab = rs.regime_labels(data, ws)
    assert set(lab) <= {"calm", "drawdown", "recovery", ""}
    assert (lab == "").sum() == rs.MOMENTUM_DAYS            # only the first 63 windows lack a trailing quarter
    close = data.nifty["Close"].to_numpy()
    dates = list(data.nifty.index)
    for i in np.random.default_rng(0).choice(len(ws.t0), 60, replace=False):
        p = dates.index(ws.t0[i])
        if p < rs.MOMENTUM_DAYS:
            assert lab[i] == ""
            continue
        dd = close[p] / close[: p + 1].max() - 1
        mom = close[p] / close[p - rs.MOMENTUM_DAYS] - 1
        expect = "calm" if dd > rs.DRAWDOWN_CUT else ("drawdown" if mom < 0 else "recovery")
        assert lab[i] == expect


def test_state_labels_use_only_past_information(real):
    """Truncating the data after t0 must not change the label of a window that starts at t0."""
    data, ws, _ = real
    lab = rs.regime_labels(data, ws)
    cut = ws.t0[1500]
    close = data.nifty["Close"].loc[:cut]
    dd = close.iloc[-1] / close.max() - 1
    mom = close.iloc[-1] / close.iloc[-1 - rs.MOMENTUM_DAYS] - 1
    expect = "calm" if dd > rs.DRAWDOWN_CUT else ("drawdown" if mom < 0 else "recovery")
    assert lab[1500] == expect


def test_state_table_counts_add_up(real):
    data, ws, bt = real
    lab = rs.regime_labels(data, ws)
    tbl = rs.regime_table(bt, lab, 20)
    assert tbl.windows.sum() == (lab != "").sum()
    assert tbl.share.sum() == pytest.approx(1.0)


# ------------------------------------------------------------------ volatility estimators
def _synthetic_data(sigma=0.2, days=420, sub=300, seed=3):
    rng = np.random.default_rng(seed)
    dt = 1 / 252 / sub
    steps = sigma * math.sqrt(dt) * rng.standard_normal((days, sub))
    logp = 7.0 + np.cumsum(steps.ravel()).reshape(days, sub)
    o = np.exp(logp[:, 0])                                   # open = first intraday tick
    h, l, c = np.exp(logp.max(1)), np.exp(logp.min(1)), np.exp(logp[:, -1])
    idx = pd.bdate_range("2020-01-01", periods=days)
    nifty = pd.DataFrame(dict(Open=o, High=np.maximum(h, o), Low=np.minimum(l, o), Close=c), index=idx)
    aligned = pd.DataFrame(dict(nifty_close=nifty.Close, vix_pct=20.0, sigma_implied=0.2), index=idx)
    return rd.RealData(nifty=nifty, vix=pd.DataFrame(dict(Close=20.0), index=idx), aligned=aligned)


def test_estimators_recover_true_volatility_on_synthetic_data():
    data = _synthetic_data()
    ws = rd.build_windows(data)
    est = rs.vol_estimators(data, ws)
    assert set(est) == set(rs.ESTIMATORS)
    for name, v in est.items():
        # the overnight gap is zero here (open = first tick, tiny), so range estimators are comparable
        assert v.mean() == pytest.approx(0.2, rel=0.07), name
    # centring only removes the (tiny) sample-mean term, so the two estimators must move together
    assert np.corrcoef(est["close_to_close"], est["close_to_close_centred"])[0, 1] > 0.95


def test_range_estimators_are_below_close_to_close_when_overnight_gaps_exist(real):
    data, ws, _ = real
    est = rs.vol_estimators(data, ws)
    for k in ("parkinson", "garman_klass", "rogers_satchell"):
        assert est[k].mean() < est["close_to_close"].mean()      # they omit close-to-open jumps


def test_estimator_comparison_table(real):
    data, ws, bt = real
    df, s = rs.estimator_comparison(data, ws, bt, 20)
    assert list(df.estimator) == list(rs.ESTIMATORS)
    base = df.iloc[0]
    assert base.r2 == pytest.approx(bk.regression_block(bt)["r2"]) and base.slope == pytest.approx(bk.regression_block(bt)["slope"])
    assert s["vol_gap_range_points"] == [pytest.approx(df.mean_vol_gap_points.min()), pytest.approx(df.mean_vol_gap_points.max())]
    assert (df.corr_with_close_to_close <= 1.0 + 1e-12).all() and (df.corr_with_close_to_close > 0.9).all()


# ------------------------------------------------------------------ chronological split
def test_split_periods_are_disjoint_ordered_and_purged(real):
    _, ws, _ = real
    for name, (cal, val, tst) in rs.SPLITS.items():
        m = [rs._period_mask(ws, *p) for p in (cal, val, tst)]
        assert not (m[0] & m[1]).any() and not (m[1] & m[2]).any() and not (m[0] & m[2]).any()
        for mask, (lo, hi) in zip(m, (cal, val, tst)):
            assert (ws.t0[mask] >= pd.Timestamp(lo)).all() and (ws.t_end[mask] <= pd.Timestamp(hi)).all()
        assert ws.t_end[m[0]].max() < ws.t0[m[1]].min() and ws.t_end[m[1]].max() < ws.t0[m[2]].min()   # no leakage


def test_split_models_fit_only_on_calibration(real):
    data, ws, bt = real
    est = rs.vol_estimators(data, ws)
    df, chosen = rs.chronological_split(ws, bt, est)
    cal = rs._period_mask(ws, *rs.SPLITS["primary"][0])
    x = ws.sigma_implied**2 - est["close_to_close"] ** 2
    beta = np.polyfit(x[cal], bt.pnl[cal], 1)
    row = df[(df.split == "primary") & (df.model == "gap_close_to_close") & (df.role == "test")].iloc[0]
    assert row.slope_fitted == pytest.approx(beta[0], rel=1e-9)
    const = df[(df.model == "constant") & (df.role != "calibration")]
    assert np.allclose(const.oos_r2, 0.0)                                  # the benchmark scores zero by construction
    ex_ante = df[df.model.isin(rs.EX_ANTE_MODELS)]
    assert not ex_ante.uses_realised_path.any() and df[~df.model.isin(rs.EX_ANTE_MODELS)].uses_realised_path.all()
    assert chosen["primary"]["selected_ex_ante"] in rs.EX_ANTE_MODELS


# ------------------------------------------------------------------ frictions
def test_zero_frictions_reproduce_baseline(real):
    _, ws, bt = real
    pnl, _, cost = fr.friction_pnl(ws)
    assert np.allclose(pnl, bt.pnl, atol=1e-12) and np.all(cost == 0)


def test_cost_is_linear_and_equals_turnover_times_rate(real):
    _, ws, _ = real
    p0, turn, _ = fr.friction_pnl(ws, cost_bps=0.0)
    p1, _, c1 = fr.friction_pnl(ws, cost_bps=2.0)
    p2, _, c2 = fr.friction_pnl(ws, cost_bps=4.0)
    assert np.allclose(c2, 2 * c1, rtol=1e-12)
    assert np.allclose(p0 - p1, c1, atol=1e-12)
    # cost per window ~ turnover x rate x (avg spot / S0), compounding at 6%/yr over a month is < 0.5%
    ratio = c1.mean() / (turn.mean() * 2e-4)
    assert 0.97 < ratio < 1.03


def test_pnl_falls_with_each_friction(real):
    _, ws, _ = real
    base = fr.friction_pnl(ws)[0].mean()
    assert fr.friction_pnl(ws, cost_bps=5)[0].mean() < base
    assert fr.friction_pnl(ws, opt_half_spread_vol=1.0)[0].mean() < base - 0.0005   # selling cheaper loses vega x h
    c = [fr.friction_pnl(ws, cost_bps=b)[0].mean() for b in (0, 1, 2, 5, 10)]
    assert all(a > b for a, b in zip(c, c[1:]))


def test_lot_rounding_produces_multiples_of_step(real):
    _, ws, _ = real
    res = hedge_pnl(ws.paths[:50], ws.paths[:50, 0], ws.sigma_implied[:50], 0.06, 21 / 252, delta_step=0.05,
                    return_deltas=True)
    q = res.deltas / 0.05
    assert np.allclose(q, np.round(q))
    same = hedge_pnl(ws.paths[:50], ws.paths[:50, 0], ws.sigma_implied[:50], 0.06, 21 / 252, delta_step=None)
    plain = hedge_pnl(ws.paths[:50], ws.paths[:50, 0], ws.sigma_implied[:50], 0.06, 21 / 252)
    assert np.array_equal(same.pnl, plain.pnl)


@pytest.mark.parametrize("bad", [0.0, -0.1, 2.0, np.nan, np.inf])
def test_bad_delta_step_rejected(bad):
    with pytest.raises(ValueError):
        hedge_pnl(np.array([[100.0, 101.0, 100.0]]), 100.0, 0.2, 0.05, 0.1, delta_step=bad)


def test_friction_input_validation(real):
    _, ws, _ = real
    with pytest.raises(ValueError):
        fr.friction_pnl(ws, cost_bps=-1.0)
    with pytest.raises(ValueError):
        fr.friction_pnl(ws, opt_half_spread_vol=-0.5)
    with pytest.raises(ValueError):
        fr.friction_pnl(ws, opt_half_spread_vol=50.0)          # would push selling volatility below zero


def test_breakeven_cost_is_where_mean_pnl_hits_zero(real):
    _, ws, _ = real
    grid = fr.friction_grid(ws)
    be = fr.breakeven_cost_bps(grid, 1)
    assert fr.friction_pnl(ws, cost_bps=be)[0].mean() == pytest.approx(0.0, abs=5e-5)


def test_rebalance_every_validated(real):
    _, ws, _ = real
    for bad in (0, -1, 22, 2.5, True):
        with pytest.raises(ValueError):
            bk.run_backtest(ws, rebalance_every=bad)


def test_block_bootstrap_ci_validates_inputs():
    x = np.arange(30.0)
    rng = np.random.default_rng(0)
    bad_calls = [
        dict(stat=np.mean, arrays=[], block=5, n_boot=10),
        dict(stat=np.mean, arrays=[x, x[:-1]], block=5, n_boot=10),
        dict(stat=np.mean, arrays=[np.where(x == 2, np.nan, x)], block=5, n_boot=10),
        dict(stat=np.mean, arrays=[x], block=0, n_boot=10),
        dict(stat=np.mean, arrays=[x], block=31, n_boot=10),
        dict(stat=np.mean, arrays=[x], block=5, n_boot=0),
        dict(stat=np.mean, arrays=[x], block=5, n_boot=10, level=1.0),
        dict(stat="mean", arrays=[x], block=5, n_boot=10),
    ]
    for kw in bad_calls:
        with pytest.raises(ValueError):
            rs.block_bootstrap_ci(rng=rng, **kw)


# ------------------------------------------------------------------ gap-inclusive estimators (review: do not treat estimators as interchangeable)
def test_every_estimator_has_a_type_and_the_three_families_are_present():
    assert set(rs.ESTIMATOR_TYPE) == set(rs.ESTIMATORS)
    assert set(rs.ESTIMATOR_TYPE.values()) == {"close-to-close", "range, intraday only (no overnight gap)", "range + overnight gap"}


def test_adding_the_overnight_gap_closes_most_of_the_range_estimator_gap(real):
    data, ws, bt = real
    df, s = rs.estimator_comparison(data, ws, bt, 20)
    base = df.set_index("estimator").mean_vol_gap_points
    for k in ("parkinson", "garman_klass", "rogers_satchell"):
        assert base[k] > base["close_to_close"] + 2.0                     # intraday-only: premium far larger
        assert abs(base[f"{k}_plus_gap"] - base["close_to_close"]) < 0.6   # gap added back: back to close-to-close territory
    assert abs(base["yang_zhang"] - base["close_to_close"]) < 0.6
    assert s["vrp_range_range_no_gap"][0] > s["vrp_range_close_to_close"][1] + 2.0
    assert s["vrp_range_range_with_gap"][1] < s["vrp_range_close_to_close"][1] + 0.6


def test_gap_inclusive_estimator_formula_on_a_small_example():
    """plus_gap variance = intraday variance + annualised mean squared overnight log move (by hand)."""
    data = _synthetic_data(days=60)
    ws = rd.build_windows(data)
    est = rs.vol_estimators(data, ws)
    o = rs.window_ohlc(data, ws)
    g = np.log(o["open"] / ws.paths[:, :-1])
    want = np.sqrt(est["parkinson"] ** 2 + 252 * (g ** 2).mean(1))
    assert np.allclose(est["parkinson_plus_gap"], want)


# ------------------------------------------------------------------ ex-ante track
def test_exante_features_use_only_information_up_to_t0(real):
    data, ws, _ = real
    f = rs.exante_features(data, ws)
    i = 1200
    cut = data.nifty.loc[: ws.t0[i]]
    lr = np.diff(np.log(cut["Close"].to_numpy()[-(ws.window + 1):]))
    manual = math.sqrt(252 / ws.window * (lr ** 2).sum())
    assert f["trailing_rv"][i] == pytest.approx(manual)
    assert f["exante_vol_gap"][i] == pytest.approx(ws.sigma_implied[i] - manual)
    assert f["exante_var_gap"][i] == pytest.approx(ws.sigma_implied[i] ** 2 - manual ** 2)


def test_exante_features_nan_only_without_history(real):
    data, ws, _ = real
    f = rs.exante_features(data, ws)["trailing_rv"]
    pos = pd.Series(np.arange(len(data.nifty)), index=data.nifty.index).loc[ws.t0].to_numpy()
    assert np.array_equal(np.isnan(f), pos < ws.window)
    assert np.isnan(f).sum() < 25


def test_split_tracks_are_labelled_and_exante_models_have_no_realised_path(real):
    data, ws, bt = real
    df, chosen = rs.chronological_split(ws, bt, rs.vol_estimators(data, ws), rs.exante_features(data, ws))
    assert set(df[df.track == "ex-ante"].model) == set(rs.EX_ANTE_MODELS)
    assert set(df[df.track == "ex-post"].model) == {"gap_close_to_close", "gap_bipower", "gap_parkinson", "gamma_estimate"}
    assert ((df.track == "ex-ante") == ~df.uses_realised_path).all()
    for s_ in rs.SPLITS:
        assert chosen[s_]["selected_ex_ante"] in rs.EX_ANTE_MODELS


def test_exante_models_do_not_beat_the_benchmark_by_much_out_of_sample(real):
    """The substantive finding, pinned: nothing known at t0 predicts PnL out of sample (R^2 within a few points of 0)."""
    data, ws, bt = real
    df, _ = rs.chronological_split(ws, bt, rs.vol_estimators(data, ws), rs.exante_features(data, ws))
    test = df[(df.role == "test") & (df.track == "ex-ante")]
    assert test.oos_r2.max() < 0.05
    post = df[(df.role == "test") & (df.model == "gamma_estimate")]
    assert post.oos_r2.min() > 0.8                                      # the ex-post mechanism explains almost everything


def test_exante_models_skip_windows_without_trailing_history(real):
    data, ws, bt = real
    ex = rs.exante_features(data, ws)
    df, _ = rs.chronological_split(ws, bt, rs.vol_estimators(data, ws), ex)
    n_nan = int(np.isnan(ex["exante_vol_gap"]).sum())
    cal = df[(df.split == "primary") & (df.role == "calibration")].set_index("model").windows
    assert cal["constant"] - cal["exante_vol_gap"] == n_nan


# ------------------------------------------------------------------ bootstrap Monte Carlo error
def test_mc_error_table_structure_and_determinism(real):
    _, _, bt = real
    t1 = rs.effect_size_mc_error(bt, 20, make_rng, reps=2)
    t2 = rs.effect_size_mc_error(bt, 20, make_rng, reps=2)
    pd.testing.assert_frame_equal(t1, t2)
    assert len(t1) == len(rs.effect_size_cis(bt, 20, make_rng))
    assert (t1.lo_sd >= 0).all() and (t1.hi_sd >= 0).all() and (t1.lo_range >= t1.lo_sd).all()
    assert (t1.max_bound_sd_over_width < 0.1).all()                    # MC error is small relative to the interval
    for bad in (1, 0, 2.5, True):
        with pytest.raises(ValueError):
            rs.effect_size_mc_error(bt, 20, make_rng, reps=bad)
