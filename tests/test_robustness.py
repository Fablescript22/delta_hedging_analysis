"""Hardening tests: invalid inputs are rejected loudly, and extreme-but-legal inputs stay finite and sane."""
import numpy as np
import pytest

from src import bs
from src.gbm_sim import make_rng, simulate_and_hedge, simulate_gbm_paths
from src.hedge import gamma_pnl_estimate, hedge_pnl

P = np.array([[100.0, 101.0, 99.0, 102.0]])
T = 0.25
KW = dict(K=100.0, sigma=0.2, r=0.05, T=T)


# ------------------------------------------------------------------ invalid scalar / size inputs
@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf, -1e-9, 1.5, -0.1])
def test_cost_rate_rejected(bad):
    with pytest.raises(ValueError):
        hedge_pnl(P, cost_rate=bad, **KW)


@pytest.mark.parametrize("bad_r", [np.nan, np.inf, "x", None])
def test_rate_rejected(bad_r):
    with pytest.raises(ValueError):
        hedge_pnl(P, K=100.0, sigma=0.2, r=bad_r, T=T)


@pytest.mark.parametrize("bad", [0.0, -0.2, np.nan, np.inf])
def test_sigma_and_strike_rejected(bad):
    with pytest.raises(ValueError):
        hedge_pnl(P, K=100.0, sigma=bad, r=0.05, T=T)
    with pytest.raises(ValueError):
        hedge_pnl(P, K=bad, sigma=0.2, r=0.05, T=T)


def test_per_path_arrays_must_match_paths():
    with pytest.raises(ValueError):
        hedge_pnl(P, K=np.array([100.0, 100.0]), sigma=0.2, r=0.05, T=T)
    with pytest.raises(ValueError):
        hedge_pnl(P, K=100.0, sigma=np.array([0.2, 0.2]), r=0.05, T=T)


def test_bad_option_type_rejected():
    with pytest.raises(ValueError):
        hedge_pnl(P, option="straddle", **KW)


@pytest.mark.parametrize("bad", [0, -5, 2.5, np.nan, np.inf, True, "10", None])
def test_simulation_sizes_rejected(bad):
    rng = make_rng(900)
    with pytest.raises(ValueError):
        simulate_gbm_paths(bad, 5, 100.0, 0.05, 0.2, 1.0, rng)
    with pytest.raises(ValueError):
        simulate_gbm_paths(5, bad, 100.0, 0.05, 0.2, 1.0, rng)


@pytest.mark.parametrize("kw", [dict(S0=0.0), dict(S0=-1.0), dict(S0=np.nan), dict(T=0.0), dict(T=-1.0),
                                dict(T=np.inf), dict(sigma_r=-0.1), dict(sigma_r=np.nan), dict(mu=np.inf)])
def test_simulator_parameters_rejected(kw):
    args = dict(n_paths=3, n_steps=4, S0=100.0, mu=0.05, sigma_r=0.2, T=1.0)
    args.update(kw)
    with pytest.raises(ValueError):
        simulate_gbm_paths(rng=make_rng(901), **args)


def test_simulate_and_hedge_rejects_bad_cost_and_chunk():
    for kw in (dict(cost_rate=-0.01), dict(cost_rate=2.0), dict(cost_rate=np.nan), dict(max_cells=0),
               dict(sigma_i=0.0), dict(K=-5.0)):
        with pytest.raises(ValueError):
            simulate_and_hedge(10, 5, sigma_r=0.2, rng=make_rng(902), **kw)


def test_gamma_estimate_rejects_bad_inputs():
    for kw in (dict(sigma=-0.1), dict(sigma=np.nan), dict(K=0.0), dict(r=np.nan), dict(realised_var=-0.01),
               dict(realised_var=np.nan)):
        args = dict(K=100.0, sigma=0.2, r=0.05)
        args.update(kw)
        rv = args.pop("realised_var", None)
        with pytest.raises(ValueError):
            gamma_pnl_estimate(P, T=T, realised_var=rv, **args)


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize("field", ["S", "K", "tau", "sigma", "r"])
def test_bs_rejects_nonfinite_inputs(bad, field):
    a = dict(S=100.0, K=100.0, tau=0.5, sigma=0.2, r=0.05)
    a[field] = bad
    for fn in (bs.call_price, bs.put_price, bs.delta, bs.gamma):
        with pytest.raises(ValueError):
            fn(**a)


# ------------------------------------------------------------------ extreme spot prices
@pytest.mark.parametrize("S", [1e-8, 1e-3, 1e3, 1e8, 1e12])
def test_extreme_spot_stays_finite_and_bounded(S):
    tau, sig, r, K = 0.5, 0.2, 0.05, 100.0
    c, p = bs.call_price(S, K, tau, sig, r), bs.put_price(S, K, tau, sig, r)
    d, g = bs.delta(S, K, tau, sig, r), bs.gamma(S, K, tau, sig, r)
    assert all(np.isfinite(v) for v in (c, p, d, g))
    assert c >= 0 and p >= 0 and 0.0 <= d <= 1.0 and g >= 0
    # put-call parity survives the extremes (relative to the scale of the numbers involved)
    assert abs(c - p - (S - K * np.exp(-r * tau))) <= 1e-9 * max(S, K)
    # deep ITM / OTM limits
    if S > 1e6:
        assert d == pytest.approx(1.0) and c == pytest.approx(S - K * np.exp(-r * tau), rel=1e-9)
    if S < 1e-3:
        assert d == pytest.approx(0.0, abs=1e-12) and c < 1e-12


@pytest.mark.parametrize("S", [1e-6, 1e6])
def test_hedging_with_extreme_price_jump_is_finite(S):
    path = np.array([[100.0, S, 100.0, S]])
    res = hedge_pnl(path, K=100.0, sigma=0.2, r=0.05, T=0.25, cost_rate=0.001)
    assert np.all(np.isfinite(res.pnl)) and np.all(np.isfinite(res.total_cost))
    assert np.isfinite(gamma_pnl_estimate(path, 100.0, 0.2, 0.05, 0.25)).all()


# ------------------------------------------------------------------ near-expiry options
@pytest.mark.parametrize("tau", [1e-12, 1e-9, 1e-6, 1e-4])
@pytest.mark.parametrize("S", [50.0, 99.999, 100.0, 100.001, 150.0])
def test_near_expiry_is_finite_and_converges_to_payoff(tau, S):
    K, sig, r = 100.0, 0.2, 0.05
    c, d, g = bs.call_price(S, K, tau, sig, r), bs.delta(S, K, tau, sig, r), bs.gamma(S, K, tau, sig, r)
    assert np.isfinite([c, d, g]).all() and 0.0 <= d <= 1.0 and g >= 0
    assert c == pytest.approx(max(S - K, 0.0), abs=0.2 * np.sqrt(tau) * S + 1e-9)  # price -> intrinsic like sqrt(tau)


def test_expiry_and_beyond_conventions():
    assert bs.call_price(105.0, 100.0, 0.0, 0.2, 0.05) == 5.0
    assert bs.delta(105.0, 100.0, 0.0, 0.2, 0.05) == 1.0
    assert bs.delta(95.0, 100.0, -1.0, 0.2, 0.05) == 0.0
    assert bs.gamma(100.0, 100.0, 0.0, 0.2, 0.05) == 0.0


def test_very_short_hedge_horizon_runs():
    paths = make_rng(903).lognormal(0, 1e-4, (50, 3)).cumprod(axis=1) * 100
    paths[:, 0] = 100.0
    res = hedge_pnl(paths, K=100.0, sigma=0.2, r=0.05, T=1e-6)
    assert np.all(np.isfinite(res.pnl))
    assert np.abs(res.pnl).max() < 1e-2           # almost no time and almost no movement: tiny PnL


# ------------------------------------------------------------------ unusual time grids that ARE legal
def test_nonuniform_grid_matches_manual_loop():
    times = np.array([0.0, 0.01, 0.05, 0.2, 0.25])
    path = np.array([[100.0, 102.0, 98.0, 101.0, 97.0]])
    K, sig, r, T = 100.0, 0.25, 0.04, 0.25
    res = hedge_pnl(path, K, sig, r, T, times=times)
    # independent scalar loop
    from math import erf, exp, log, sqrt
    N = lambda x: 0.5 * (1 + erf(x / sqrt(2)))
    def d1(S, tau): return (log(S / K) + (r + 0.5 * sig**2) * tau) / (sig * sqrt(tau))
    prem = path[0, 0] * N(d1(100.0, T)) - K * exp(-r * T) * N(d1(100.0, T) - sig * sqrt(T))
    cash, shares = prem, 0.0
    for i in range(4):
        new = N(d1(path[0, i], T - times[i]))
        cash -= (new - shares) * path[0, i]
        shares = new
        cash *= exp(r * (times[i + 1] - times[i]))
    pnl = cash + shares * path[0, -1] - max(path[0, -1] - K, 0.0)
    assert res.pnl[0] == pytest.approx(pnl, abs=1e-9)


def test_single_step_grid():
    res = hedge_pnl(np.array([[100.0, 110.0]]), K=100.0, sigma=0.2, r=0.0, T=1.0)
    assert np.isfinite(res.pnl).all() and res.n_rebalances == 1
