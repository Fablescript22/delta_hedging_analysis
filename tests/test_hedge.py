import numpy as np
import pytest

from src import bs
from src.gbm_sim import make_rng, simulate_and_hedge, simulate_gbm_paths
from src.hedge import gamma_pnl_estimate, hedge_pnl


def _paths(m=200, n=50, sigma=0.2, mu=0.05, seed=1):
    return simulate_gbm_paths(m, n, 100.0, mu, sigma, 1.0, make_rng(seed))


def test_zero_cost_means_zero_total_cost():
    res = hedge_pnl(_paths(), 100.0, 0.2, 0.05, 1.0)
    assert np.all(res.total_cost == 0.0)


def test_costs_reduce_pnl_by_exactly_total_cost():
    P = _paths()
    base = hedge_pnl(P, 100.0, 0.2, 0.05, 1.0)
    costly = hedge_pnl(P, 100.0, 0.2, 0.05, 1.0, cost_rate=0.001)
    assert np.all(costly.total_cost > 0)
    assert np.allclose(costly.pnl, base.pnl - costly.total_cost)


def test_rebalance_idx_subset_equals_subsampled_path():
    P = _paths(n=60)
    sub = hedge_pnl(P, 100.0, 0.2, 0.05, 1.0, rebalance_idx=range(0, 60, 3))
    direct = hedge_pnl(P[:, ::3], 100.0, 0.2, 0.05, 1.0)          # 20 equally spaced steps
    assert np.allclose(sub.pnl, direct.pnl)
    assert sub.n_rebalances == 20


def test_rebalance_idx_not_dividing_window_still_marks_to_maturity():
    P = _paths(n=21)
    res = hedge_pnl(P, 100.0, 0.2, 0.05, 1.0, rebalance_idx=range(0, 21, 5))   # 0,5,10,15,20
    assert res.n_rebalances == 5 and np.all(np.isfinite(res.pnl))


def test_error_shrinks_with_more_rebalances():
    stds = []
    for n in (5, 20, 80):
        out, _ = simulate_and_hedge(4000, n, sigma_r=0.2, rng=make_rng(3, n))
        stds.append(out["pnl"].std())
    assert stds[0] > stds[1] > stds[2]


def test_put_and_call_hedging_pnl_are_identical_path_by_path():
    # Put = Call - forward, and a forward is replicated exactly by holding one unit of the
    # underlying, so hedging a short put or a short call (same K, sigma) leaves the SAME error.
    # (First version of this test used an arbitrary "std < 5% of premium" cap that the put,
    # with its smaller premium, missed at N=500 - the engine was right, the cap was ill-chosen.)
    P = simulate_gbm_paths(2000, 500, 100.0, 0.05, 0.2, 1.0, make_rng(5))
    call = hedge_pnl(P, 100.0, 0.2, 0.05, 1.0, option="call")
    put = hedge_pnl(P, 100.0, 0.2, 0.05, 1.0, option="put")
    assert np.max(np.abs(call.pnl - put.pnl)) < 1e-9
    assert put.premium[0] < call.premium[0]


def test_per_path_sigma_and_strike_arrays():
    P = _paths(m=10)
    sig = np.linspace(0.15, 0.3, 10)
    K = P[:, 0]
    res = hedge_pnl(P, K, sig, 0.05, 1.0)
    one = hedge_pnl(P[3:4], K[3], sig[3], 0.05, 1.0)
    assert res.pnl[3] == pytest.approx(one.pnl[0])


def test_premium_matches_black_scholes():
    res = hedge_pnl(_paths(m=3), 100.0, 0.2, 0.05, 1.0)
    assert np.allclose(res.premium, bs.call_price(100.0, 100.0, 1.0, 0.2, 0.05))


def test_invalid_inputs():
    with pytest.raises(ValueError):
        hedge_pnl(np.array([[100.0, -1.0]]), 100.0, 0.2, 0.05, 1.0)
    with pytest.raises(ValueError):
        hedge_pnl(_paths(), 100.0, 0.2, 0.05, 1.0, rebalance_idx=[1, 2, 3])
    with pytest.raises(ValueError):
        hedge_pnl(_paths(n=10), 100.0, 0.2, 0.05, 2.0, times=np.linspace(0, 1, 11))


_P3 = np.array([[100.0, 101.0, 102.0]])


@pytest.mark.parametrize("bad_times, msg", [
    ([0.1, 0.55, 1.0], "start at zero"),            # nonzero first timestamp
    ([-0.2, 0.5, 1.0], "start at zero"),
    ([0.0, np.nan, 1.0], "finite"),
    ([0.0, 0.5, np.inf], "finite"),
    ([0.0, 0.5, 0.5], "strictly increasing"),
    ([0.0, 0.5], "one entry per path column"),
    ([0.0, 0.5, 0.9], "equal T"),
])
def test_time_grid_validation_rejects_bad_grids(bad_times, msg):
    t = np.array(bad_times, dtype=float)
    with pytest.raises(ValueError, match=msg):
        hedge_pnl(_P3, 100.0, 0.2, 0.05, 1.0, times=t)
    with pytest.raises(ValueError, match=msg):
        gamma_pnl_estimate(_P3, 100.0, 0.2, 0.05, 1.0, times=t)


@pytest.mark.parametrize("bad_T", [np.nan, np.inf, 0.0, -1.0])
def test_T_must_be_finite_and_positive(bad_T):
    with pytest.raises(ValueError, match="T must be finite and positive"):
        hedge_pnl(_P3, 100.0, 0.2, 0.05, bad_T, times=np.array([0.0, 0.5, 1.0]))


def test_valid_nonuniform_grid_from_zero_still_accepted():
    res = hedge_pnl(_P3, 100.0, 0.2, 0.05, 1.0, times=np.array([0.0, 0.3, 1.0]))
    assert np.isfinite(res.pnl).all() and res.premium[0] == pytest.approx(bs.price(100.0, 100.0, 1.0, 0.2, 0.05))


# ---------------------------------------------------------------- gamma_pnl_estimate: return conventions
def _hedged(n, sigma_r=0.25, m=3000, seed=11):
    P = simulate_gbm_paths(m, n, 100.0, 0.05, sigma_r, 1.0, make_rng(seed))
    return P, hedge_pnl(P, 100.0, 0.2, 0.05, 1.0).pnl


def test_return_convention_default_is_log_and_invalid_rejected():
    P, _ = _hedged(20, m=50)
    a = gamma_pnl_estimate(P, 100.0, 0.2, 0.05, 1.0)
    assert np.array_equal(a, gamma_pnl_estimate(P, 100.0, 0.2, 0.05, 1.0, return_convention="log"))
    with pytest.raises(ValueError, match="return_convention"):
        gamma_pnl_estimate(P, 100.0, 0.2, 0.05, 1.0, return_convention="bogus")


def test_demeaned_equals_log_when_mean_log_return_is_zero():
    # alternating +x / -x log returns have exactly zero mean, so demeaning changes nothing
    x = 0.01
    lr = np.array([x, -x, x, -x, x, -x])
    P = (100.0 * np.exp(np.concatenate([[0.0], np.cumsum(lr)])))[None, :]
    a = gamma_pnl_estimate(P, 100.0, 0.2, 0.05, 1.0, return_convention="log")
    b = gamma_pnl_estimate(P, 100.0, 0.2, 0.05, 1.0, return_convention="log_demeaned")
    assert a == pytest.approx(b, rel=1e-12, abs=1e-12)


def test_simple_return_convention_equals_direct_second_order_taylor_term():
    # w_i * (S_{i+1}/S_i - 1)^2  with  w_i = e^{r tau} Gamma S^2 / 2   ==   e^{r tau} (1/2) Gamma (dS)^2
    P, _ = _hedged(10, m=5)
    r, T, sig = 0.05, 1.0, 0.2
    n = P.shape[1] - 1
    t = np.linspace(0, T, n + 1)
    tau = T - t[:-1]
    dS = np.diff(P, axis=1)
    g = bs.gamma(P[:, :-1], 100.0, tau[None, :], sig, r)
    direct = (np.exp(r * tau)[None, :] * 0.5 * g * (sig**2 * P[:, :-1] ** 2 * np.diff(t)[None, :] - dS**2)).sum(axis=1)
    assert gamma_pnl_estimate(P, 100.0, sig, r, T, return_convention="simple") == pytest.approx(direct, rel=1e-10)


def test_approximation_error_shrinks_as_the_grid_is_refined():
    """The estimate is an approximation, not an identity: its RMSE against the actual PnL must fall with N."""
    rmse = {}
    for n in (10, 50, 250):
        P, pnl = _hedged(n)
        est = gamma_pnl_estimate(P, 100.0, 0.2, 0.05, 1.0)
        rmse[n] = float(np.sqrt(np.mean((est - pnl) ** 2)))
    assert rmse[10] > rmse[50] > rmse[250] > 0.0          # converges, but is never exactly zero
    assert rmse[250] < 0.2 * rmse[10]                     # at least a 5-fold reduction over a 25-fold finer grid


def test_log_and_simple_conventions_agree_on_a_fine_grid_and_both_track_pnl():
    P, pnl = _hedged(500)
    a = gamma_pnl_estimate(P, 100.0, 0.2, 0.05, 1.0, return_convention="log")
    b = gamma_pnl_estimate(P, 100.0, 0.2, 0.05, 1.0, return_convention="simple")
    assert np.sqrt(np.mean((a - b) ** 2)) < 0.01 * 10.45          # within 1% of the premium
    for est in (a, b):
        assert np.corrcoef(est, pnl)[0, 1] ** 2 > 0.99


@pytest.mark.parametrize("bad", [
    [0, 1.9, 2], [False, True, 2], [True, 2, 3], [0, 1, 2.5], [0, float("nan"), 2], [0, float("inf")],
    [], [0, 2, 1], [0, 1, 1], [1, 2], [0, 1, 50], ["0", "1"], [0, None],
])
def test_rebalance_idx_rejects_fractional_boolean_and_malformed(bad):
    """Regression: [0, 1.9, 2] used to become [0, 1, 2] and [False, True, 2] too, silently."""
    with pytest.raises(ValueError):
        hedge_pnl(_paths(n=50), 100.0, 0.2, 0.05, 1.0, rebalance_idx=bad)


def test_rebalance_idx_accepts_ints_numpy_ints_and_whole_floats():
    P = _paths(n=50)
    ref = hedge_pnl(P, 100.0, 0.2, 0.05, 1.0, rebalance_idx=[0, 5, 10])
    for ok in (np.array([0, 5, 10]), [np.int64(0), np.int32(5), 10], [0.0, 5.0, 10.0], (0, 5, 10)):
        assert np.allclose(hedge_pnl(P, 100.0, 0.2, 0.05, 1.0, rebalance_idx=ok).pnl, ref.pnl)


def test_rebalance_idx_non_iterable_rejected():
    with pytest.raises(ValueError):
        hedge_pnl(_paths(n=50), 100.0, 0.2, 0.05, 1.0, rebalance_idx=5)
