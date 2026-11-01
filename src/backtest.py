"""Part B: daily delta-hedging backtest on NIFTY 50 with India VIX as the implied volatility.

For every start date t0 that has a valid India VIX close:
  * sell one ATM call (K = NIFTY close at t0) priced by Black-Scholes at sigma_i = VIX_t0 / 100;
  * delta-hedge with NIFTY daily closes, delta computed with sigma_i held fixed (baseline hedger);
  * 21 trading days later compute the hedging PnL, expressed as a fraction of S0.

Time convention: one trading day = 1/252 year, window T = 21/252 (so realised and implied variance are
both "per trading-year" quantities).  Realised volatility = sqrt( (252/21) * sum of squared daily log
returns ), the un-centred form.  It is the natural proxy for the quadratic variation that appears in the
continuous-time gamma-PnL expression (a sum of squared returns, with no mean subtracted), but with daily data it
is only an approximation to that quantity, not an exact match: the continuous-time result is an identity only
for a continuously observed path and continuous hedging.  ``estimator_conventions`` measures how much the fit to
the actual hedging PnL changes when the realised-variance proxy is defined differently.
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config as cfg
from . import rates
from . import real_data as rd
from .gbm_sim import make_rng, simulate_gbm_paths
from .hedge import gamma_pnl_estimate, hedge_pnl

CRASH_START, CRASH_END = pd.Timestamp("2020-02-20"), pd.Timestamp("2020-04-30")   # Covid crash, fixed a priori
CALM_VIX, TURBULENT_VIX = 15.0, 20.0                                              # regime cut-offs, fixed a priori
FREQ_INTERVALS = (1, 2, 5, 10)                                                    # rebalance every k trading days


# ----------------------------------------------------------------------------------------- input checks
def check_lag(lag, n: int, name: str = "lag") -> int:
    """A HAC lag must be an integer (not Boolean) with 0 <= lag < n."""
    if isinstance(lag, (bool, np.bool_)) or not isinstance(lag, (int, np.integer)):
        raise ValueError(f"{name} must be an integer")
    if not 0 <= lag < n:
        raise ValueError(f"{name} must satisfy 0 <= {name} < n (got {lag} with n={n})")
    return int(lag)


def check_series(name: str, a, min_n: int = 2) -> np.ndarray:
    """1-D, finite, float array with at least ``min_n`` observations."""
    arr = np.asarray(a, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size < min_n:
        raise ValueError(f"{name} needs at least {min_n} observations (got {arr.size})")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values")
    return arr


def check_xy(x, y, min_n: int = 3) -> tuple[np.ndarray, np.ndarray]:
    x, y = check_series("x", x, min_n), check_series("y", y, min_n)
    if x.size != y.size:
        raise ValueError(f"x and y must have the same length (got {x.size} and {y.size})")
    return x, y


def check_bootstrap(n: int, block, n_boot) -> None:
    for nm, v, hi in (("block", block, n), ("n_boot", n_boot, None)):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1 or (hi and v > hi):
            raise ValueError(f"{nm} must be an integer >= 1" + (f" and <= n ({n})" if hi else ""))


# ----------------------------------------------------------------------------------------- helpers
def realised_vol(paths: np.ndarray, trading_days: int = cfg.TRADING_DAYS_PER_YEAR, centred: bool = False) -> np.ndarray:
    """Annualised realised vol per window from daily log returns (un-centred RMS by default)."""
    lr = np.diff(np.log(paths), axis=1)
    n = lr.shape[1]
    if centred:
        return np.sqrt(trading_days * lr.var(axis=1, ddof=1))
    return np.sqrt(trading_days / n * np.square(lr).sum(axis=1))


def ols_hac(x: np.ndarray, y: np.ndarray, lag: int) -> dict:
    """OLS of y on [1, x] with plain and Newey-West (Bartlett kernel, ``lag``) standard errors."""
    x, y = check_xy(x, y)
    n = len(x)
    lag = check_lag(lag, n)
    if np.ptp(x) == 0.0:
        raise ValueError("x is constant: the regression slope is not identified")
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    u = y - X @ beta
    ss_res, ss_tot = float(u @ u), float(((y - y.mean()) ** 2).sum())
    se_ols = np.sqrt(np.diag(XtX_inv) * ss_res / (n - 2))
    Xu = X * u[:, None]
    S = Xu.T @ Xu
    for k in range(1, lag + 1):
        G = Xu[k:].T @ Xu[:-k]
        S += (1.0 - k / (lag + 1.0)) * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv * n / (n - 2)           # small-sample scaling
    se_hac = np.sqrt(np.diag(cov))
    return dict(n=n, intercept=float(beta[0]), slope=float(beta[1]), r2=1.0 - ss_res / ss_tot,
                se_slope_ols=float(se_ols[1]), se_slope_hac=float(se_hac[1]),
                se_intercept_hac=float(se_hac[0]), lag=lag)


def mean_hac(y: np.ndarray, lag: int) -> tuple[float, float]:
    """Sample mean and its Newey-West (Bartlett) standard error."""
    y = check_series("y", y, 2)
    n = len(y)
    lag = check_lag(lag, n)
    u = y - y.mean()
    s = float(u @ u) / n
    for k in range(1, lag + 1):
        s += 2.0 * (1.0 - k / (lag + 1.0)) * float(u[k:] @ u[:-k]) / n
    return float(y.mean()), float(np.sqrt(s / n))


def block_bootstrap_r2(x, y, block: int, n_boot: int, rng) -> tuple[float, float, float]:
    """Circular moving-block bootstrap of the OLS R^2 (2.5%, 50%, 97.5% percentiles)."""
    x, y = check_xy(x, y)
    n = len(x)
    check_bootstrap(n, block, n_boot)
    n_blocks = int(np.ceil(n / block))
    out = np.empty(n_boot)
    for b in range(n_boot):
        starts = rng.integers(0, n, n_blocks)
        idx = ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]
        out[b] = np.corrcoef(x[idx], y[idx])[0, 1] ** 2
    return tuple(float(v) for v in np.percentile(out, [2.5, 50, 97.5]))


# ----------------------------------------------------------------------------------------- core
@dataclass
class Backtest:
    ws: rd.WindowSet
    r: object                  # float, or (M,) array of per-window rates
    pnl: np.ndarray            # (M,) PnL as a FRACTION of S0, daily rebalancing
    premium: np.ndarray        # (M,) premium as a fraction of S0
    rv: np.ndarray             # (M,) annualised realised vol (decimal)
    gap: np.ndarray            # (M,) sigma_i^2 - sigma_r^2
    est: np.ndarray            # (M,) gamma-formula estimate (fraction of S0)
    slope_theory: np.ndarray   # (M,) per-window sum_i w_i dt / S0  (the gamma-formula slope on the gap)


BASELINE_RATE_LABEL = "RBI 91-day T-bill (baseline)"


def window_rate(ws: rd.WindowSet, r=None):
    """The rate to use: ``r`` if given (scalar or one per window), else the window set's own T-bill rate, else the
    constant fallback (synthetic fixtures without a rate)."""
    if r is not None:
        return r
    return ws.r if ws.r is not None else cfg.RISK_FREE_RATE_B


def run_backtest(ws: rd.WindowSet, r=None, centred_rv: bool = False,
                 rebalance_every: int = 1, sigma_hedge=None) -> Backtest:
    W = ws.window
    if isinstance(rebalance_every, bool) or not isinstance(rebalance_every, (int, np.integer)) \
            or not 1 <= rebalance_every <= W:
        raise ValueError(f"rebalance_every must be an integer in [1, {W}]")
    r = window_rate(ws, r)
    if not np.all(np.isfinite(r)):
        raise ValueError("r must be finite")
    T = W / cfg.TRADING_DAYS_PER_YEAR
    times = np.arange(W + 1) / cfg.TRADING_DAYS_PER_YEAR
    S0 = ws.paths[:, 0]
    res = hedge_pnl(ws.paths, S0, ws.sigma_implied, r, T, times=times,
                    rebalance_idx=range(0, W, rebalance_every), sigma_hedge=sigma_hedge)
    rv = realised_vol(ws.paths, centred=centred_rv)
    gap = ws.sigma_implied**2 - rv**2
    est = gamma_pnl_estimate(ws.paths, S0, ws.sigma_implied, r, T, times=times)
    w_sum = gamma_pnl_estimate(ws.paths, S0, ws.sigma_implied, r, T, times=times, realised_var=0.0) / ws.sigma_implied**2
    return Backtest(ws=ws, r=r, pnl=res.pnl / S0, premium=res.premium / S0, rv=rv, gap=gap,
                    est=est / S0, slope_theory=w_sum / S0)


def regression_block(bt: Backtest, mask=None, lag: int = None) -> dict:
    lag = bt.ws.window - 1 if lag is None else lag
    m = np.ones(len(bt.pnl), bool) if mask is None else np.asarray(mask)
    return ols_hac(bt.gap[m], bt.pnl[m], lag)


def _row(name: str, bt: Backtest, mask=None, lag=None) -> dict:
    m = np.ones(len(bt.pnl), bool) if mask is None else np.asarray(mask)
    reg = regression_block(bt, m, lag)
    p = bt.pnl[m] * 100
    return dict(group=name, windows=int(m.sum()), mean_pnl_pct=p.mean(), std_pnl_pct=p.std(ddof=1),
                median_pnl_pct=float(np.median(p)), frac_profit=float((p > 0).mean()),
                mean_vix=float(bt.ws.vix_pct[m].mean()), mean_rv_pct=float(bt.rv[m].mean() * 100),
                slope=reg["slope"], se_slope_hac=reg["se_slope_hac"], r2=reg["r2"])


def crash_mask(ws: rd.WindowSet) -> np.ndarray:
    return np.asarray((ws.t0 <= CRASH_END) & (ws.t_end >= CRASH_START))


# ----------------------------------------------------------------------------------------- GBM benchmarks
def gbm_benchmarks(ws: rd.WindowSet, bt: Backtest, r=None, n_paths: int = 40_000,
                   twin_reps: int = 20) -> pd.DataFrame:
    """What the Part B regression would look like if returns were exactly GBM (same 21 daily steps).

    (a) 'constant GBM': one volatility regime: sigma_i = mean VIX, sigma_r = mean realised vol of the real sample.
    (b) 'GBM twin', a benchmark *conditional on the historical realised-volatility estimates*: every real window
        gets GBM paths (``twin_reps`` per window) whose TRUE (input) volatility is that window's historical
        realised vol and whose implied vol is that window's VIX.  The distribution of the *input* volatilities
        therefore matches the historical one, and the data have no fat tails, vol clustering or trend-dependence.
        The regressor, however, is the realised vol of each *simulated* path, which equals the input vol plus
        21-observation sampling noise, so the regressor distribution is NOT identical to the historical one
        (see ``twin_volatility_comparison``).
    The simulated drift is mu = r (the window's own T-bill rate in the twin; the sample mean rate in the constant-GBM row).
    """
    r = window_rate(ws, r) if r is not None else bt.r
    r_mean = float(np.mean(r))
    W = ws.window
    T = W / cfg.TRADING_DAYS_PER_YEAR
    dt = 1.0 / cfg.TRADING_DAYS_PER_YEAR
    times = np.arange(W + 1) * dt
    rows = []

    si, sr = float(ws.sigma_implied.mean()), float(bt.rv.mean())
    rng = make_rng(11, 0)
    P = simulate_gbm_paths(n_paths, W, 100.0, r_mean, sr, T, rng)
    res = hedge_pnl(P, 100.0, si, r_mean, T, times=times)
    o = ols_hac(si**2 - realised_vol(P) ** 2, res.pnl / 100.0, 0)
    rows.append(dict(benchmark="constant GBM (sigma_i = mean VIX, sigma_r = mean realised vol)", observations=n_paths,
                     slope=o["slope"], r2=o["r2"], mean_pnl_pct=float(res.pnl.mean()), std_pnl_pct=float(res.pnl.std(ddof=1)),
                     sigma_i=si, sigma_r=sr))

    X, Y = [], []
    for k in range(twin_reps):
        rng = make_rng(12, k)
        sig_r = bt.rv[:, None]
        Z = rng.standard_normal((len(bt.rv), W))
        r_col = np.broadcast_to(np.asarray(r, dtype=float), (len(bt.rv),))[:, None]
        lr = (r_col - 0.5 * sig_r**2) * dt + sig_r * np.sqrt(dt) * Z
        P = 100.0 * np.exp(np.concatenate([np.zeros((len(bt.rv), 1)), np.cumsum(lr, axis=1)], axis=1))
        res = hedge_pnl(P, 100.0, ws.sigma_implied, r, T, times=times)
        X.append(ws.sigma_implied**2 - realised_vol(P) ** 2)
        Y.append(res.pnl / 100.0)
    X, Y = np.concatenate(X), np.concatenate(Y)
    o = ols_hac(X, Y, 0)
    rows.append(dict(benchmark="GBM twin, conditional on historical realised vol (sigma_i = VIX_t0; true vol of each path = that window's historical realised vol)",
                     observations=len(Y), slope=o["slope"], r2=o["r2"], mean_pnl_pct=float(Y.mean() * 100),
                     std_pnl_pct=float(Y.std(ddof=1) * 100), sigma_i=float("nan"), sigma_r=float("nan")))
    return pd.DataFrame(rows)


def timing_effect(paths: np.ndarray, sigma_i: np.ndarray, r) -> np.ndarray:
    """(path-wise gamma estimate) - (constant-realised-variance gamma estimate), as % of S0, per window.

    Zero on average when the variance of a window is spread evenly in time relative to where gamma is large.
    Positive when the variance arrives while the index is far from the strike (gamma small): the seller then
    loses less than the window-average realised variance suggests.
    """
    W = paths.shape[1] - 1
    T = W / cfg.TRADING_DAYS_PER_YEAR
    times = np.arange(W + 1) / cfg.TRADING_DAYS_PER_YEAR
    S0 = paths[:, 0]
    rv2 = realised_vol(paths) ** 2
    e_path = gamma_pnl_estimate(paths, S0, sigma_i, r, T, times=times)
    e_const = gamma_pnl_estimate(paths, S0, sigma_i, r, T, times=times, realised_var=rv2)
    return (e_path - e_const) / S0 * 100.0


def _twin_paths(bt: Backtest, r, k: int = 0) -> np.ndarray:
    """One GBM path per real window with true vol = that window's realised vol (replicate k)."""
    W = bt.ws.window
    dt = 1.0 / cfg.TRADING_DAYS_PER_YEAR
    rng = make_rng(12, k)
    sig_r = bt.rv[:, None]
    Z = rng.standard_normal((len(bt.rv), W))
    r_col = np.broadcast_to(np.asarray(r, dtype=float), (len(bt.rv),))[:, None]
    lr = (r_col - 0.5 * sig_r**2) * dt + sig_r * np.sqrt(dt) * Z
    return 100.0 * np.exp(np.concatenate([np.zeros((len(bt.rv), 1)), np.cumsum(lr, axis=1)], axis=1))


def twin_volatility_comparison(ws: rd.WindowSet, bt: Backtest, r=None,
                               twin_reps: int = 20) -> tuple[pd.DataFrame, dict]:
    """Compare the GBM twin's volatilities with the historical ones, and say what the twin does and does not match.

    The twin simulates every window with true (input) volatility = that window's HISTORICAL realised vol.  Each
    simulated 21-step path then has its own realised vol = input vol + sampling noise, and that simulated realised
    vol is what the benchmark regression uses.  So the twin matches the distribution of the *input* volatilities to
    the historical realised vols; it does not reproduce the historical realised-vol regressor window by window.
    Uses the same random streams as ``gbm_benchmarks`` (so the twin regression below equals the benchmark row).
    """
    r = bt.r if r is None else r
    W = ws.window
    T = W / cfg.TRADING_DAYS_PER_YEAR
    times = np.arange(W + 1) / cfg.TRADING_DAYS_PER_YEAR
    sim_rv, pnl = [], []
    for k in range(twin_reps):
        P = _twin_paths(bt, r, k)
        sim_rv.append(realised_vol(P))
        pnl.append(hedge_pnl(P, 100.0, ws.sigma_implied, r, T, times=times).pnl / 100.0)
    sim_rv, pnl = np.concatenate(sim_rv), np.concatenate(pnl)
    inp = np.tile(bt.rv, twin_reps)                    # the twin's input (true) volatility
    si = np.tile(ws.sigma_implied, twin_reps)
    gap_sim, gap_in = si**2 - sim_rv**2, si**2 - inp**2

    def line(name, unit, x):
        q = np.percentile(x, [5, 25, 50, 75, 95])
        return dict(quantity=name, unit=unit, n=len(x), mean=float(np.mean(x)), std=float(np.std(x, ddof=1)),
                    p05=float(q[0]), p25=float(q[1]), p50=float(q[2]), p75=float(q[3]), p95=float(q[4]))

    rows = [
        line("VIX at window start (the implied vol, sigma_i)", "vol points", ws.vix_pct),
        line("historical realised vol = the twin's input (true) vol", "vol points", bt.rv * 100),
        line("simulated realised vol of the twin paths (the regressor's ingredient)", "vol points", sim_rv * 100),
        line("simulated minus input vol (pure sampling noise)", "vol points", (sim_rv - inp) * 100),
        line("variance gap, real windows (VIX^2 - historical realised^2)", "variance", bt.gap),
        line("variance gap, twin, as regressed in the benchmark (VIX^2 - simulated realised^2)", "variance", gap_sim),
        line("variance gap, twin, using the input vol (VIX^2 - input^2, no sampling noise)", "variance", gap_in),
    ]
    o_sim, o_in = ols_hac(gap_sim, pnl, 0), ols_hac(gap_in, pnl, 0)
    summary = dict(
        replicates=twin_reps, observations=int(len(pnl)),
        slope_on_simulated_gap=o_sim["slope"], r2_on_simulated_gap=o_sim["r2"],
        slope_on_input_gap=o_in["slope"], r2_on_input_gap=o_in["r2"],
        std_ratio_simulated_over_historical_rv=float(sim_rv.std(ddof=1) / bt.rv.std(ddof=1)),
        std_ratio_simulated_over_historical_gap=float(gap_sim.std(ddof=1) / bt.gap.std(ddof=1)),
        relative_sampling_noise_std=float(np.std(sim_rv / inp - 1.0, ddof=1)),
        theory_relative_sampling_noise=float(1.0 / np.sqrt(2.0 * W)),
        corr_vix_rv_real=float(np.corrcoef(ws.sigma_implied, bt.rv)[0, 1]),
        corr_vix_rv_twin=float(np.corrcoef(si, sim_rv)[0, 1]),
        share_rv_below_vix_real=float((bt.rv < ws.sigma_implied).mean()),
        share_rv_below_vix_twin=float((sim_rv < si).mean()),
    )
    return pd.DataFrame(rows), summary


# ----------------------------------------------------------------------------------------- dependence robustness
HAC_LAGS = (0, 5, 10, 20, 42, 63)       # lag 0 = White (no autocorrelation correction); the headline analysis uses 20
BOOT_BLOCKS = (21, 42, 63)              # circular block-bootstrap block lengths, in windows
BOOT_REPEATS, BOOT_N = 5, 2000          # independent repeats of a full 2,000-resample bootstrap per block length


def hac_lag_sensitivity(bt: Backtest, lags=HAC_LAGS) -> pd.DataFrame:
    """Standard error of the headline slope under several dependence assumptions (the headline analysis uses lag 20).

    First row: plain OLS (assumes independent, equal-variance errors).  Lag 0: White heteroskedasticity-robust
    (still ignores the overlap).  Lags >= 1: Newey-West with a Bartlett kernel.
    """
    def row(method, lag, o, se):
        return dict(method=method, lag=lag, se_slope=se, t_stat=o["slope"] / se,
                    slope_ci_low=o["slope"] - 1.96 * se, slope_ci_high=o["slope"] + 1.96 * se)

    rows = []
    o = ols_hac(bt.gap, bt.pnl, 0)
    rows.append(row("plain OLS (ignores the overlap; not valid here)", 0, o, o["se_slope_ols"]))
    for L in lags:
        o = ols_hac(bt.gap, bt.pnl, L)
        if L == 0:
            rows.append(row("White, heteroskedasticity-robust only (ignores the overlap)", 0, o, o["se_slope_hac"]))
        else:
            rows.append(row("Newey-West" + (" (headline: lag = window length - 1)" if L == 20 else ""), L, o, o["se_slope_hac"]))
    return pd.DataFrame(rows)


def bootstrap_block_sensitivity(bt: Backtest, blocks=BOOT_BLOCKS, repeats: int = BOOT_REPEATS,
                                n_boot: int = BOOT_N) -> pd.DataFrame:
    """Block-bootstrap 95% interval for the headline R^2 at several block lengths.

    Each block length is run ``repeats`` times with independent random streams, each with the full ``n_boot``
    resamples, so the Monte Carlo noise of the interval bounds can be separated from the effect of the block length.
    Reported: mean of the repeats' bounds and the range (min, max) across repeats.
    """
    rows = []
    for blk in blocks:
        res = np.array([block_bootstrap_r2(bt.gap, bt.pnl, block=blk, n_boot=n_boot, rng=make_rng(13, blk, rep))
                        for rep in range(repeats)])               # (repeats, 3): 2.5%, 50%, 97.5%
        rows.append(dict(block_length=blk, resamples_per_run=n_boot, runs=repeats,
                         r2_low=float(res[:, 0].mean()), r2_median=float(res[:, 1].mean()), r2_high=float(res[:, 2].mean()),
                         r2_low_min=float(res[:, 0].min()), r2_low_max=float(res[:, 0].max()),
                         r2_high_min=float(res[:, 2].min()), r2_high_max=float(res[:, 2].max())))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------- estimator conventions
ESTIMATOR_CONVENTIONS = (
    ("path-wise, squared log returns (baseline)", dict(return_convention="log")),
    ("path-wise, squared simple returns", dict(return_convention="simple")),
    ("path-wise, log returns demeaned within the window", dict(return_convention="log_demeaned")),
    ("constant variance, un-centred window variance", dict(const="uncentred")),
    ("constant variance, centred (sample) variance", dict(const="centred")),
)


def _fit_row(est: np.ndarray, pnl: np.ndarray) -> dict:
    """How well an estimate of PnL (fractions of S0) tracks the actual hedging PnL."""
    c = np.corrcoef(est, pnl)[0, 1]
    slope = float(np.cov(est, pnl, ddof=1)[0, 1] / est.var(ddof=1))
    err = est - pnl
    return dict(r2=float(c**2), slope_pnl_on_estimate=slope, mean_error_pct=float(err.mean() * 100),
                rmse_pct=float(np.sqrt((err**2).mean()) * 100), rmse_over_std_pnl=float(np.sqrt((err**2).mean()) / pnl.std(ddof=1)))


def estimator_conventions(ws: rd.WindowSet, bt: Backtest, crash: np.ndarray, r=None) -> pd.DataFrame:
    """Sensitivity of the gamma-PnL approximation to how realised variance is defined.

    For each convention, the discrete gamma estimate is compared with the actual hedging PnL on (a) all real
    windows, (b) real windows excluding the 2020 crash windows, and (c) a GBM twin (one simulated path per real
    window; returns exactly lognormal, so the only error is the discretisation).
    """
    r = bt.r if r is None else r
    W = ws.window
    T = W / cfg.TRADING_DAYS_PER_YEAR
    times = np.arange(W + 1) / cfg.TRADING_DAYS_PER_YEAR
    twin = _twin_paths(bt, r, 0)
    twin_pnl = hedge_pnl(twin, 100.0, ws.sigma_implied, r, T, times=times).pnl / 100.0
    sets = [("real, all windows", ws.paths, bt.pnl, np.ones(len(ws.t0), bool)),
            ("real, excluding crash windows", ws.paths, bt.pnl, ~crash),
            ("GBM twin (all windows)", twin, twin_pnl, np.ones(len(ws.t0), bool))]
    rows = []
    for dname, paths, pnl, mask in sets:
        S0 = paths[:, 0]
        for cname, kw in ESTIMATOR_CONVENTIONS:
            if "const" in kw:
                rv2 = realised_vol(paths, centred=(kw["const"] == "centred")) ** 2
                est = gamma_pnl_estimate(paths, S0, ws.sigma_implied, r, T, times=times, realised_var=rv2) / S0
            else:
                est = gamma_pnl_estimate(paths, S0, ws.sigma_implied, r, T, times=times, **kw) / S0
            rows.append(dict(dataset=dname, convention=cname, windows=int(mask.sum()), **_fit_row(est[mask], pnl[mask])))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------- study
def run_part_b(save: bool = True) -> dict:
    data = rd.load_all()
    ws = rd.build_windows(data)
    bt = run_backtest(ws)
    M, W = len(ws.t0), ws.window
    lag = W - 1
    crash = crash_mask(ws)
    rng = make_rng(10, 0)

    # ---- headline regression (overlapping windows -> HAC SE, block bootstrap, non-overlapping subsamples)
    head = regression_block(bt)
    r2_lo, r2_med, r2_hi = block_bootstrap_r2(bt.gap, bt.pnl, block=W, n_boot=2000, rng=rng)
    non_overlap = []
    for off in range(W):
        idx = np.arange(off, M, W)
        o = ols_hac(bt.gap[idx], bt.pnl[idx], 0)
        non_overlap.append(dict(offset=off, n=o["n"], slope=o["slope"], r2=o["r2"], se_slope=o["se_slope_ols"]))
    no = pd.DataFrame(non_overlap)
    corr_est = float(np.corrcoef(bt.pnl, bt.est)[0, 1] ** 2)
    ols_est = ols_hac(bt.est, bt.pnl, lag)

    # ---- volatility risk premium
    vix_iv = ws.sigma_implied * 100
    rv_pct = bt.rv * 100
    d_vol = vix_iv - rv_pct
    d_var = bt.gap
    _, se_vol = mean_hac(d_vol, lag)
    _, se_var = mean_hac(d_var, lag)
    vrp = dict(mean_vix=float(vix_iv.mean()), mean_rv=float(rv_pct.mean()),
               mean_vol_gap_points=float(d_vol.mean()), se_vol_gap_hac=se_vol,
               median_vol_gap_points=float(np.median(d_vol)), frac_windows_rv_below_vix=float((d_vol > 0).mean()),
               mean_var_gap=float(d_var.mean()), se_var_gap_hac=se_var,
               corr_vix_rv=float(np.corrcoef(vix_iv, rv_pct)[0, 1]))
    reg_rv_on_vix = ols_hac(vix_iv, rv_pct, lag)
    vrp.update(rv_on_vix_slope=reg_rv_on_vix["slope"], rv_on_vix_intercept=reg_rv_on_vix["intercept"],
               rv_on_vix_r2=reg_rv_on_vix["r2"], rv_on_vix_slope_se_hac=reg_rv_on_vix["se_slope_hac"])

    # ---- rebalancing-frequency study
    bts = {k: run_backtest(ws, rebalance_every=k) for k in FREQ_INTERVALS}
    freq_rows = []
    for k, b in bts.items():
        p = b.pnl * 100
        diff = (b.pnl - bts[1].pnl) * 100
        n_reb = len(range(0, W, k))
        reg = regression_block(b)
        q = np.percentile(p, [5, 95])
        freq_rows.append(dict(interval_days=k, rebalances=n_reb, mean_pnl_pct=p.mean(), std_pnl_pct=p.std(ddof=1),
                              p05_pct=q[0], p95_pct=q[1], std_diff_vs_daily_pct=diff.std(ddof=1),
                              r2_on_gap=reg["r2"], slope_on_gap=reg["slope"]))
    freq = pd.DataFrame(freq_rows)
    # 1/sqrt(N) law for the discretisation part: e_k = c/sqrt(N_k) independent of the daily error, so
    # std(PnL_k - PnL_1) = c * sqrt(1/N_k - 1/N_1).  c is fitted by least squares over k = 2, 5, 10.
    basis = np.sqrt(1.0 / freq["rebalances"] - 1.0 / freq["rebalances"].iloc[0]).to_numpy()
    obs = freq["std_diff_vs_daily_pct"].to_numpy()
    c_fit = float((basis[1:] * obs[1:]).sum() / (basis[1:] ** 2).sum())
    freq["std_diff_predicted_by_sqrtN_law_pct"] = c_fit * basis
    freq["c_fitted"] = c_fit

    # ---- sensitivities
    r_mean = float(np.mean(bt.r))                       # mean T-bill rate over the sample (decimal)
    sens_rows = [dict(_row(BASELINE_RATE_LABEL, bt), risk_free=r_mean)]
    for r in cfg.RISK_FREE_SENSITIVITY_B:
        b = run_backtest(ws, r=r)
        row = _row(f"constant risk-free {r:.0%}", b)
        row["risk_free"] = r
        sens_rows.append(row)
    centred = run_backtest(ws, centred_rv=True)
    sens_rows.append(dict(_row("realised vol = sample std (centred)", centred), risk_free=r_mean))
    sens_rows.append(dict(_row("excluding Feb 20 - Apr 30 2020 windows", bt, ~crash), risk_free=r_mean))
    data_nw = rd.load_all(drop_weekend_sessions=True)
    ws_nw = rd.build_windows(data_nw)
    bt_nw = run_backtest(ws_nw)
    sens_rows.append(dict(_row("weekend sessions dropped", bt_nw), risk_free=float(np.mean(bt_nw.r))))
    sens = pd.DataFrame(sens_rows)
    base = sens.loc[sens["group"] == BASELINE_RATE_LABEL].iloc[0]
    rate_rows = []
    for label in [BASELINE_RATE_LABEL] + [f"constant risk-free {r:.0%}" for r in cfg.RISK_FREE_SENSITIVITY_B]:
        row = sens.loc[sens["group"] == label].iloc[0]
        rate_rows.append(dict(rate=label.replace("constant risk-free ", "constant "), risk_free=row["risk_free"],
                              **{c: row[c] for c in ("mean_pnl_pct", "std_pnl_pct", "frac_profit", "slope", "r2")},
                              **{f"diff_{c}": row[c] - base[c] for c in ("mean_pnl_pct", "std_pnl_pct", "frac_profit", "slope", "r2")}))
    rate_sens = pd.DataFrame(rate_rows)
    bench = gbm_benchmarks(ws, bt)
    conventions = estimator_conventions(ws, bt, crash)
    twin_vol, twin_cond = twin_volatility_comparison(ws, bt)
    hac_sens = hac_lag_sensitivity(bt)
    boot_sens = bootstrap_block_sensitivity(bt)

    # ---- extended empirical research (src/research.py): effect-size CIs, regimes, estimators, chronological split
    from . import frictions as fr
    from . import research as rs
    friction_tbl = fr.friction_grid(ws)
    effects = rs.effect_size_cis(bt, lag, make_rng)
    slope_boot_crash = rs.slope_bootstrap_vs_crash_share(bt, crash, make_rng)
    state_lab = rs.regime_labels(data, ws)
    market_states = rs.regime_table(bt, state_lab, lag)
    est_table, est_summary = rs.estimator_comparison(data, ws, bt, lag)
    split_table, split_summary = rs.chronological_split(ws, bt, rs.vol_estimators(data, ws), rs.exante_features(data, ws))
    mc_error = rs.effect_size_mc_error(bt, lag, make_rng)
    from . import calendar_time as ct
    cal_table, cal_info = ct.calendar_time_comparison(data, ws, bt, lag)

    # ---- regimes and years
    v0 = ws.vix_pct
    regimes = pd.DataFrame([
        _row("all windows", bt),
        _row(f"calm: VIX at start < {CALM_VIX:g}", bt, v0 < CALM_VIX),
        _row(f"middle: {CALM_VIX:g} <= VIX < {TURBULENT_VIX:g}", bt, (v0 >= CALM_VIX) & (v0 < TURBULENT_VIX)),
        _row(f"turbulent: VIX at start >= {TURBULENT_VIX:g}", bt, v0 >= TURBULENT_VIX),
        _row("windows overlapping Feb 20 - Apr 30 2020", bt, crash),
    ])
    yr = pd.Series(ws.t0.year, index=range(M))
    yearly = pd.DataFrame([_row(str(y), bt, (yr == y).to_numpy(), lag=min(lag, 5)) for y in sorted(yr.unique())])
    yearly = yearly.drop(columns=["slope", "se_slope_hac", "r2"])    # too few independent obs per year to regress

    # ---- stretch: re-hedge each day with that day's VIX (only windows with a valid VIX on every day)
    vix_full = data.vix["Close"]
    pos = {d: i for i, d in enumerate(data.nifty.index)}
    start_pos = np.array([pos[d] for d in ws.t0])
    dates = data.nifty.index
    sig_path = vix_full.reindex(dates).to_numpy()[start_pos[:, None] + np.arange(W)[None, :]] / 100.0   # (M, W)
    ok = np.isfinite(sig_path).all(axis=1)
    ws_ok = rd.WindowSet(t0=ws.t0[ok], t_end=ws.t_end[ok], paths=ws.paths[ok], sigma_implied=ws.sigma_implied[ok],
                         vix_pct=ws.vix_pct[ok], calendar_days=ws.calendar_days[ok], window=W,
                         n_excluded_long_span=0, r=None if ws.r is None else ws.r[ok],
                         tbill_yield_pct=None if ws.tbill_yield_pct is None else ws.tbill_yield_pct[ok],
                         rate_stale_days=None if ws.rate_stale_days is None else ws.rate_stale_days[ok])
    bt_fixed = run_backtest(ws_ok)
    bt_vix = run_backtest(ws_ok, sigma_hedge=sig_path[ok])
    stretch = pd.DataFrame([dict(hedger="fixed initial VIX (baseline)", **{k: v for k, v in _row("", bt_fixed).items() if k != "group"}),
                            dict(hedger="re-hedge with each day's VIX", **{k: v for k, v in _row("", bt_vix).items() if k != "group"})])

    t_real = timing_effect(ws.paths, ws.sigma_implied, bt.r)
    t_twin = timing_effect(_twin_paths(bt, bt.r), ws.sigma_implied, bt.r)
    timing = {name: dict(all=float(t.mean()), crash=float(t[crash].mean()), non_crash=float(t[~crash].mean()))
              for name, t in (("real", t_real), ("gbm_twin", t_twin))}
    gw = dict(all=float(bt.slope_theory.mean()), crash=float(bt.slope_theory[crash].mean()),
              non_crash=float(bt.slope_theory[~crash].mean()))

    summary = dict(
        timing_effect_pct_of_S0=timing, gamma_weight_slope_by_group=gw,
        windows=M, first_t0=str(ws.t0[0].date()), last_t0=str(ws.t0[-1].date()), last_end=str(ws.t_end[-1].date()),
        window_trading_days=W, risk_free=cfg.RISK_FREE_RATE_B, risk_free_is_placeholder=cfg.RISK_FREE_IS_PLACEHOLDER,
        rate_baseline=dict(source=cfg.RISK_FREE_SOURCE_B, mean_rate_pct=float(np.mean(bt.r)) * 100,
                           min_rate_pct=float(np.min(bt.r)) * 100, max_rate_pct=float(np.max(bt.r)) * 100,
                           mean_yield_pct=float(np.mean(ws.tbill_yield_pct)),
                           first_auction=str(data.tbill.index[0].date()), last_auction=str(data.tbill.index[-1].date()),
                           auctions=int(len(data.tbill)),
                           stale_days_max=int(ws.rate_stale_days.max()),
                           windows_stale_over_28_days=int((ws.rate_stale_days > rates.FLAG_STALE_DAYS).sum()),
                           first_stale_window=(str(ws.t0[ws.rate_stale_days > rates.FLAG_STALE_DAYS][0].date())
                                               if (ws.rate_stale_days > rates.FLAG_STALE_DAYS).any() else None)),
        calendar_days_min=int(ws.calendar_days.min()), calendar_days_max=int(ws.calendar_days.max()),
        crash_windows=int(crash.sum()), data_report=data.report,
        headline=dict(**head, r2_block_bootstrap_ci=[r2_lo, r2_hi], r2_block_bootstrap_median=r2_med,
                      mean_pnl_pct=float(bt.pnl.mean() * 100), std_pnl_pct=float(bt.pnl.std(ddof=1) * 100),
                      frac_profit=float((bt.pnl > 0).mean()), mean_premium_pct=float(bt.premium.mean() * 100)),
        non_overlapping=dict(n_per_subsample=int(no["n"].min()), r2_min=float(no["r2"].min()), r2_median=float(no["r2"].median()),
                             r2_max=float(no["r2"].max()), slope_median=float(no["slope"].median()),
                             slope_min=float(no["slope"].min()), slope_max=float(no["slope"].max())),
        theory_slope_mean=float(bt.slope_theory.mean()), theory_slope_min=float(bt.slope_theory.min()),
        theory_slope_max=float(bt.slope_theory.max()),
        r2_pnl_vs_gamma_estimate=corr_est, ols_pnl_on_gamma_estimate=ols_est,
        vrp=vrp,
    )
    summary["twin_conditional"] = twin_cond
    summary["market_states"] = dict(drawdown_cut=rs.DRAWDOWN_CUT, momentum_days=rs.MOMENTUM_DAYS,
                                    unlabelled_windows=int((state_lab == "").sum()))
    summary["frictions"] = fr.summarise(friction_tbl)
    summary["vol_estimators"] = est_summary
    summary["chronological_split"] = split_summary
    summary["effect_size_block"] = dict(block=rs.CI_BLOCK, n_boot=rs.CI_BOOT, slope_bootstrap_vs_crash_share=slope_boot_crash,
                                        mc_reps=int(mc_error.reps.iloc[0]),
                                        mc_max_bound_sd_over_width=float(mc_error.max_bound_sd_over_width.max()))
    summary["calendar_time"] = cal_info
    out = dict(bt=bt, ws=ws, data=data, summary=summary, freq=freq, sens=sens, regimes=regimes, yearly=yearly,
               stretch=stretch, non_overlap=no, crash=crash, rate_sens=rate_sens, bench=bench, conventions=conventions, twin_vol=twin_vol,
               hac_sens=hac_sens, boot_sens=boot_sens, effects=effects, market_states=market_states,
               est_table=est_table, split_table=split_table, friction_tbl=friction_tbl, mc_error=mc_error,
               calendar_table=cal_table)
    if save:
        cfg.RESULTS_DIR.mkdir(exist_ok=True)
        pd.DataFrame(dict(t0=ws.t0, t_end=ws.t_end, vix_pct=ws.vix_pct, sigma_implied=ws.sigma_implied,
                          realised_vol=bt.rv, variance_gap=bt.gap, premium_pct=bt.premium * 100, pnl_pct=bt.pnl * 100,
                          gamma_estimate_pct=bt.est * 100, covid_crash_window=crash,
                          tbill_yield_pct=ws.tbill_yield_pct, risk_free_rate=bt.r, rate_stale_days=ws.rate_stale_days)).to_csv(
            cfg.RESULTS_DIR / "partB_windows.csv", index=False)
        freq.to_csv(cfg.RESULTS_DIR / "partB_frequency.csv", index=False)
        sens.to_csv(cfg.RESULTS_DIR / "partB_sensitivity.csv", index=False)
        rate_sens.to_csv(cfg.RESULTS_DIR / "partB_rate_sensitivity.csv", index=False)
        bench.to_csv(cfg.RESULTS_DIR / "partB_gbm_benchmarks.csv", index=False)
        conventions.to_csv(cfg.RESULTS_DIR / "partB_gamma_estimator_conventions.csv", index=False)
        twin_vol.to_csv(cfg.RESULTS_DIR / "partB_gbm_twin_volatility.csv", index=False)
        hac_sens.to_csv(cfg.RESULTS_DIR / "partB_hac_lag_sensitivity.csv", index=False)
        boot_sens.to_csv(cfg.RESULTS_DIR / "partB_bootstrap_block_sensitivity.csv", index=False)
        friction_tbl.to_csv(cfg.RESULTS_DIR / "partB_friction_scenarios.csv", index=False)
        effects.to_csv(cfg.RESULTS_DIR / "partB_effect_size_intervals.csv", index=False)
        market_states.to_csv(cfg.RESULTS_DIR / "partB_market_states.csv", index=False)
        est_table.to_csv(cfg.RESULTS_DIR / "partB_vol_estimator_comparison.csv", index=False)
        split_table.to_csv(cfg.RESULTS_DIR / "partB_chronological_split.csv", index=False)
        mc_error.to_csv(cfg.RESULTS_DIR / "partB_effect_size_mc_error.csv", index=False)
        cal_table.to_csv(cfg.RESULTS_DIR / "partB_calendar_time_sensitivity.csv", index=False)
        regimes.to_csv(cfg.RESULTS_DIR / "partB_regimes.csv", index=False)
        yearly.to_csv(cfg.RESULTS_DIR / "partB_yearly.csv", index=False)
        stretch.to_csv(cfg.RESULTS_DIR / "partB_stretch_vix_rehedge.csv", index=False)
        no.to_csv(cfg.RESULTS_DIR / "partB_nonoverlapping_subsamples.csv", index=False)
        (cfg.RESULTS_DIR / "partB_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    return out
