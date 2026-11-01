"""Generic discrete delta-hedging engine.

A seller sells ONE option at time 0 for its Black-Scholes price (computed with
the implied volatility ``sigma``) and then replicates it:

* holds ``delta`` units of the underlying, re-set at the rebalancing times;
* funds the position from a cash account that earns / pays the risk-free rate ``r``;
* at maturity the hedging PnL is  (value of hedge portfolio) - (option payoff owed).

Self-financing recursion (V = value of premium + hedge, delta_i set at t_i):

    V_{i+1} = V_i e^{r dt_i} + delta_i * ( S_{i+1} - S_i e^{r dt_i} ),    V_0 = premium

Unrolled (this is what the code evaluates, vectorised over paths and steps):

    V_T = premium e^{rT} + sum_i delta_i ( S_{i+1} - S_i e^{r dt_i} ) e^{r (T - t_{i+1})}

PnL is therefore measured in money *at maturity* (not discounted back to t=0).

Optional proportional transaction costs: every trade of size |d delta| in the
underlying at price S costs ``cost_rate * |d delta| * S`` (paid in cash at the
time of the trade and compounded to T).  The initial purchase counts.  If
``cost_rate > 0`` the final hedge position is also liquidated at maturity and
that trade is charged too (the option is treated as cash-settled).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import bs


@dataclass
class HedgeResult:
    pnl: np.ndarray            # (M,) hedging PnL at maturity (after costs)
    premium: np.ndarray        # (M,) option premium received at t0
    total_cost: np.ndarray     # (M,) transaction costs compounded to maturity (0 if cost_rate == 0)
    n_rebalances: int          # number of times the hedge was (re)set, including t0
    deltas: np.ndarray | None  # (M, n_rebalances) hedge ratios, only if requested


def _as_paths(paths) -> np.ndarray:
    p = np.asarray(paths, dtype=float)
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] < 2:
        raise ValueError("paths must have shape (n_paths, n_steps + 1) with n_steps >= 1")
    if not np.all(np.isfinite(p)) or np.any(p <= 0):
        raise ValueError("paths must be finite and strictly positive")
    return p


def _check_scalar(name: str, x, *, positive: bool = False, nonneg: bool = False, upper: float | None = None) -> float:
    """Validate a scalar model input: finite, optionally > 0 / >= 0 / <= upper."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a real number") from None
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite")
    if positive and v <= 0:
        raise ValueError(f"{name} must be strictly positive")
    if nonneg and v < 0:
        raise ValueError(f"{name} must be non-negative")
    if upper is not None and v > upper:
        raise ValueError(f"{name} must be <= {upper}")
    return v


def _check_per_path(name: str, x, M: int, *, positive: bool = True) -> np.ndarray:
    """Validate a per-path array (strike or volatility): finite, > 0, broadcastable to M paths."""
    a = np.asarray(x, dtype=float)
    if a.ndim > 1 or (a.ndim == 1 and a.shape[0] != M):
        raise ValueError(f"{name} must be a scalar or have one entry per path ({M})")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} must be finite")
    if positive and np.any(a <= 0):
        raise ValueError(f"{name} must be strictly positive")
    return a


def _check_times(times, ncol: int, T: float) -> np.ndarray:
    """Validate a time grid: finite, one entry per path column, strictly increasing, starting at 0, ending at T.

    ``T`` is the elapsed time from inception (t = 0) to maturity, so the premium (priced with T - t0) and its
    growth at the risk-free rate (e^{rT}) are only consistent when the grid starts at zero.
    """
    times = np.asarray(times, dtype=float)
    if not np.isfinite(T) or T <= 0:
        raise ValueError("T must be finite and positive")
    if not np.all(np.isfinite(times)):
        raise ValueError("times must be finite")
    if times.shape != (ncol,) or np.any(np.diff(times) <= 0):
        raise ValueError("times must be strictly increasing with one entry per path column")
    if abs(times[0]) > 1e-9:
        raise ValueError("times must start at zero")
    if abs(times[-1] - T) > 1e-9:
        raise ValueError("last entry of times must equal T")
    return times


def _check_rebalance_idx(rebalance_idx, n: int) -> np.ndarray:
    """Validate the ORIGINAL elements before any integer conversion.

    Casting first would silently turn 1.9 into 1 and True into 1, i.e. hedge on a schedule other than the
    one requested.  Accepts integers (Python or NumPy) only; rejects Booleans, NaN/inf and non-integral
    floats, then requires index 0 first, strict increase, and every index <= n-1.
    """
    try:
        items = list(rebalance_idx)
    except TypeError as exc:
        raise ValueError("rebalance_idx must be a sequence of integers") from exc
    if not items:
        raise ValueError("rebalance_idx must not be empty")
    out = []
    for v in items:
        if isinstance(v, (bool, np.bool_)):
            raise ValueError("rebalance_idx must contain integers, not Booleans")
        if isinstance(v, (int, np.integer)):
            out.append(int(v))
            continue
        if isinstance(v, (float, np.floating)):
            if not np.isfinite(v):
                raise ValueError("rebalance_idx must be finite")
            if float(v) != int(v):
                raise ValueError(f"rebalance_idx must contain whole numbers; got {v!r}")
            out.append(int(v))
            continue
        raise ValueError(f"rebalance_idx must contain integers; got {type(v).__name__}")
    idx = np.asarray(out, dtype=int)
    if idx[0] != 0 or np.any(np.diff(idx) <= 0) or idx[-1] > n - 1:
        raise ValueError("rebalance_idx must start at 0, increase strictly and be <= n-1")
    return idx


def hedge_pnl(
    paths,
    K,
    sigma,
    r,
    T,
    *,
    times=None,
    rebalance_idx=None,
    option: str = "call",
    sigma_hedge=None,
    cost_rate: float = 0.0,
    delta_step: float | None = None,
    return_deltas: bool = False,
) -> HedgeResult:
    """Delta-hedge a short option along each price path.

    Parameters
    ----------
    paths : array (M, n+1) or (n+1,)
        Underlying prices on the full time grid; column 0 is t0, the last column is maturity T.
    K : float or array (M,)
        Strike (per path if an array).
    sigma : float or array (M,)
        Implied volatility used to price the option at t0 AND (unless ``sigma_hedge`` is
        given) to compute every delta.  Fixed for the life of the trade (baseline hedger).
    r : float
        Continuously compounded risk-free rate.
    T : float
        Time of the last column, in years from t0.
    times : array (n+1,), optional
        Time stamps in years of each column (default: equally spaced on [0, T]).
    rebalance_idx : sequence of int, optional
        Column indices at which the hedge is (re)set.  Must start at 0, be strictly increasing and
        lie in [0, n-1].  Default: every column except the last.  Use e.g. ``range(0, n, k)`` to
        rebalance every k steps while still marking to market at maturity.
    option : 'call' | 'put'
        The option the seller is short.
    sigma_hedge : array (M, n), optional
        Per-step volatility to use for the delta at each *column* 0..n-1 (stretch goal: re-hedge
        with the current VIX).  Premium is still priced with ``sigma``.
    cost_rate : float
        Proportional transaction cost per unit of notional traded.
    delta_step : float, optional
        Trading-size granularity: every hedge ratio is rounded to the nearest multiple of ``delta_step``
        (e.g. 0.01 = hedging in 1% slices of the option's notional, a stylised stand-in for whole contract
        lots).  ``None`` (default) means perfectly divisible positions and leaves all earlier results unchanged.
    return_deltas : bool
        Also return the delta used at each rebalance.
    """
    P = _as_paths(paths)
    M, ncol = P.shape
    n = ncol - 1
    if option not in ("call", "put"):
        raise ValueError("option must be 'call' or 'put'")
    r = _check_scalar("r", r)
    cost_rate = _check_scalar("cost_rate", cost_rate, nonneg=True, upper=1.0)
    K = _check_per_path("K", K, M)
    sigma = _check_per_path("sigma", sigma, M)
    T = float(T)
    if times is None:
        times = np.linspace(0.0, T, ncol)
    times = _check_times(times, ncol, T)

    if rebalance_idx is None:
        idx = np.arange(0, n)
    else:
        idx = _check_rebalance_idx(rebalance_idx, n)
    grid = np.concatenate([idx, [n]])           # hedge is held between consecutive grid points
    S_g = P[:, grid]
    t_g = times[grid]

    K_arr = np.broadcast_to(np.asarray(K, dtype=float), (M,))[:, None]
    sig_arr = np.broadcast_to(np.asarray(sigma, dtype=float), (M,))[:, None]

    tau = (T - t_g[:-1])[None, :]
    if sigma_hedge is None:
        sig_h = sig_arr
    else:
        sig_h = np.asarray(sigma_hedge, dtype=float)
        if sig_h.shape != (M, n):
            raise ValueError("sigma_hedge must have shape (n_paths, n_steps)")
        sig_h = sig_h[:, idx]
    delta = bs.delta(S_g[:, :-1], K_arr, tau, sig_h, r, option)          # (M, len(idx))
    if delta_step is not None:
        step = _check_scalar("delta_step", delta_step, positive=True, upper=1.0)
        delta = np.round(delta / step) * step

    premium = bs.price(P[:, 0], K_arr[:, 0], T - times[0], sig_arr[:, 0], r, option)

    dt = np.diff(t_g)
    grow = np.exp(r * dt)[None, :]                       # e^{r dt_i}
    to_T = np.exp(r * (T - t_g[1:]))[None, :]            # e^{r (T - t_{i+1})}
    gains = (delta * (S_g[:, 1:] - S_g[:, :-1] * grow) * to_T).sum(axis=1)
    V_T = premium * np.exp(r * T) + gains

    S_T = P[:, -1]
    payoff = np.maximum(S_T - K_arr[:, 0], 0.0) if option == "call" else np.maximum(K_arr[:, 0] - S_T, 0.0)

    cost = np.zeros(M)
    if cost_rate > 0.0:
        d_prev = np.concatenate([np.zeros((M, 1)), delta[:, :-1]], axis=1)
        trade_cost = cost_rate * np.abs(delta - d_prev) * S_g[:, :-1]    # at each rebalance time
        cost = (trade_cost * np.exp(r * (T - t_g[:-1]))[None, :]).sum(axis=1)
        cost += cost_rate * np.abs(delta[:, -1]) * S_T                   # liquidate hedge at T
    pnl = V_T - payoff - cost

    return HedgeResult(
        pnl=pnl,
        premium=premium,
        total_cost=cost,
        n_rebalances=len(idx),
        deltas=delta if return_deltas else None,
    )


RETURN_CONVENTIONS = ("log", "simple", "log_demeaned")


def gamma_pnl_estimate(paths, K, sigma, r, T, *, realised_var=None, times=None,
                       return_convention: str = "log") -> np.ndarray:
    """Discrete, path-wise *approximation* to the gamma-PnL expression, evaluated along each path.

    Continuous-time statement (theory).  For a seller who hedges continuously with delta computed at the implied
    volatility ``sigma`` while the price has quadratic variation d<ln S>_t = sigma_realised(t)^2 dt, to first order

        PnL ~= integral_0^T  e^{r (T - t)} * (1/2) * Gamma_t * S_t^2 * ( sigma^2 dt - d<ln S>_t )

    which equals  integral e^{r(T-t)} (1/2) Gamma_t S_t^2 (sigma^2 - sigma_realised^2) dt.  This is a statement about a
    continuously observed path; it is an identity only in that continuous-time, continuous-hedging limit.

    What this function computes (an implementation on a finite grid).  Daily observations do not give the quadratic
    variation, only a proxy for it, so the result is an approximation, not an identity.  Three discretisation choices
    are involved: (i) the realised-variance proxy for d<ln S> over a step, (ii) Gamma_t S_t^2 frozen at the start of
    each step (left endpoint), and (iii) a second-order Taylor expansion that ignores higher-order terms and the
    effect of re-hedging only at grid times.  Its accuracy improves as the grid is refined (see experiment A6 in
    ``gbm_sim.py``) and on real data it should be read as an empirical fit, not as an exact decomposition.

    * ``realised_var=None`` : path-wise version.  Each step's squared return stands in for the step's variance,
                              sum_i w_i ( sigma^2 dt_i - q_i ),  with q_i the squared return of step i
                              (convention set by ``return_convention``, see below).
    * ``realised_var=v``    : one constant realised variance per path (annualised, scalar or (M,)),
                              sum_i w_i ( sigma^2 - v ) dt_i                           [the brief's regressor form]

    where w_i = e^{r (T - t_i)} * Gamma_i * S_i^2 / 2.  Returns an array of shape (M,).

    ``return_convention`` (path-wise version only):
      "log"          q_i = (ln S_{i+1}/S_i)^2                     (default)
      "simple"       q_i = (S_{i+1}/S_i - 1)^2   so that w_i q_i = (1/2) Gamma_i (dS_i)^2, the direct Taylor term
      "log_demeaned" q_i = (ln S_{i+1}/S_i - mean log return of the path)^2
    """
    if return_convention not in RETURN_CONVENTIONS:
        raise ValueError(f"return_convention must be one of {RETURN_CONVENTIONS}")
    P = _as_paths(paths)
    M, ncol = P.shape
    r = _check_scalar("r", r)
    K = _check_per_path("K", K, M)
    sigma = _check_per_path("sigma", sigma, M)
    if realised_var is not None:
        rvv = np.asarray(realised_var, dtype=float)
        if not np.all(np.isfinite(rvv)) or np.any(rvv < 0):
            raise ValueError("realised_var must be finite and non-negative")
    if times is None:
        times = np.linspace(0.0, float(T), ncol)
    times = _check_times(times, ncol, float(T))
    dt = np.diff(times)[None, :]
    S = P[:, :-1]
    K_arr = np.broadcast_to(np.asarray(K, dtype=float), (M,))[:, None]
    sig = np.broadcast_to(np.asarray(sigma, dtype=float), (M,))[:, None]
    tau = (T - times[:-1])[None, :]
    w = np.exp(r * tau) * 0.5 * bs.gamma(S, K_arr, tau, sig, r) * S**2
    if realised_var is None:
        if return_convention == "simple":
            sq = np.square(P[:, 1:] / P[:, :-1] - 1.0)
        else:
            lr = np.diff(np.log(P), axis=1)
            if return_convention == "log_demeaned":
                lr = lr - lr.mean(axis=1, keepdims=True)
            sq = np.square(lr)
        return (w * (sig**2 * dt - sq)).sum(axis=1)
    rv = np.broadcast_to(np.asarray(realised_var, dtype=float), (M,))[:, None]
    return (w * (sig**2 - rv) * dt).sum(axis=1)
