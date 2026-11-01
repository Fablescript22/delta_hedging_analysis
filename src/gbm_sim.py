"""Part A: synthetic geometric-Brownian-motion experiments.

This module currently provides the simulation primitives (path generator and a
chunked simulate-and-hedge runner).  The experiments A1-A4 are added on top of
these in later iterations.

GBM step (exact discretisation):
    S_{t+dt} = S_t * exp( (mu - 0.5 sigma_r^2) dt + sigma_r sqrt(dt) Z ),   Z ~ N(0, 1)
"""
from __future__ import annotations

import numpy as np

from . import config as cfg
from .hedge import HedgeResult, _check_scalar, hedge_pnl


def _check_count(name: str, x) -> int:
    """Validate a size parameter: a positive whole number (bool, float with fraction, NaN all rejected)."""
    if isinstance(x, bool) or not isinstance(x, (int, np.integer, float, np.floating)):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(x, (float, np.floating)) and (not np.isfinite(x) or x != int(x)):
        raise ValueError(f"{name} must be a whole number")
    if int(x) < 1:
        raise ValueError(f"{name} must be at least 1")
    return int(x)


def make_rng(*stream: int, seed: int = cfg.SEED) -> np.random.Generator:
    """Reproducible generator.  Different ``stream`` ids give independent streams."""
    return np.random.default_rng(np.random.SeedSequence(seed, spawn_key=tuple(stream)))


def simulate_gbm_paths(
    n_paths: int,
    n_steps: int,
    S0: float,
    mu: float,
    sigma_r: float,
    T: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return array (n_paths, n_steps + 1) of GBM prices on an equally spaced grid."""
    n_paths = _check_count("n_paths", n_paths)
    n_steps = _check_count("n_steps", n_steps)
    S0 = _check_scalar("S0", S0, positive=True)
    mu = _check_scalar("mu", mu)
    sigma_r = _check_scalar("sigma_r", sigma_r, nonneg=True)
    T = _check_scalar("T", T, positive=True)
    dt = T / n_steps
    Z = rng.standard_normal((n_paths, n_steps))
    log_inc = (mu - 0.5 * sigma_r**2) * dt + sigma_r * np.sqrt(dt) * Z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_inc, axis=1)], axis=1)
    return S0 * np.exp(log_paths)


def simulate_and_hedge(
    n_paths: int,
    n_steps: int,
    *,
    sigma_r: float,
    sigma_i: float = cfg.A_SIGMA_I,
    mu: float = cfg.A_MU,
    S0: float = cfg.A_S0,
    K: float = cfg.A_K,
    T: float = cfg.A_T,
    r: float = cfg.A_R,
    cost_rate: float = 0.0,
    rng: np.random.Generator,
    max_cells: int = 4_000_000,
    keep_paths: bool = False,
):
    """Simulate GBM paths (true vol ``sigma_r``) and delta-hedge a short call sold at ``sigma_i``.

    Paths are processed in chunks so memory stays bounded for very large ``n_steps``.
    Returns (HedgeResult-like dict of concatenated arrays, optional paths).
    """
    n_paths = _check_count("n_paths", n_paths)
    n_steps = _check_count("n_steps", n_steps)
    max_cells = _check_count("max_cells", max_cells)
    _check_scalar("sigma_i", sigma_i, positive=True)
    _check_scalar("K", K, positive=True)
    _check_scalar("cost_rate", cost_rate, nonneg=True, upper=1.0)
    chunk = max(1, int(max_cells // (n_steps + 1)))
    pnl, premium, cost, real_var = [], [], [], []
    kept = []
    done = 0
    while done < n_paths:
        m = min(chunk, n_paths - done)
        P = simulate_gbm_paths(m, n_steps, S0, mu, sigma_r, T, rng)
        res: HedgeResult = hedge_pnl(P, K, sigma_i, r, T, cost_rate=cost_rate)
        pnl.append(res.pnl)
        premium.append(res.premium)
        cost.append(res.total_cost)
        # per-path realised variance (annualised, uncentred sum of squared log returns / T)
        real_var.append(np.square(np.diff(np.log(P), axis=1)).sum(axis=1) / T)
        if keep_paths:
            kept.append(P)
        done += m
    out = dict(
        pnl=np.concatenate(pnl),
        premium=np.concatenate(premium),
        total_cost=np.concatenate(cost),
        realised_var=np.concatenate(real_var),
        n_steps=n_steps,
    )
    return (out, np.concatenate(kept)) if keep_paths else (out, None)


# =============================================================================================
# Experiments A1-A5.  Every experiment draws from its own RNG stream: make_rng(experiment_id, k).
# =============================================================================================
import json  # noqa: E402

import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from . import bs  # noqa: E402
from .hedge import gamma_pnl_estimate  # noqa: E402

A3_SIGMA_R_GRID = (0.10, 0.15, 0.18, 0.20, 0.22, 0.25, 0.30)
A3_N_STEPS = 250
A4_N_GRID = (5, 10, 20, 50, 100, 250, 500, 1000)
A4_COST_RATES = (0.0005, 0.001)            # 5 and 10 basis points of traded notional
A4_LAMBDAS = (0.5, 1.0, 2.0)               # risk-aversion weights in  mean(net) - lambda * std(net)
A5_MU_GRID = (-0.20, 0.0, 0.05, 0.20, 0.40)
A6_N_GRID = (5, 10, 20, 50, 100, 250, 500)
A6_PATHS_PER_SIGMA = 4000                   # x 7 volatility settings = 28,000 paths per N


def _mean_se_std(x: np.ndarray) -> tuple[float, float, float, float]:
    n = len(x)
    sd = float(np.std(x, ddof=1))
    return float(np.mean(x)), sd / np.sqrt(n), sd, sd / np.sqrt(2.0 * (n - 1))   # mean, SE(mean), std, SE(std)


def exp_a1(n_paths: int = cfg.A_N_PATHS) -> pd.DataFrame:
    """A1 sanity: sigma_r = sigma_i, mu = r.  Mean PnL ~ 0 and the dispersion shrinks as N grows."""
    rows = []
    for k, N in enumerate((10, 100, 1000, 5000)):
        out, _ = simulate_and_hedge(n_paths, N, sigma_r=cfg.A_SIGMA_I, mu=cfg.A_R, rng=make_rng(1, k))
        m, se, sd, se_sd = _mean_se_std(out["pnl"])
        rows.append(dict(N=N, paths=n_paths, mean_pnl=m, se_mean=se, z_mean=m / se, std_pnl=sd, se_std=se_sd,
                         premium=float(out["premium"][0])))
    return pd.DataFrame(rows)


def exp_a2(n_paths: int = cfg.A_N_PATHS):
    """A2 frequency: std of hedging PnL vs N and the log-log slope."""
    rows, samples = [], {}
    for k, N in enumerate(cfg.A_N_GRID):
        out, _ = simulate_and_hedge(n_paths, N, sigma_r=cfg.A_SIGMA_I, mu=cfg.A_R, rng=make_rng(2, k))
        m, se, sd, se_sd = _mean_se_std(out["pnl"])
        rows.append(dict(N=N, paths=n_paths, mean_pnl=m, se_mean=se, std_pnl=sd, se_std=se_sd,
                         std_over_premium=sd / float(out["premium"][0]), std_times_sqrtN=sd * np.sqrt(N)))
        if N in (5, 20, 100, 500):
            samples[N] = out["pnl"]
    df = pd.DataFrame(rows)
    lx, ly = np.log(df["N"].to_numpy(float)), np.log(df["std_pnl"].to_numpy())
    fit_all = stats.linregress(lx, ly)
    sub = df["N"].to_numpy() >= 20
    fit_ge20 = stats.linregress(lx[sub], ly[sub])
    fits = dict(
        slope_all=fit_all.slope, slope_all_se=fit_all.stderr, intercept_all=fit_all.intercept, r2_all=fit_all.rvalue**2,
        slope_N_ge_20=fit_ge20.slope, slope_N_ge_20_se=fit_ge20.stderr, r2_N_ge_20=fit_ge20.rvalue**2,
        intercept_N_ge_20=fit_ge20.intercept,
    )
    # what a 1/N law would look like, for contrast: ratio of std at N=5 vs N=500 (100x more rebalances)
    s5, s500 = df.loc[df.N == 5, "std_pnl"].item(), df.loc[df.N == 500, "std_pnl"].item()
    fits["std_ratio_N5_over_N500"] = s5 / s500
    fits["ratio_if_1_over_sqrtN"] = float(np.sqrt(500 / 5))
    fits["ratio_if_1_over_N"] = 500 / 5
    return df, fits, samples


def expected_gamma_s2(t, sigma_r: float, sigma_i: float, *, mu: float, S0: float, K: float, T: float, r: float):
    """Closed form for E[ S_t^2 * Gamma(S_t, T-t; sigma_i) ] when S_t is lognormal with drift mu, vol sigma_r.

    With S_t = exp(m + s z), z ~ N(0,1), m = ln S0 + (mu - sigma_r^2/2) t, s = sigma_r sqrt(t), tau = T - t:
        S^2 Gamma = S n(d1) / (sigma_i sqrt(tau)),   d1 = a + b z,
        a = (m - ln K + (r + sigma_i^2/2) tau) / (sigma_i sqrt(tau)),   b = s / (sigma_i sqrt(tau)).
    Completing the square in the Gaussian integral gives
        E = exp( m + (s^2 - a^2 - 2 a b s) / (2 (1 + b^2)) ) / ( sigma_i sqrt(tau) * sqrt(2 pi (1 + b^2)) ).
    """
    t = np.asarray(t, dtype=float)
    tau = T - t
    m = np.log(S0) + (mu - 0.5 * sigma_r**2) * t
    s = sigma_r * np.sqrt(t)
    a = (m - np.log(K) + (r + 0.5 * sigma_i**2) * tau) / (sigma_i * np.sqrt(tau))
    b = s / (sigma_i * np.sqrt(tau))
    expo = m + (s**2 - a**2 - 2.0 * a * b * s) / (2.0 * (1.0 + b**2))
    return np.exp(expo) / (sigma_i * np.sqrt(tau) * np.sqrt(2.0 * np.pi * (1.0 + b**2)))


def theory_mean_pnl(sigma_r: float, sigma_i: float = cfg.A_SIGMA_I, *, mu: float = cfg.A_MU, S0: float = cfg.A_S0,
                    K: float = cfg.A_K, T: float = cfg.A_T, r: float = cfg.A_R, n_u: int = 2000) -> float:
    """Continuous-time expected hedging PnL (money at T) from the gamma formula.

        E[PnL] = (sigma_i^2 - sigma_r^2) * int_0^T e^{r(T-t)} (1/2) E[ Gamma_t S_t^2 ] dt,   S_t lognormal(mu, sigma_r)

    The inner expectation is the closed form above; the time integral uses Gauss-Legendre in u with
    t = T - u^2 (a smooth integrand).  (A first version used Gauss-Hermite for the inner expectation; it did
    not converge near expiry, where Gamma is a spike - found by a node-count convergence check, see README.)
    """
    x, wx = np.polynomial.legendre.leggauss(n_u)
    u = 0.5 * np.sqrt(T) * (x + 1.0)
    wu = 0.5 * np.sqrt(T) * wx
    t, tau, w_t = T - u**2, u**2, 2.0 * u * wu
    inner = 0.5 * np.exp(r * tau) * expected_gamma_s2(t, sigma_r, sigma_i, mu=mu, S0=S0, K=K, T=T, r=r)
    return float((sigma_i**2 - sigma_r**2) * np.sum(inner * w_t))


def exp_a3(n_paths: int = cfg.A_N_PATHS, n_steps: int = A3_N_STEPS):
    """A3 vol gap: sigma_r below / equal / above sigma_i.  Mean PnL vs (sigma_i^2 - sigma_r^2)."""
    rows, pooled = [], dict(gap_hat=[], pnl=[], est_path=[], est_const=[])
    for k, sr in enumerate(A3_SIGMA_R_GRID):
        out, P = simulate_and_hedge(n_paths, n_steps, sigma_r=sr, mu=cfg.A_MU, rng=make_rng(3, k), keep_paths=True)
        pnl, rv = out["pnl"], out["realised_var"]
        est_path = gamma_pnl_estimate(P, cfg.A_K, cfg.A_SIGMA_I, cfg.A_R, cfg.A_T)
        est_const = gamma_pnl_estimate(P, cfg.A_K, cfg.A_SIGMA_I, cfg.A_R, cfg.A_T, realised_var=rv)
        m, se, sd, _ = _mean_se_std(pnl)
        th = theory_mean_pnl(sr)
        gap = cfg.A_SIGMA_I**2 - sr**2
        rows.append(dict(sigma_r=sr, gap=gap, mean_pnl=m, se_mean=se, std_pnl=sd, theory_mean=th,
                         diff_sim_minus_theory=m - th, diff_in_se=(m - th) / se,
                         sign_ok=bool(np.sign(m) == np.sign(gap)) if gap != 0 else abs(m) < 3 * se,
                         frac_paths_profit=float(np.mean(pnl > 0)),
                         r2_pnl_on_gaphat_within=float(stats.linregress(cfg.A_SIGMA_I**2 - rv, pnl).rvalue ** 2)))
        pooled["gap_hat"].append(cfg.A_SIGMA_I**2 - rv); pooled["pnl"].append(pnl)
        pooled["est_path"].append(est_path); pooled["est_const"].append(est_const)
    df = pd.DataFrame(rows)
    # (i) across the 7 settings: mean PnL on the fixed variance gap
    f_mean = stats.linregress(df["gap"], df["mean_pnl"])
    # (ii) pooled per-path: PnL on (sigma_i^2 - realised variance of that path)  -- same regression as Part B
    gap_hat, pnl = np.concatenate(pooled["gap_hat"]), np.concatenate(pooled["pnl"])
    f_path = stats.linregress(gap_hat, pnl)
    est_path, est_const = np.concatenate(pooled["est_path"]), np.concatenate(pooled["est_const"])
    summary = dict(
        mean_on_gap_slope=f_mean.slope, mean_on_gap_slope_se=f_mean.stderr, mean_on_gap_r2=f_mean.rvalue**2,
        pooled_pnl_on_gaphat_slope=f_path.slope, pooled_pnl_on_gaphat_slope_se=f_path.stderr,
        pooled_pnl_on_gaphat_intercept=f_path.intercept, pooled_pnl_on_gaphat_r2=f_path.rvalue**2,
        r2_pnl_vs_gamma_estimate_pathwise=float(np.corrcoef(pnl, est_path)[0, 1] ** 2),
        r2_pnl_vs_gamma_estimate_constvar=float(np.corrcoef(pnl, est_const)[0, 1] ** 2),
        n_pooled=int(len(pnl)), n_steps=n_steps,
        # why pooled R^2 > within-world R^2: the spread of the regressor (sigma_i^2 - realised variance)
        gaphat_std_pooled=float(np.std(gap_hat, ddof=1)),
        gaphat_std_within_mean=float(np.mean([np.std(g, ddof=1) for g in pooled["gap_hat"]])),
        within_r2_min=float(df["r2_pnl_on_gaphat_within"].min()), within_r2_max=float(df["r2_pnl_on_gaphat_within"].max()),
    )
    rng = np.random.default_rng(cfg.SEED)
    sel = rng.choice(len(pnl), size=min(5000, len(pnl)), replace=False)
    scatter = dict(gap_hat=gap_hat[sel], pnl=pnl[sel], est_path=est_path[sel])
    return df, summary, scatter


def exp_a4(n_paths: int = cfg.A_N_PATHS) -> pd.DataFrame:
    """A4 transaction costs.  sigma_r = sigma_i, mu = r, proportional cost c per unit traded notional."""
    rows = []
    for k, N in enumerate(A4_N_GRID):
        rng = make_rng(4, k)
        chunk = max(1, int(4_000_000 // (N + 1)))
        acc = {c: dict(net=[], cost=[]) for c in A4_COST_RATES}
        gross = []
        done = 0
        while done < n_paths:
            m = min(chunk, n_paths - done)
            P = simulate_gbm_paths(m, N, cfg.A_S0, cfg.A_R, cfg.A_SIGMA_I, cfg.A_T, rng)
            gross.append(hedge_pnl(P, cfg.A_K, cfg.A_SIGMA_I, cfg.A_R, cfg.A_T).pnl)
            for c in A4_COST_RATES:
                res = hedge_pnl(P, cfg.A_K, cfg.A_SIGMA_I, cfg.A_R, cfg.A_T, cost_rate=c)
                acc[c]["net"].append(res.pnl); acc[c]["cost"].append(res.total_cost)
            done += m
        g = np.concatenate(gross)
        for c in A4_COST_RATES:
            net, cost = np.concatenate(acc[c]["net"]), np.concatenate(acc[c]["cost"])
            rows.append(dict(N=N, cost_rate=c, mean_gross=g.mean(), mean_cost=cost.mean(), se_cost=cost.std(ddof=1) / np.sqrt(n_paths),
                             mean_net=net.mean(), std_net=net.std(ddof=1),
                             **{f"objective_lambda_{lam}": net.mean() - lam * net.std(ddof=1) for lam in A4_LAMBDAS}))
    return pd.DataFrame(rows)


def a4_optima(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c, g in df.groupby("cost_rate"):
        for lam in A4_LAMBDAS:
            col = f"objective_lambda_{lam}"
            best = g.loc[g[col].idxmax()]
            rows.append(dict(cost_rate=c, risk_aversion_lambda=lam, best_N=int(best["N"]), objective_at_best=best[col]))
    return pd.DataFrame(rows)


def exp_a5(n_paths: int = cfg.A_N_PATHS, n_steps: int = 250) -> pd.DataFrame:
    """A5 (extra) drift: does the real-world drift mu matter for a hedged option?  sigma_r = sigma_i."""
    rows = []
    for k, mu in enumerate(A5_MU_GRID):
        out, _ = simulate_and_hedge(n_paths, n_steps, sigma_r=cfg.A_SIGMA_I, mu=mu, rng=make_rng(5, k))
        m, se, sd, se_sd = _mean_se_std(out["pnl"])
        rows.append(dict(mu=mu, mean_pnl=m, se_mean=se, std_pnl=sd, se_std=se_sd))
    return pd.DataFrame(rows)


def exp_a6(n_paths_per_sigma: int = A6_PATHS_PER_SIGMA) -> pd.DataFrame:
    """A6 (extra): how well does the discrete gamma estimate approximate the actual hedging PnL as sampling gets finer?

    The gamma-PnL expression is exact only for a continuously observed path and continuous hedging; on a grid of N
    steps it is an approximation.  For each N, paths from all seven A3 volatility settings (pooled, mu = r) are
    hedged and compared with three estimates: path-wise with squared log returns, path-wise with squared simple
    returns, and the constant-variance form.  Absolute RMSE of (estimate - PnL) is the clean convergence measure;
    pooled R^2 is also shown (it is flattered by the wide spread of gaps across settings, see the A3 discussion), as
    is R^2 within the single setting sigma_r = sigma_i, where the gap carries no information.
    """
    rows = []
    for i, n in enumerate(A6_N_GRID):
        pnl_all, est_log, est_simple, est_const, setting = [], [], [], [], []
        for k, sr in enumerate(A3_SIGMA_R_GRID):
            out, P = simulate_and_hedge(n_paths_per_sigma, n, sigma_r=sr, mu=cfg.A_R, rng=make_rng(6, i, k), keep_paths=True)
            pnl_all.append(out["pnl"])
            est_log.append(gamma_pnl_estimate(P, cfg.A_K, cfg.A_SIGMA_I, cfg.A_R, cfg.A_T, return_convention="log"))
            est_simple.append(gamma_pnl_estimate(P, cfg.A_K, cfg.A_SIGMA_I, cfg.A_R, cfg.A_T, return_convention="simple"))
            est_const.append(gamma_pnl_estimate(P, cfg.A_K, cfg.A_SIGMA_I, cfg.A_R, cfg.A_T, realised_var=out["realised_var"]))
            setting.append(np.full(len(out["pnl"]), sr))
        pnl, setting = np.concatenate(pnl_all), np.concatenate(setting)
        premium = float(out["premium"][0])
        row = dict(N=n, paths=len(pnl), premium=premium)
        for name, e in (("log", est_log), ("simple", est_simple), ("const", est_const)):
            e = np.concatenate(e)
            err = e - pnl
            row[f"rmse_{name}"] = float(np.sqrt((err**2).mean()))
            row[f"rmse_{name}_pct_premium"] = float(np.sqrt((err**2).mean()) / premium * 100)
            row[f"r2_pooled_{name}"] = float(np.corrcoef(e, pnl)[0, 1] ** 2)
            m = np.isclose(setting, cfg.A_SIGMA_I)
            row[f"r2_within_equal_vol_{name}"] = float(np.corrcoef(e[m], pnl[m])[0, 1] ** 2)
        rows.append(row)
    df = pd.DataFrame(rows)
    # log-log slope of the RMSE of the path-wise (log) estimate against N
    x = np.log(df["N"].to_numpy(float))
    df.attrs["rmse_loglog_slope_log"] = float(np.polyfit(x, np.log(df["rmse_log"].to_numpy()), 1)[0])
    return df


def run_part_a(save: bool = True) -> dict:
    """Run A1-A5 with the fixed seed; write tidy CSV/JSON files to results/; return everything."""
    a1 = exp_a1()
    a2, a2_fits, a2_samples = exp_a2()
    a3, a3_sum, a3_scatter = exp_a3()
    a4 = exp_a4()
    a4_opt = a4_optima(a4)
    a5 = exp_a5()
    a6 = exp_a6()
    res = dict(a1=a1, a2=a2, a2_fits=a2_fits, a2_samples=a2_samples, a3=a3, a3_summary=a3_sum, a3_scatter=a3_scatter,
               a4=a4, a4_optima=a4_opt, a5=a5, a6=a6)
    if save:
        cfg.RESULTS_DIR.mkdir(exist_ok=True)
        a1.to_csv(cfg.RESULTS_DIR / "partA_A1_sanity.csv", index=False)
        a2.to_csv(cfg.RESULTS_DIR / "partA_A2_frequency.csv", index=False)
        a3.to_csv(cfg.RESULTS_DIR / "partA_A3_volgap.csv", index=False)
        a4.to_csv(cfg.RESULTS_DIR / "partA_A4_costs.csv", index=False)
        a4_opt.to_csv(cfg.RESULTS_DIR / "partA_A4_optima.csv", index=False)
        a5.to_csv(cfg.RESULTS_DIR / "partA_A5_drift.csv", index=False)
        a6.to_csv(cfg.RESULTS_DIR / "partA_A6_estimator_convergence.csv", index=False)
        (cfg.RESULTS_DIR / "partA_summary.json").write_text(json.dumps(
            dict(seed=cfg.SEED, paths_per_setting=cfg.A_N_PATHS, a2_fits=a2_fits, a3_summary=a3_sum,
                 a6_rmse_loglog_slope=a6.attrs['rmse_loglog_slope_log']), indent=2, default=float))
    return res
