"""Part B extended empirical research: effect-size intervals, chronological validation, market regimes and
alternative volatility estimators.

Everything here is computed from the same 2,629 windows as the headline backtest (no new data).  Every
choice that could be tuned (split dates, regime thresholds, estimator list, decay constant) is fixed in this
file before the tables are produced and is written next to the results by ``run_part_b``.

Honesty notes that also appear in the README:
  * The full-sample headline numbers were known when the splits and regimes below were defined, so these
    are *not* blind out-of-sample tests; they are a discipline check (does a rule fitted on the past hold up
    on later windows?) rather than proof of predictive value.
  * Windows overlap (stride 1 day, length 21), so every interval uses HAC or block-bootstrap methods and the
    split periods drop any window that straddles a boundary (purging).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import config as cfg
from . import real_data as rd
from .backtest import Backtest, check_bootstrap, check_series, mean_hac, ols_hac

# ----------------------------------------------------------------------------------------- fixed choices
CI_BLOCK, CI_BOOT = 42, 4000                     # block length (2 windows) and resamples for effect-size CIs
CI_STREAM = (14, 0)

SPLITS = {   # name -> (calibration, validation, test), inclusive dates; windows must lie wholly inside a period
    "primary": (("2016-01-01", "2021-12-31"), ("2022-01-01", "2023-12-31"), ("2024-01-01", "2026-12-31")),
    "crash_held_out": (("2016-01-01", "2019-12-31"), ("2020-01-01", "2022-12-31"), ("2023-01-01", "2026-12-31")),
}

DRAWDOWN_CUT = -0.05          # regime thresholds use only information available at t0 (trailing, no look-ahead)
MOMENTUM_DAYS = 63            # about one quarter of trading days
# Models whose inputs are all known on the day the option is sold: the calibration mean, VIX, and VIX against the
# volatility of the 21 trading days BEFORE t0.  Every other model uses the window's own realised path (ex-post).
EX_ANTE_MODELS = ("constant", "vix_level", "exante_var_gap", "exante_vol_gap")
EWMA_LAMBDA = 0.94            # RiskMetrics daily decay
ESTIMATORS = ("close_to_close", "close_to_close_centred", "ewma_0.94", "bipower", "parkinson", "garman_klass",
              "rogers_satchell", "parkinson_plus_gap", "garman_klass_plus_gap", "rogers_satchell_plus_gap", "yang_zhang")
# What each estimator measures.  A daily-close hedger experiences the overnight move (close to next open) as well
# as the move within the day, so only estimators that include it are like-for-like with the hedge's own risk.
ESTIMATOR_TYPE = {
    "close_to_close": "close-to-close", "close_to_close_centred": "close-to-close", "ewma_0.94": "close-to-close",
    "bipower": "close-to-close",
    "parkinson": "range, intraday only (no overnight gap)", "garman_klass": "range, intraday only (no overnight gap)",
    "rogers_satchell": "range, intraday only (no overnight gap)",
    "parkinson_plus_gap": "range + overnight gap", "garman_klass_plus_gap": "range + overnight gap",
    "rogers_satchell_plus_gap": "range + overnight gap", "yang_zhang": "range + overnight gap",
}


# ----------------------------------------------------------------------------------------- bootstrap CIs
def _circular_blocks(n: int, block: int, rng) -> np.ndarray:
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, n_blocks)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def block_bootstrap_ci(stat, arrays, block: int, n_boot: int, rng, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval of ``stat(*arrays)`` under a circular moving-block bootstrap of the window index."""
    if not callable(stat):
        raise ValueError("stat must be callable")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    if len(arrays) == 0:
        raise ValueError("arrays must not be empty")
    arrays = [check_series(f"arrays[{i}]", a, 2) for i, a in enumerate(arrays)]
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("all arrays must have the same length")
    check_bootstrap(n, block, n_boot)
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = _circular_blocks(n, block, rng)
        out[b] = stat(*[a[idx] for a in arrays])
    lo, hi = np.percentile(out, [50 * (1 - level), 50 * (1 + level)])
    return float(lo), float(hi)


MC_REPS = 5                      # independent repeats of the effect-size bootstrap for the Monte Carlo error table


def effect_size_mc_error(bt: Backtest, lag: int, make_rng, reps: int = MC_REPS) -> pd.DataFrame:
    """How much do the bootstrap interval ENDS move when the bootstrap is simply repeated with new random numbers?

    Each repeat redraws all ``CI_BOOT`` resamples from its own RNG stream (14, 2+k).  The table gives, per effect,
    the mean and the spread (sd and range) of the lower and upper bootstrap bounds over the repeats; the HAC
    interval is deterministic and has no Monte Carlo error.  The spread is Monte Carlo error only, not sampling
    uncertainty, which the interval itself describes.
    """
    if isinstance(reps, bool) or not isinstance(reps, int) or reps < 2:
        raise ValueError("reps must be an integer >= 2")
    runs = []
    for k in range(reps):
        t = effect_size_cis(bt, lag, lambda *stream, k=k: make_rng(14, 2 + k))
        runs.append(t[["effect", "boot_ci_lo", "boot_ci_hi"]].assign(rep=k))
    all_ = pd.concat(runs)
    g = all_.groupby("effect", sort=False)
    out = pd.DataFrame(dict(
        reps=reps, resamples=CI_BOOT,
        lo_mean=g.boot_ci_lo.mean(), lo_sd=g.boot_ci_lo.std(ddof=1), lo_range=g.boot_ci_lo.max() - g.boot_ci_lo.min(),
        hi_mean=g.boot_ci_hi.mean(), hi_sd=g.boot_ci_hi.std(ddof=1), hi_range=g.boot_ci_hi.max() - g.boot_ci_hi.min(),
    )).reset_index()
    width = out.hi_mean - out.lo_mean
    out["max_bound_sd_over_width"] = np.maximum(out.lo_sd, out.hi_sd) / width
    return out


def _slope(x, y):
    xc = x - x.mean()
    return float(xc @ (y - y.mean()) / (xc @ xc))


def _r2(x, y):
    return float(np.corrcoef(x, y)[0, 1] ** 2)


def effect_size_cis(bt: Backtest, lag: int, make_rng) -> pd.DataFrame:
    """Headline effect sizes with 95% intervals (HAC normal interval where one exists, plus block bootstrap)."""
    rng = make_rng(*CI_STREAM)
    pnl, gap, vol_gap = bt.pnl * 100, bt.gap, bt.ws.vix_pct - bt.rv * 100
    profit = (pnl > 0).astype(float)
    rows = []

    def add(name, unit, est, hac_se, boot, note=""):
        lo_h, hi_h = (est - 1.96 * hac_se, est + 1.96 * hac_se) if hac_se is not None else (np.nan, np.nan)
        rows.append(dict(effect=name, unit=unit, estimate=est, hac_se=hac_se if hac_se is not None else np.nan,
                         hac_ci_lo=lo_h, hac_ci_hi=hi_h, boot_ci_lo=boot[0], boot_ci_hi=boot[1], note=note))

    m, se = mean_hac(pnl, lag)
    add("mean hedging PnL", "% of S0 per 21-day window", m, se,
        block_bootstrap_ci(lambda a: a.mean(), [pnl], CI_BLOCK, CI_BOOT, rng))
    pm, pse = mean_hac(profit, lag)
    add("share of profitable windows", "fraction", pm, pse,
        block_bootstrap_ci(lambda a: a.mean(), [profit], CI_BLOCK, CI_BOOT, rng))
    sh = float(pnl.mean() / pnl.std(ddof=1))
    add("PnL per unit of PnL risk (mean/std per window)", "ratio", sh, None,
        block_bootstrap_ci(lambda a: a.mean() / a.std(ddof=1), [pnl], CI_BLOCK, CI_BOOT, rng),
        "not annualised; windows overlap")
    vg, vse = mean_hac(vol_gap, lag)
    add("volatility risk premium (VIX - realised vol)", "vol points", vg, vse,
        block_bootstrap_ci(lambda a: a.mean(), [vol_gap], CI_BLOCK, CI_BOOT, rng))
    reg = ols_hac(gap, bt.pnl, lag)
    add("slope of PnL on variance gap", "PnL (fraction of S0) per unit variance gap", reg["slope"], reg["se_slope_hac"],
        block_bootstrap_ci(_slope, [gap, bt.pnl], CI_BLOCK, CI_BOOT, rng))
    add("R-squared of PnL on variance gap", "fraction of variance", reg["r2"], None,
        block_bootstrap_ci(_r2, [gap, bt.pnl], CI_BLOCK, CI_BOOT, rng))
    return pd.DataFrame(rows)


def slope_bootstrap_vs_crash_share(bt: Backtest, crash: np.ndarray, make_rng) -> dict:
    """Why is the bootstrap interval of the slope so wide above the estimate?  Relate each resample's slope to the
    share of March-2020 crash windows that the resample happens to contain."""
    rng = make_rng(*CI_STREAM[:1], 1)
    n = len(crash)
    c = crash.astype(float)
    slopes, share = np.empty(CI_BOOT), np.empty(CI_BOOT)
    for b in range(CI_BOOT):
        idx = _circular_blocks(n, CI_BLOCK, rng)
        slopes[b] = _slope(bt.gap[idx], bt.pnl[idx])
        share[b] = c[idx].mean()
    few = share < 0.005
    return dict(corr_slope_vs_crash_share=float(np.corrcoef(slopes, share)[0, 1]), crash_share_in_data=float(c.mean()),
                share_of_resamples_with_under_half_percent_crash=float(few.mean()),
                median_slope_those=float(np.median(slopes[few])) if few.any() else float("nan"),
                median_slope_others=float(np.median(slopes[~few])))


# ----------------------------------------------------------------------------------------- regimes
def regime_labels(data: rd.RealData, ws: rd.WindowSet) -> np.ndarray:
    """'calm' / 'drawdown' / 'recovery' from trailing NIFTY information at t0 (NaN where < 63 prior days exist).

    drawdown_t0 = S_t0 / (running maximum of the NIFTY close since the start of the sample) - 1
    momentum_t0 = S_t0 / S_{t0 - 63 trading days} - 1
      calm      : drawdown > -5%
      drawdown  : drawdown <= -5% and momentum < 0   (still falling or not yet bouncing)
      recovery  : drawdown <= -5% and momentum >= 0  (below the peak but rising over the last quarter)
    """
    close = data.nifty["Close"]
    dd = close / close.cummax() - 1.0
    mom = close / close.shift(MOMENTUM_DAYS) - 1.0
    d, m = dd.loc[ws.t0].to_numpy(), mom.loc[ws.t0].to_numpy()
    lab = np.full(len(d), "", dtype=object)
    lab[d > DRAWDOWN_CUT] = "calm"
    lab[(d <= DRAWDOWN_CUT) & (m < 0)] = "drawdown"
    lab[(d <= DRAWDOWN_CUT) & (m >= 0)] = "recovery"
    lab[np.isnan(m)] = ""
    return lab


def regime_table(bt: Backtest, labels: np.ndarray, lag: int) -> pd.DataFrame:
    rows = []
    for name in ("calm", "drawdown", "recovery"):
        m = labels == name
        n = int(m.sum())
        if n < 40:
            rows.append(dict(regime=name, windows=n))
            continue
        pnl, vg = bt.pnl[m] * 100, bt.ws.vix_pct[m] - bt.rv[m] * 100
        mean, se = mean_hac(pnl, min(lag, n // 4))
        reg = ols_hac(bt.gap[m], bt.pnl[m], min(lag, n // 4))
        rows.append(dict(regime=name, windows=n, share=n / float((labels != "").sum()), mean_pnl_pct=mean,
                         mean_pnl_ci_lo=mean - 1.96 * se, mean_pnl_ci_hi=mean + 1.96 * se,
                         std_pnl_pct=float(pnl.std(ddof=1)), worst_pnl_pct=float(pnl.min()),
                         frac_profit=float((pnl > 0).mean()), mean_vix=float(bt.ws.vix_pct[m].mean()),
                         mean_rv_pct=float(bt.rv[m].mean() * 100), vol_gap_points=float(vg.mean()),
                         slope=reg["slope"], slope_se_hac=reg["se_slope_hac"], r2=reg["r2"]))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------- estimators
def window_ohlc(data: rd.RealData, ws: rd.WindowSet) -> dict[str, np.ndarray]:
    """O/H/L/C of the 21 trading days that follow each t0, shape (M, W)."""
    nf = data.nifty
    pos = pd.Series(np.arange(len(nf)), index=nf.index).loc[ws.t0].to_numpy()
    idx = pos[:, None] + 1 + np.arange(ws.window)[None, :]
    return {c: nf[c.capitalize()].to_numpy()[idx] for c in ("open", "high", "low", "close")}


def vol_estimators(data: rd.RealData, ws: rd.WindowSet) -> dict[str, np.ndarray]:
    """Annualised volatility (decimal) of each window under alternative estimators."""
    W, D = ws.window, cfg.TRADING_DAYS_PER_YEAR
    o = window_ohlc(data, ws)
    r = np.diff(np.log(ws.paths), axis=1)                                   # (M, W) daily close-to-close log returns
    out = {}
    out["close_to_close"] = np.sqrt(D / W * (r**2).sum(1))
    out["close_to_close_centred"] = np.sqrt(D * r.var(axis=1, ddof=1))
    w = EWMA_LAMBDA ** np.arange(W - 1, -1, -1)
    out["ewma_0.94"] = np.sqrt(D * ((r**2) * (w / w.sum())).sum(1))
    out["bipower"] = np.sqrt(D * (math.pi / 2) * (np.abs(r[:, 1:]) * np.abs(r[:, :-1])).sum(1) / (W - 1))
    hl, co = np.log(o["high"] / o["low"]), np.log(o["close"] / o["open"])
    out["parkinson"] = np.sqrt(D * (hl**2).mean(1) / (4 * math.log(2)))
    out["garman_klass"] = np.sqrt(D * (0.5 * hl**2 - (2 * math.log(2) - 1) * co**2).mean(1))
    out["rogers_satchell"] = np.sqrt(D * (np.log(o["high"] / o["close"]) * np.log(o["high"] / o["open"])
                                          + np.log(o["low"] / o["close"]) * np.log(o["low"] / o["open"])).mean(1))
    # Gap-inclusive versions: add the (un-centred, like the close-to-close baseline) squared overnight log move
    # log(open_t / close_{t-1}) to each intraday estimator, so they measure close-to-close variance by a different route.
    gap2 = (np.log(o["open"] / ws.paths[:, :-1]) ** 2).mean(1)
    for base in ("parkinson", "garman_klass", "rogers_satchell"):
        out[f"{base}_plus_gap"] = np.sqrt(out[base] ** 2 + D * gap2)
    # Yang-Zhang (2000): overnight variance + k * open-to-close variance + (1 - k) * Rogers-Satchell, with the
    # variances taken around their sample means and k minimising the estimator's variance.
    gap = np.log(o["open"] / ws.paths[:, :-1])
    oc = np.log(o["close"] / o["open"])
    rs_var = (np.log(o["high"] / o["close"]) * np.log(o["high"] / o["open"])
              + np.log(o["low"] / o["close"]) * np.log(o["low"] / o["open"])).mean(1)
    k = 0.34 / (1.34 + (W + 1) / (W - 1))
    out["yang_zhang"] = np.sqrt(D * (gap.var(axis=1, ddof=1) + k * oc.var(axis=1, ddof=1) + (1 - k) * rs_var))
    for k_, v in out.items():
        if not np.all(np.isfinite(v)):
            raise ValueError(f"estimator {k_} produced non-finite values")
    return out


def estimator_comparison(data: rd.RealData, ws: rd.WindowSet, bt: Backtest, lag: int) -> tuple[pd.DataFrame, dict]:
    est = vol_estimators(data, ws)
    base = est["close_to_close"]
    vix = ws.vix_pct
    rows = []
    for name in ESTIMATORS:
        rv = est[name]
        gap = ws.sigma_implied**2 - rv**2
        vg = vix - rv * 100
        mean_vg, se_vg = mean_hac(vg, lag)
        reg = ols_hac(gap, bt.pnl, lag)
        rows.append(dict(estimator=name, type=ESTIMATOR_TYPE[name], mean_rv_pct=float(rv.mean() * 100), mean_vol_gap_points=mean_vg,
                         se_vol_gap_hac=se_vg, frac_rv_below_vix=float((vg > 0).mean()),
                         corr_with_close_to_close=float(np.corrcoef(rv, base)[0, 1]),
                         slope=reg["slope"], slope_se_hac=reg["se_slope_hac"], slope_t=reg["slope"] / reg["se_slope_hac"],
                         r2=reg["r2"]))
    df = pd.DataFrame(rows)
    summ = dict(
        n_estimators=len(df),
        vol_gap_range_points=[float(df.mean_vol_gap_points.min()), float(df.mean_vol_gap_points.max())],
        r2_range=[float(df.r2.min()), float(df.r2.max())],
        slope_range=[float(df.slope.min()), float(df.slope.max())],
        all_vrp_positive=bool((df.mean_vol_gap_points > 0).all()),
        all_vrp_significant=bool((df.mean_vol_gap_points / df.se_vol_gap_hac > 1.96).all()),
        all_slopes_significant=bool((df.slope_t.abs() > 1.96).all()),
        best_r2_estimator=str(df.loc[df.r2.idxmax(), "estimator"]),
    )
    for label, key in (("close-to-close", "close_to_close"), ("range, intraday only (no overnight gap)", "range_no_gap"),
                       ("range + overnight gap", "range_with_gap")):
        sub = df[df["type"] == label]
        summ[f"vrp_range_{key}"] = [float(sub.mean_vol_gap_points.min()), float(sub.mean_vol_gap_points.max())]
        summ[f"r2_range_{key}"] = [float(sub.r2.min()), float(sub.r2.max())]
    return df, summ


# ----------------------------------------------------------------------------------------- chronological split
def _period_mask(ws: rd.WindowSet, lo: str, hi: str) -> np.ndarray:
    return np.asarray((ws.t0 >= pd.Timestamp(lo)) & (ws.t_end <= pd.Timestamp(hi)))


def _fit(x, y):
    X = np.column_stack([np.ones(len(x)), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta


def exante_features(data: rd.RealData, ws: rd.WindowSet) -> dict[str, np.ndarray]:
    """Features known at t0 only: VIX against the trailing realised volatility of the 21 trading days up to t0.

    Uses closes up to and including t0 and nothing later (tested by truncating the data).  The first windows have
    fewer than 21 trailing returns and get NaN; the split code leaves them out of the ex-ante models.
    """
    W = ws.window
    close = data.nifty["Close"].to_numpy()
    pos = pd.Series(np.arange(len(close)), index=data.nifty.index).loc[ws.t0].to_numpy()
    trailing = np.full(len(pos), np.nan)
    ok = pos >= W
    idx = pos[ok][:, None] - np.arange(W, -1, -1)[None, :]            # closes t0-W ... t0
    trailing[ok] = np.sqrt(cfg.TRADING_DAYS_PER_YEAR / W * (np.diff(np.log(close[idx]), axis=1) ** 2).sum(1))
    return {"trailing_rv": trailing, "exante_var_gap": ws.sigma_implied**2 - trailing**2,
            "exante_vol_gap": ws.sigma_implied - trailing}


def chronological_split(ws: rd.WindowSet, bt: Backtest, est: dict[str, np.ndarray],
                        exante: dict[str, np.ndarray] | None = None) -> tuple[pd.DataFrame, dict]:
    """Fit on the calibration period, choose on validation, report once on the test period.

    Models (all predict the window's hedging PnL as a fraction of S0, parameters fitted on calibration only):
      constant              : the calibration mean
      vix_level             : PnL on VIX_t0 (what is known when the option is sold)
      gap_<estimator>       : PnL on (VIX^2 - realised variance) using that volatility estimator
      gamma_estimate        : PnL on the path-wise discrete gamma-PnL approximation
      exante_var_gap / exante_vol_gap : PnL on VIX^2 - trailing_RV^2 (or VIX - trailing_RV), where trailing_RV is the
                              realised volatility of the 21 trading days BEFORE t0 (known when the option is sold)

    Two tracks, never mixed in a claim.  EX-ANTE models (``EX_ANTE_MODELS``) use only information available at t0, so
    only they can say anything about prediction.  EX-POST models (gap_*, gamma_estimate) use the window's own future
    path: they explain the PnL after the fact (a mechanism check) and would not be available when selling the option.
    Out-of-sample R^2 = 1 - SSE_model / SSE_constant, where the constant is the calibration mean (can be negative).
    """
    feats = {"constant": None, "vix_level": ws.sigma_implied,
             **{f"gap_{k}": ws.sigma_implied**2 - v**2 for k, v in est.items() if k in ("close_to_close", "bipower", "parkinson")},
             "gamma_estimate": bt.est}
    if exante is not None:
        feats.update(exante_var_gap=exante["exante_var_gap"], exante_vol_gap=exante["exante_vol_gap"])
    rows, chosen = [], {}
    for sname, (cal, val, tst) in SPLITS.items():
        masks = {"calibration": _period_mask(ws, *cal), "validation": _period_mask(ws, *val), "test": _period_mask(ws, *tst)}
        in_any = masks["calibration"] | masks["validation"] | masks["test"]
        purged = int((~in_any).sum())
        y = bt.pnl
        c = masks["calibration"]
        const = y[c].mean()
        scores = {}
        for model, x in feats.items():
            fin = np.ones(len(y), bool) if x is None else np.isfinite(x)      # early windows lack a trailing history
            beta = None if x is None else _fit(x[c & fin], y[c & fin])
            for role in ("calibration", "validation", "test"):
                m = masks[role] & fin
                pred = np.full(m.sum(), const) if beta is None else beta[0] + beta[1] * x[m]
                sse = float(((y[m] - pred) ** 2).sum())
                sse_c = float(((y[m] - const) ** 2).sum())
                rows.append(dict(split=sname, model=model, role=role, track="ex-ante" if model in EX_ANTE_MODELS else "ex-post",
                                 uses_realised_path=model not in EX_ANTE_MODELS, windows=int(m.sum()), purged_windows=purged,
                                 period=f"{(cal, val, tst)[('calibration', 'validation', 'test').index(role)][0]} to "
                                        f"{(cal, val, tst)[('calibration', 'validation', 'test').index(role)][1]}",
                                 slope_fitted=np.nan if beta is None else float(beta[1]),
                                 rmse_pct=100 * math.sqrt(sse / m.sum()), oos_r2=1.0 - sse / sse_c,
                                 mean_pnl_pct=float(y[m].mean() * 100)))
                if role == "validation":
                    scores[model] = math.sqrt(sse / m.sum())
        best = min(scores, key=scores.get)
        best_ex = min((m for m in scores if m in EX_ANTE_MODELS), key=scores.get)
        chosen[sname] = dict(selected_on_validation=best, validation_rmse_pct=100 * scores[best],
                             selected_ex_ante=best_ex, purged_windows=purged)
    df = pd.DataFrame(rows)
    for sname in SPLITS:
        for key, tag in (("selected_on_validation", ""), ("selected_ex_ante", "_ex_ante")):
            t = df[(df.split == sname) & (df.role == "test") & (df.model == chosen[sname][key])].iloc[0]
            chosen[sname].update({f"test_oos_r2{tag}": float(t.oos_r2), f"test_rmse_pct{tag}": float(t.rmse_pct)})
        chosen[sname]["test_windows"] = int(t.windows)
        w = df[(df.split == sname) & (df.role == "test") & (df.model == "gap_close_to_close")].iloc[0]
        v = df[(df.split == sname) & (df.role == "validation") & (df.model == "gap_close_to_close")].iloc[0]
        chosen[sname].update(gap_model_validation_oos_r2=float(v.oos_r2), gap_model_test_oos_r2=float(w.oos_r2),
                             gap_model_fitted_slope=float(w.slope_fitted))
    return df, chosen
