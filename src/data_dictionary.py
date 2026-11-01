"""Generates docs/DATA_DICTIONARY.md: every input file and every column of every results file, with units.

Descriptions live here, written by hand, because a column name cannot say what it means.  ``missing()`` lists any
results column that has no description, and a test fails if there is one, so the dictionary cannot silently go stale.

    python -m src.data_dictionary      # rewrite docs/DATA_DICTIONARY.md
"""
from __future__ import annotations

import glob
import json
from pathlib import Path

import pandas as pd

from . import config as cfg

DOC = cfg.ROOT / "docs" / "DATA_DICTIONARY.md"

RAW = {
    "NIFTY 50_Historical_PR_<start>to<end>.csv": ("NIFTY 50 price-return index, one file per calendar year (11 files). Source: "
        "niftyindices.com historical data. Quoted CSV, Windows line endings, newest row first.",
        {"Index Name": "constant string 'NIFTY 50'", "Date": "trading date, format '30 Dec 2016'",
         "Open / High / Low / Close": "index level in index points (price return, dividends excluded)"}),
    "hist_india_vix_<start>-to-<end>.csv": ("India VIX, one file per year (11 files). Source: nseindia.com historical VIX report. "
        "UTF-8 with byte-order mark; column names carry trailing spaces; oldest row first.",
        {"Date": "trading date, format '01-JAN-2016'", "Open / High / Low / Close": "VIX level in PERCENT points (14.06 = 14.06%)",
         "Prev. Close": "previous day's close, percent", "Change / % Change": "day change, not used"}),
    "MANIFEST.json": ("SHA-256, size, row count and date range of each raw file (see src/integrity.py).",
        {"sha256": "hash of the file bytes", "rows": "number of data rows", "first_date / last_date": "ISO dates"}),
}

FILES = {
    "partA_A1_sanity.csv": "A1: GBM sanity check, sigma_r = sigma_i = 20%, mu = r; mean PnL should be 0 (4 values of N).",
    "partA_A2_frequency.csv": "A2: spread of hedging PnL against number of rebalances N (20,000 paths each).",
    "partA_A3_volgap.csv": "A3: mean PnL against the volatility gap, simulation vs closed-form theory, N = 250.",
    "partA_A4_costs.csv": "A4: PnL net of proportional transaction costs, by N and cost rate, with risk-averse objectives.",
    "partA_A4_optima.csv": "A4: the cost-minimising N for each cost rate and risk-aversion weight.",
    "partA_A5_drift.csv": "A5: effect of the real-world drift mu on hedging PnL at finite N.",
    "partA_A6_estimator_convergence.csv": "A6: error of the discrete gamma-PnL approximation against the true PnL as N grows.",
    "partB_windows.csv": "One row per 21-trading-day window of real data: inputs, outcome, estimator. The base table of Part B. Not included in the public repository (it holds the VIX level of every date); `python -m src reproduce` regenerates it from the raw data.",
    "partB_frequency.csv": "Part B: hedging every k trading days instead of daily.",
    "partB_regimes.csv": "Part B: results by VIX regime at window start (calm / middle / turbulent) and for crash windows.",
    "partB_yearly.csv": "Part B: results by calendar year of the window start.",
    "partB_sensitivity.csv": "Part B: headline statistics under alternative assumptions (rate, centred vol, exclusions).",
    "partB_rate_sensitivity.csv": "Part B: effect of the risk-free rate (4%, 6%, 8%) relative to the 6% placeholder.",
    "partB_nonoverlapping_subsamples.csv": "Part B: regression on each of the 21 non-overlapping subsamples.",
    "partB_gbm_benchmarks.csv": "Part B: the same regression on idealised GBM data (constant-vol and conditional 'twin').",
    "partB_gbm_twin_volatility.csv": "Part B: distributions of the twin's input and simulated realised volatilities vs historical.",
    "partB_gamma_estimator_conventions.csv": "Part B: fit of the discrete gamma-PnL approximation under different return conventions.",
    "partB_hac_lag_sensitivity.csv": "Part B: standard error of the headline slope under plain OLS, White and Newey-West lags 0..63.",
    "partB_bootstrap_block_sensitivity.csv": "Part B: block-bootstrap interval of the headline R-squared for block lengths 21/42/63.",
    "partB_stretch_vix_rehedge.csv": "Part B stretch: re-hedging with each day's VIX versus the fixed initial VIX.",
    "partB_effect_size_intervals.csv": "Part B: headline effect sizes with 95% HAC and block-bootstrap intervals.",
    "partB_market_states.csv": "Part B: results by market state (calm / drawdown / recovery), defined from trailing index data.",
    "partB_vol_estimator_comparison.csv": "Part B: premium and regression fit under seven alternative realised-volatility estimators.",
    "partB_chronological_split.csv": "Part B: models fitted on a calibration period, scored on validation and test periods.",
    "partB_effect_size_mc_error.csv": "Part B: Monte Carlo error of the bootstrap interval ends, from repeating the effect-size bootstrap with independent random numbers.",
    "partB_calendar_time_sensitivity.csv": "Part B: headline statistics on the trading-day clock (baseline) against an ACT/365 calendar clock using the actual dates.",
    "partB_friction_scenarios.csv": "Part B: scenario grid of trading frictions (costs, option spread, lot size, interval). Assumptions, not measurements.",
}

# generic column meanings; a file-specific meaning goes in OVERRIDE[(file, column)]
COLUMNS = {
    "N": "number of equally spaced rebalancing times", "paths": "number of simulated paths", "mean_pnl": "mean hedging PnL, money units, undiscounted at maturity",
    "se_mean": "standard error of the mean", "z_mean": "mean / standard error", "std_pnl": "standard deviation of hedging PnL, money units",
    "se_std": "standard error of the standard deviation", "premium": "option premium received at t0, money units (Part A: 10.4506)",
    "std_over_premium": "std of PnL as a fraction of the premium", "std_times_sqrtN": "std(PnL) x sqrt(N); constant if error ~ 1/sqrt(N)",
    "sigma_r": "true (realised) annual volatility of the simulated path or the benchmark, decimal", "gap": "volatility gap sigma_i^2 - sigma_r^2 (annual variance units)",
    "theory_mean": "closed-form continuous-time expected PnL", "diff_sim_minus_theory": "simulated mean minus closed-form mean",
    "diff_in_se": "that difference in units of the simulation's standard error", "sign_ok": "True if the sign of the mean PnL matches the sign of the gap",
    "frac_paths_profit": "share of paths with PnL > 0", "r2_pnl_on_gaphat_within": "R-squared of PnL on the path's own variance gap within this single sigma_r setting",
    "cost_rate": "proportional transaction cost per unit notional traded (0.0005 = 5 bp)", "mean_gross": "mean PnL before costs", "mean_cost": "mean transaction costs",
    "se_cost": "standard error of mean cost", "mean_net": "mean PnL after costs", "std_net": "std of PnL after costs",
    "objective_lambda_0.5": "mean(net) - 0.5 x std(net)", "objective_lambda_1.0": "mean(net) - 1.0 x std(net)", "objective_lambda_2.0": "mean(net) - 2.0 x std(net)",
    "risk_aversion_lambda": "weight on std in the objective", "best_N": "N that maximises the objective", "objective_at_best": "objective value at best_N",
    "mu": "real-world annual drift of the simulated stock, decimal",
    "rmse_log": "RMSE of the approximation (squared log returns), money units", "rmse_log_pct_premium": "same, % of premium",
    "r2_pooled_log": "R-squared of true PnL on the approximation, all volatility settings pooled", "r2_within_equal_vol_log": "same, within the single setting sigma_r = sigma_i only (the gap carries no information there)",
    "rmse_simple": "as rmse_log, squared simple returns", "rmse_simple_pct_premium": "% of premium", "r2_pooled_simple": "pooled R-squared, simple returns",
    "r2_within_equal_vol_simple": "within sigma_r = sigma_i R-squared, simple returns", "rmse_const": "as rmse_log, constant-variance form",
    "rmse_const_pct_premium": "% of premium", "r2_pooled_const": "pooled R-squared, constant variance", "r2_within_equal_vol_const": "within sigma_r = sigma_i R-squared, constant variance",
    "block_length": "block length of the circular block bootstrap, in windows", "resamples_per_run": "bootstrap resamples per run", "runs": "independent bootstrap repeats",
    "r2_low": "2.5th percentile of bootstrapped R-squared (mean over runs)", "r2_median": "median bootstrapped R-squared", "r2_high": "97.5th percentile (mean over runs)",
    "r2_low_min": "smallest r2_low across runs", "r2_low_max": "largest r2_low across runs", "r2_high_min": "smallest r2_high across runs", "r2_high_max": "largest r2_high across runs",
    "split": "name of the chronological split ('primary' or 'crash_held_out')", "model": "predictor of window PnL; parameters fitted on the calibration period only",
    "role": "calibration / validation / test period", "uses_realised_path": "True if the predictor needs the window's own realised price path (not known when the option is sold)",
    "windows": "number of 21-day windows", "purged_windows": "windows dropped because they straddle a period boundary", "period": "date range of the role",
    "slope_fitted": "slope fitted on the calibration period", "rmse_pct": "root-mean-square prediction error, % of S0", "oos_r2": "1 - SSE(model)/SSE(calibration-mean benchmark); can be negative",
    "mean_pnl_pct": "mean hedging PnL, % of S0 (initial index level) per window", "effect": "name of the effect size", "unit": "unit of the quantity",
    "estimate": "point estimate", "hac_se": "Newey-West (lag 20) standard error", "hac_ci_lo": "lower end of estimate +/- 1.96 HAC SE", "hac_ci_hi": "upper end",
    "boot_ci_lo": "2.5th percentile, circular block bootstrap (block 42, 4,000 resamples)", "boot_ci_hi": "97.5th percentile", "note": "free-text caveat",
    "interval_days": "rebalance every k trading days", "rebalances": "number of hedge times in the window", "std_pnl_pct": "standard deviation of PnL, % of S0",
    "p05_pct": "5th percentile of PnL, % of S0", "p95_pct": "95th percentile of PnL, % of S0", "std_diff_vs_daily_pct": "std of (PnL at k days - daily-hedged PnL), % of S0",
    "r2_on_gap": "R-squared of PnL on variance gap", "slope_on_gap": "slope of PnL on variance gap", "std_diff_predicted_by_sqrtN_law_pct": "prediction of the previous column from c x sqrt(1/N_k - 1/N_1)",
    "c_fitted": "the single fitted constant c", "rebalance_every": "rebalance every k trading days",
    "one_way_cost_bps": "one-way proportional hedging cost, basis points of traded notional (scenario input)", "option_half_spread_vol_pts": "seller receives VIX minus this many vol points (scenario input)",
    "delta_step": "hedge ratios rounded to multiples of this (blank = divisible)", "p05_pnl_pct": "5th percentile of PnL, % of S0", "mean_turnover": "mean sum of |change in hedge ratio|, in option notionals",
    "mean_cost_pct": "mean transaction cost, % of S0", "dataset": "which set of windows the row is computed on", "convention": "definition of realised variance in the estimator",
    "r2": "R-squared", "slope_pnl_on_estimate": "slope of actual PnL on the estimate (1 = perfect scale)", "mean_error_pct": "mean(estimate - actual PnL), % of S0",
    "rmse_over_std_pnl": "RMSE of the estimate divided by the std of actual PnL", "benchmark": "which GBM benchmark", "observations": "number of simulated windows",
    "slope": "slope of PnL (fraction of S0) on variance gap", "sigma_i": "implied volatility used, decimal", "quantity": "which volatility series",
    "n": "number of observations", "mean": "mean", "std": "standard deviation", "p05": "5th percentile", "p25": "25th percentile", "p50": "median", "p75": "75th percentile", "p95": "95th percentile",
    "method": "standard-error method", "lag": "Newey-West lag (0 = none)", "se_slope": "standard error of the slope", "t_stat": "slope / standard error",
    "slope_ci_low": "slope - 1.96 SE", "slope_ci_high": "slope + 1.96 SE", "regime": "market state at window start", "share": "share of labelled windows",
    "mean_pnl_ci_lo": "mean PnL - 1.96 HAC SE, % of S0", "mean_pnl_ci_hi": "mean PnL + 1.96 HAC SE, % of S0", "worst_pnl_pct": "smallest PnL in the group, % of S0",
    "vol_gap_points": "mean(VIX - realised vol), vol points", "slope_se_hac": "Newey-West standard error of the slope", "offset": "start offset (0..20) of the non-overlapping subsample",
    "risk_free": "risk-free rate, decimal", "diff_mean_pnl_pct": "mean PnL minus the 6% value, percentage points", "diff_std_pnl_pct": "std minus the 6% value, pp",
    "diff_frac_profit": "share profitable minus the 6% value", "diff_slope": "slope minus the 6% value", "diff_r2": "R-squared minus the 6% value", "group": "label of the subset",
    "median_pnl_pct": "median PnL, % of S0", "se_slope_hac": "Newey-West standard error of the slope", "hedger": "which hedger", "estimator": "realised-volatility estimator",
    "mean_vol_gap_points": "mean(VIX - estimator), vol points", "se_vol_gap_hac": "Newey-West SE of that mean", "frac_rv_below_vix": "share of windows with estimator below VIX",
    "corr_with_close_to_close": "correlation with the baseline close-to-close estimate", "slope_t": "slope / HAC SE",
    "t0": "window start date (the day the option is sold)", "t_end": "window end date (21 trading days later)", "vix_pct": "India VIX at t0, percent points",
    "sigma_implied": "VIX / 100, decimal", "realised_vol": "annualised realised vol of the window (uncentred daily log returns), decimal", "variance_gap": "sigma_implied^2 - realised_vol^2",
    "premium_pct": "Black-Scholes premium, % of S0", "pnl_pct": "hedging PnL, % of S0, undiscounted at the window end", "gamma_estimate_pct": "discrete path-wise gamma-PnL approximation, % of S0",
    "covid_crash_window": "True if the window overlaps 20 Feb - 30 Apr 2020", "frac_profit": "share of windows with PnL > 0", "mean_vix": "mean VIX at window start, percent points",
    "mean_rv_pct": "mean realised vol, percent", "std_pnl": "standard deviation of PnL, money units",
    "clock": "which time convention the row uses (trading-day baseline or ACT/365 calendar)",
    "mean_pnl_hac_lo": "mean PnL - 1.96 HAC SE, % of S0", "mean_pnl_hac_hi": "mean PnL + 1.96 HAC SE, % of S0",
    "mean_premium_pct": "mean Black-Scholes premium, % of S0",
    "vrp_vol_points": "mean(VIX - realised vol), vol points", "vrp_hac_lo": "that mean - 1.96 HAC SE", "vrp_hac_hi": "that mean + 1.96 HAC SE",
    "slope_hac_lo": "slope - 1.96 HAC SE", "slope_hac_hi": "slope + 1.96 HAC SE",
    "pathwise_estimate_r2": "R-squared of actual PnL on the path-wise discrete gamma-PnL approximation",
    "track": "'ex-ante' = uses only information known when the option is sold; 'ex-post' = uses the window's own realised path",
    "reps": "number of independent repeats of the bootstrap", "resamples": "bootstrap resamples in each repeat",
    "lo_mean": "mean over repeats of the lower bootstrap bound", "lo_sd": "standard deviation of the lower bound over repeats", "lo_range": "max - min of the lower bound over repeats",
    "hi_mean": "mean over repeats of the upper bootstrap bound", "hi_sd": "standard deviation of the upper bound over repeats", "hi_range": "max - min of the upper bound over repeats",
    "max_bound_sd_over_width": "larger of the two bound sds divided by the mean interval width (Monte Carlo error relative to the interval)",
    "type": "what the volatility estimator measures (close-to-close, range without overnight gap, range with overnight gap)",
}
OVERRIDE = {
    ("partA_A2_frequency.csv", "mean_pnl"): "mean hedging PnL, money units; near 0 because sigma_r = sigma_i and mu = r",
    ("partB_frequency.csv", "mean_pnl_pct"): "mean hedging PnL, % of S0",
    ("partB_effect_size_intervals.csv", "unit"): "unit of the estimate",
    ("partB_gbm_twin_volatility.csv", "unit"): "unit of the row (vol points or variance)",
    ("partB_gbm_benchmarks.csv", "sigma_r"): "constant-GBM row only: true volatility used, decimal",
    ("partB_gbm_benchmarks.csv", "sigma_i"): "constant-GBM row only: implied volatility used, decimal",
    ("partB_rate_sensitivity.csv", "slope"): "slope of PnL on variance gap at this rate",
    ("partB_rate_sensitivity.csv", "r2"): "R-squared of PnL on variance gap at this rate",
    ("partB_nonoverlapping_subsamples.csv", "se_slope"): "plain OLS standard error of the slope (valid: windows do not overlap)",
    ("partB_nonoverlapping_subsamples.csv", "n"): "windows in this subsample",
    ("partB_gbm_twin_volatility.csv", "n"): "number of observations",
}


def describe(file: str, col: str) -> str | None:
    return OVERRIDE.get((file, col)) or COLUMNS.get(col)


WINDOWS_FILE = "partB_windows.csv"
WINDOWS_COLUMNS = ("t0", "t_end", "vix_pct", "sigma_implied", "realised_vol", "variance_gap", "premium_pct", "pnl_pct",
                   "gamma_estimate_pct", "covid_crash_window")      # must equal the columns written by backtest.run_part_b


def results_files() -> list[Path]:
    return [Path(p) for p in sorted(glob.glob(str(cfg.RESULTS_DIR / "*.csv")))]


def _described_files() -> list[tuple[str, tuple[str, ...], int]]:
    """(name, columns, rows) for every results CSV; the windows table is described from the summary when its file is absent
    (the public repository does not ship it), so the document is identical either way."""
    out = {}
    for p in results_files():
        df = pd.read_csv(p)
        out[p.name] = (tuple(df.columns), len(df))
    if WINDOWS_FILE not in out:
        import json
        out[WINDOWS_FILE] = (WINDOWS_COLUMNS, int(json.loads((cfg.RESULTS_DIR / "partB_summary.json").read_text())["windows"]))
    return [(n, c, r) for n, (c, r) in sorted(out.items())]


def missing() -> list[str]:
    out = []
    for p in results_files():
        if p.name not in FILES:
            out.append(f"{p.name}: file not described")
        for c in pd.read_csv(p, nrows=0).columns:
            if describe(p.name, c) is None:
                out.append(f"{p.name}: column {c!r} not described")
    return out


def build() -> str:
    lines = ["# Data dictionary", "",
             "Generated by `python -m src.data_dictionary`. Every results column below is covered by a test, so this file cannot silently fall "
             "behind the code. Conventions used everywhere: PnL is measured at the option's maturity, undiscounted; Part B PnL is expressed as "
             "% of S0 (the index level on the day the option is sold); volatilities are annualised; one trading day = 1/252 year; "
             "'vol points' = percentage points of annualised volatility.", "",
             "## Raw inputs (`data/raw/`)", "",
             "Provenance, download dates and known problems of each file: `data/raw/SOURCES.md`. Integrity: `python -m src.integrity`.", ""]
    for name, (desc, cols) in RAW.items():
        lines += [f"### `{name}`", "", desc, "", "| column | meaning |", "|---|---|"] + [f"| {k} | {v} |" for k, v in cols.items()] + [""]
    lines += ["## Derived quantities used in Part B", "",
              "| quantity | definition |", "|---|---|",
              "| window | 21 consecutive trading days of NIFTY closes starting at t0 (stride 1 day, so windows overlap); kept if t0 has a valid VIX and the window spans at most 45 calendar days |",
              "| sigma_implied | VIX close at t0 divided by 100 |", "| S0, K | NIFTY close at t0; the call is at the money, K = S0 |",
              "| realised volatility | sqrt(252/21 x sum of squared daily log returns), no mean subtracted |",
              "| variance gap | sigma_implied^2 minus realised volatility^2 |",
              "| hedging PnL | self-financing daily delta hedge of one short call, Black-Scholes delta at the fixed VIX, risk-free rate 6% (placeholder) |",
              "| market state | calm: drawdown from the running peak > -5%; drawdown: <= -5% and 63-day return < 0; recovery: <= -5% and 63-day return >= 0 |", ""]
    lines += ["## Results files (`results/`)", ""]
    for name, columns, n_rows in _described_files():
        lines += [f"### `{name}`", "", f"{FILES.get(name, '(not described)')} {n_rows:,} rows.", "",
                  "| column | meaning |", "|---|---|"] + [f"| `{c}` | {describe(name, c) or '(not described)'} |" for c in columns] + [""]
    other = [("partA_summary.json", "Part A summary numbers: seed, paths per setting, log-log fits, A3 aggregates, A6 slope."),
             ("partB_summary.json", "Part B summary numbers: headline regression, volatility premium, timing effect, twin comparison, market states, "
                                    "estimator, split and friction summaries, data-cleaning report."),
             ("validation_report.md", "Output of the executable validation checklist."), ("data_inspection.txt", "Raw-data inspection report.")]
    lines += ["### Other results files", "", "| file | content |", "|---|---|"] + [f"| `{a}` | {b} |" for a, b in other] + [""]
    return "\n".join(lines)


def main() -> None:
    DOC.parent.mkdir(exist_ok=True)
    DOC.write_text(build())
    print(f"wrote {DOC}")


if __name__ == "__main__":
    main()
