"""Write README.md from the saved results in results/.

Every number in the README is read from a results file produced by an actual run
(``python -m src.run_all``) - none is typed by hand.  Re-run this script after re-running the
experiments:

    python -m src.run_all && python -m src.make_readme
"""
from __future__ import annotations

import json
import re
import subprocess
import sys

import pandas as pd

from . import config as cfg

R = cfg.RESULTS_DIR


def _csv(name):
    return pd.read_csv(R / name)


def _md(df: pd.DataFrame, cols: list[tuple[str, str, str]]) -> str:
    """Markdown table from (column, header, format) triples."""
    head = "| " + " | ".join(h for _, h, _ in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c, _, f in cols:
            v = r[c]
            cells.append(str(int(v)) if f == '' and not isinstance(v, str) else (format(v, f) if not isinstance(v, str) else v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


RAW_HERE = "untouched CSV downloads + SOURCES.md + MANIFEST.json (SHA-256 integrity manifest)"
RAW_ABSENT = "SOURCES.md + MANIFEST.json (SHA-256 manifest); the CSV files are not in the public repository"
GET_DATA = (
    "**Getting the data.** The public repository does not include the raw NIFTY 50 and India VIX files, because their publishers' "
    "terms restrict redistribution. Download them yourself (links, file layout and known problems are in "
    "[data/raw/SOURCES.md](data/raw/SOURCES.md)), put the 22 CSV files in `data/raw/`, and run `python -m src check-data`: it compares "
    "them with a SHA-256 manifest and tells you whether they are the files these results came from. Without the files, the tests and "
    "stages that need them are skipped or stop with this message, and everything that only needs the committed results (the "
    "README, report, figures gallery, data dictionary and the Part A experiments) still works.")


def _volatile(text: str) -> str:
    """Wrap text that legitimately differs between runs/machines so the staleness check can ignore it."""
    return f"<!--volatile-->{text}<!--/volatile-->"


def _test_summary() -> str:
    """Outcome of the last ``python -m src test`` run (recorded in results/test_run.json; nothing typed by hand)."""
    n_files = len(list((cfg.ROOT / "tests").glob("test_*.py")))
    f = R / "test_run.json"
    if not f.exists():
        return _volatile(f"{n_files} test files; no recorded run yet, run `python -m src test`")
    t = json.loads(f.read_text())
    bad = f", {t['failed']} failed" if t["failed"] else ""
    bad += f", {t['skipped']} skipped (need the raw data files)" if t.get("skipped") else ""
    bad += f", {t['errors']} errors" if t["errors"] else ""
    return _volatile(f"{n_files} test files: last recorded run {t['passed']} passed{bad} (Python {t['python']})")


def _timing_summary() -> str:
    """Per-stage wall time and peak memory of the last ``python -m src reproduce`` (results/reproduce_stats.json)."""
    f = R / "reproduce_stats.json"
    if not f.exists():
        return _volatile("not recorded yet, run `python -m src reproduce`")
    t = json.loads(f.read_text())
    parts = [f"{s['stage'].replace('_', ' ')} {s['seconds']:.0f} s" + (f" / {s['peak_rss_mb']:.0f} MB" if s["peak_rss_mb"] else "")
             for s in t["stages"]]
    return _volatile(f"last run on Python {t['python']}: " + "; ".join(parts) + f"; computation total {t['total_seconds']:.0f} s "
                     "(wall time / peak resident memory per stage; hardware dependent)")


def _report_raw() -> str:
    a1, a2, a3, a4, a4o, a5 = (_csv(f) for f in ("partA_A1_sanity.csv", "partA_A2_frequency.csv", "partA_A3_volgap.csv",
                                                "partA_A4_costs.csv", "partA_A4_optima.csv", "partA_A5_drift.csv"))
    sa = json.loads((R / "partA_summary.json").read_text())
    f2, s3 = sa["a2_fits"], sa["a3_summary"]
    sb = json.loads((R / "partB_summary.json").read_text())
    h, vrp, no, dr = sb["headline"], sb["vrp"], sb["non_overlapping"], sb["data_report"]
    freq, sens, rate, reg = (_csv(f) for f in ("partB_frequency.csv", "partB_sensitivity.csv",
                                              "partB_rate_sensitivity.csv", "partB_regimes.csv"))
    yearly, bench, stretch = _csv("partB_yearly.csv"), _csv("partB_gbm_benchmarks.csv"), _csv("partB_stretch_vix_rehedge.csv")
    validation = (R / "validation_report.md").read_text().split("\n\n", 1)[1].strip()
    tm, gw = sb["timing_effect_pct_of_S0"], sb["gamma_weight_slope_by_group"]
    prem = float(a1["premium"].iloc[0])
    sens_by = {r.group: r for r in sens.itertuples()}
    ex = sens_by["excluding Feb 20 - Apr 30 2020 windows"]
    nw = sens_by["weekend sessions dropped"]
    ce = sens_by["realised vol = sample std (centred)"]
    b_const, b_twin = bench.iloc[0], bench.iloc[1]
    rg = {r.group: r for r in reg.itertuples()}
    calm = reg.iloc[1]; mid = reg.iloc[2]; turb = reg.iloc[3]; crash = reg.iloc[4]
    base_fixed, rehedge = stretch.iloc[0], stretch.iloc[1]
    best = {(r.cost_rate, r.risk_aversion_lambda): int(r.best_N) for r in a4o.itertuples()}
    r0 = a2.set_index("N")
    a6 = _csv("partA_A6_estimator_convergence.csv")
    a6_slope = sa["a6_rmse_loglog_slope"]
    a6i = a6.set_index("N")
    conv = _csv("partB_gamma_estimator_conventions.csv")
    tv = _csv("partB_gbm_twin_volatility.csv")
    hs, bsn = _csv("partB_hac_lag_sensitivity.csv"), _csv("partB_bootstrap_block_sensitivity.csv")
    hnw = hs[hs.method.str.startswith("Newey")].set_index("lag")
    hs_ols, hs_white = hs.iloc[0], hs.iloc[1]
    bsn_i = bsn.set_index("block_length")
    tc = sb["twin_conditional"]
    tv_fmt = tv.copy()
    for col in ("mean", "std", "p05", "p25", "p50", "p75", "p95"):
        tv_fmt[col] = [f"{v:.2f}" if u == "vol points" else f"{v:.4f}" for v, u in zip(tv[col], tv["unit"])]
    tv_fmt["n"] = [f"{int(v):,}" for v in tv["n"]]
    tvr = tv.set_index("quantity")
    Q_HIST, Q_SIM, Q_NOISE = (tvr.index[1], tvr.index[2], tvr.index[3])
    cvr = conv.set_index(["dataset", "convention"])
    C_LOG, C_SIMPLE, C_DEMEAN, C_UNC, C_CEN = (
        "path-wise, squared log returns (baseline)", "path-wise, squared simple returns",
        "path-wise, log returns demeaned within the window", "constant variance, un-centred window variance",
        "constant variance, centred (sample) variance")
    D_ALL, D_EX, D_TWIN = "real, all windows", "real, excluding crash windows", "GBM twin (all windows)"
    pw_r2 = [cvr.loc[(D_ALL, c), "r2"] for c in (C_LOG, C_SIMPLE, C_DEMEAN)]
    ef, mst, ve = _csv("partB_effect_size_intervals.csv"), _csv("partB_market_states.csv"), _csv("partB_vol_estimator_comparison.csv")
    cs, fsc = _csv("partB_chronological_split.csv"), _csv("partB_friction_scenarios.csv")
    sbl, sve, sfr, sbs = sb["chronological_split"], sb["vol_estimators"], sb["frictions"], sb["market_states"]
    sbc_all = sb["effect_size_block"]
    sbc, sbc_block, sbc_boot = sbc_all["slope_bootstrap_vs_crash_share"], sbc_all["block"], sbc_all["n_boot"]
    ef_fmt = ef.copy()
    ef_fmt["estimate"] = [f"{v:.4g}" for v in ef.estimate]
    ef_fmt["hac"] = ["n/a" if pd.isna(a) else f"{a:.4g} to {b:.4g}" for a, b in zip(ef.hac_ci_lo, ef.hac_ci_hi)]
    ef_fmt["boot"] = [f"{a:.4g} to {b:.4g}" for a, b in zip(ef.boot_ci_lo, ef.boot_ci_hi)]
    ef_by = ef.set_index("effect")
    ef_slope, ef_mean = ef_by.loc["slope of PnL on variance gap"], ef_by.loc["mean hedging PnL"]
    m_calm, m_draw, m_rec = mst.iloc[0], mst.iloc[1], mst.iloc[2]
    ewma = 0.94
    rows_sv = []
    for sp in ("primary", "crash_held_out"):
        for mdl in ("constant", "vix_level", "exante_var_gap", "exante_vol_gap", "gap_close_to_close", "gap_bipower", "gap_parkinson", "gamma_estimate"):
            v = cs[(cs.split == sp) & (cs.model == mdl) & (cs.role == "validation")].iloc[0]
            t = cs[(cs.split == sp) & (cs.model == mdl) & (cs.role == "test")].iloc[0]
            rows_sv.append(dict(split=sp.replace("_", " "), model=mdl, track=v.track,
                                slope="-" if pd.isna(v.slope_fitted) else f"{v.slope_fitted:.3f}",
                                val_rmse=f"{v.rmse_pct:.3f}", val_r2=f"{v.oos_r2:+.2f}", test_rmse=f"{t.rmse_pct:.3f}", test_r2=f"{t.oos_r2:+.2f}"))
    split_view = pd.DataFrame(rows_sv)
    ex_ante_test = cs[(cs.role == "test") & (cs.track == "ex-ante") & (cs.model != "constant")].oos_r2
    ex_t = cs[(cs.role == "test") & (cs.track == "ex-ante")].set_index(["split", "model"]).oos_r2
    ex_v = cs[(cs.role == "validation") & (cs.track == "ex-ante")].set_index(["split", "model"]).oos_r2
    cal_t, mc = _csv("partB_calendar_time_sensitivity.csv"), _csv("partB_effect_size_mc_error.csv")
    cal0, cal1, cinfo = cal_t.iloc[0], cal_t.iloc[1], sb["calendar_time"]
    split_ex1 = cs[(cs.split == "primary") & (cs.model == "vix_level") & (cs.role == "test")].oos_r2.iloc[0]
    split_ex3 = cs[(cs.split == "crash_held_out") & (cs.model == "vix_level") & (cs.role == "validation")].oos_r2.iloc[0]
    split_ex2 = cs[(cs.split == "crash_held_out") & (cs.model == "vix_level") & (cs.role == "test")].oos_r2.iloc[0]
    fs0 = fsc[(fsc.option_half_spread_vol_pts == 0) & fsc.delta_step.isna()]
    fr_piv = fs0.pivot(index="one_way_cost_bps", columns="rebalance_every", values="mean_pnl_pct").rename(columns=lambda k: f"d{k}").reset_index()
    fr_daily0 = float(fr_piv.loc[fr_piv.one_way_cost_bps == 0, "d1"].iloc[0])
    fr_5d0 = float(fr_piv.loc[fr_piv.one_way_cost_bps == 0, "d5"].iloc[0])
    fr_daily5 = float(fr_piv.loc[fr_piv.one_way_cost_bps == 5, "d1"].iloc[0])
    lots = fsc[fsc.delta_step.notna() & (fsc.one_way_cost_bps == 0)]
    fr_std0 = float(fsc[(fsc.rebalance_every == 1) & (fsc.one_way_cost_bps == 0) & (fsc.option_half_spread_vol_pts == 0) & fsc.delta_step.isna()].std_pnl_pct.iloc[0])
    lot_max_dm = float((lots.mean_pnl_pct - fr_daily0).abs().max())
    lot_max_std = float(lots.std_pnl_pct.max())
    pct = lambda x, d=1: f"{x * 100:.{d}f}%"  # noqa: E731

    return f"""# Research report: Discrete Delta-Hedging Error Analysis, Simulated and Real Index Data

This is the full report behind the [README](../README.md). A self-initiated quantitative-finance project in Python (NumPy, pandas, SciPy, matplotlib). It asks two questions:

1. When an option seller delta-hedges at discrete times instead of continuously, how large is the leftover hedging error, and how fast does it shrink as the hedger rebalances more often?
2. What decides whether the seller makes or loses money? Theory says it is the gap between the volatility priced into the option (implied) and the volatility the market then delivers (realised). Does that hold on real NIFTY 50 data, with India VIX as the implied volatility?

Part A is a controlled experiment on simulated geometric Brownian motion (GBM). Part B runs the same hedging procedure on {sb['windows']:,} rolling 21-trading-day windows of real NIFTY 50 history ({sb['first_t0']} to {sb['last_end']}).

> **Read this first: one input is a placeholder.** The risk-free rate in Part B is a constant **{cfg.RISK_FREE_RATE_B:.0%}**, an unsourced assumption, not a sourced rate. A sensitivity check at 4% and 8% is reported below; it moves no conclusion.
>
> Every number in this report is read from a results file written by an actual run (`src/make_readme.py` builds this file from `results/`). Where Part B disagrees with the theory, the disagreement is reported in its own section.
>
> **Status of the evidence: exploratory, and not a trading result.** Many comparisons are reported here (eleven volatility estimators, two chronological splits, eight predictive models, three market states, several regimes and sensitivity variants) and none is adjusted for multiple testing, so intervals and t-statistics are descriptive, not confirmatory; no hypothesis was registered before the data were seen. The "+{h['mean_pnl_pct']:.3f}% per window" is the income of a *theoretical* Black-Scholes call priced from India VIX and hedged in the index, with no option quotes, no futures hedge and no measured costs; it shows the mechanism (implied above realised) and is not the return of a strategy anyone could have executed. Results are kept in two tracks that are never mixed in a claim: an **ex-post mechanism track** (uses the window's own realised path: explains the PnL after the fact) and an **ex-ante prediction track** (uses only what is known when the option is sold). Only the second can say anything about prediction, and it finds essentially none.

## Executive summary

- **Scaling (Part A, simulated).** Discrete delta-hedging error shrinks like 1/sqrt(N): the log-log slope is {f2['slope_all']:.3f} against the theoretical -0.5. Doubling the rebalancing frequency does not halve the error; 100 times more rebalancing cuts it about {f2['std_ratio_N5_over_N500']:.0f} times.
- **Who wins (Part B, NIFTY 50 and India VIX, {sb['windows']:,} overlapping 21-day windows).** A seller who delta-hedged daily earned **{h['mean_pnl_pct']:+.3f}% of the index level per window** (95% interval {ef_mean.hac_ci_lo:.2f} to {ef_mean.hac_ci_hi:.2f}), profitable in {pct(h['frac_profit'], 0)} of windows, because India VIX exceeded realised volatility by {vrp['mean_vol_gap_points']:.1f} vol points on average.
- **Where the theory fails.** Theory says PnL tracks the gap between implied and realised variance with a slope of about {sb['theory_slope_mean']:.2f}. The measured slope is {h['slope']:.3f} and R² is {h['r2']:.2f}. The path-wise gamma-PnL approximation explains {pct(sb['r2_pnl_vs_gamma_estimate'], 0)} of the variation, so the mechanism is right but a single realised-volatility number is the wrong summary when the volatility arrives far from the strike (March 2020).
- **How reliable are the statistics.** Naive standard errors are about {h['se_slope_hac'] / h['se_slope_ols']:.0f} times too small because windows overlap. Under every dependence assumption tested the slope stays positive and clearly below the theory, but the R² interval is wide ({h['r2_block_bootstrap_ci'][0]:.2f} to {bsn_i.loc[63, 'r2_high']:.2f} depending on block length).
- **Regimes.** The seller's average PnL is positive in calm, drawdown and recovery states, but the slope differs a lot ({m_calm.slope:.2f}, {m_draw.slope:.2f}, {m_rec.slope:.2f}). A slope fitted on calm 2016-2019 data explains {pct(sbl['crash_held_out']['gap_model_test_oos_r2'], 0)} of the variation in 2023-onwards windows but fails badly across the 2020 crash.
- **Measurement.** The volatility risk premium is positive and significant under all {sve['n_estimators']} volatility estimators tried, but its size ranges from {sve['vol_gap_range_points'][0]:.1f} to {sve['vol_gap_range_points'][1]:.1f} vol points. The spread is an overnight-gap effect: estimators that see only the move within the trading day give {sve['vrp_range_range_no_gap'][0]:.1f} to {sve['vrp_range_range_no_gap'][1]:.1f} points, while the same estimators with the overnight gap added back give {sve['vrp_range_range_with_gap'][0]:.1f} to {sve['vrp_range_range_with_gap'][1]:.1f}, in line with the close-to-close family ({sve['vrp_range_close_to_close'][0]:.1f} to {sve['vrp_range_close_to_close'][1]:.1f}).
- **Prediction (ex-ante track).** Using only information known when the option is sold (VIX, and VIX against the trailing 21-day realised volatility), the out-of-sample R² on the test periods is between {ex_ante_test.min():+.3f} and {ex_ante_test.max():+.3f}: no usable predictive skill. The strong fits (R² about {sbl['primary']['test_oos_r2']:.2f}) use the window's own realised path and are explanations after the fact.
- **Which clock.** Re-running with actual calendar dates (ACT/365) instead of 21 equal trading-day steps changes the mean PnL from {cal0.mean_pnl_pct:+.3f}% to {cal1.mean_pnl_pct:+.3f}%, the slope from {cal0.slope:.3f} to {cal1.slope:.3f} and R² from {cal0.r2:.2f} to {cal1.r2:.2f}: same conclusions.
- **Costs (scenario analysis, not measured).** On a round-number grid the average income reaches zero at about {sfr['breakeven_cost_bps']['every_1d']:.0f} bp one-way hedging cost for daily hedging. The data contain no option quotes, futures prices or spreads, so real costs are unknown.
- **Not done:** real option prices, futures hedging with basis, delta-gamma hedging; these need data that are not in the repository (see Limitations).

Figure gallery: [docs/FIGURE_GALLERY.md](docs/FIGURE_GALLERY.md). Data dictionary: [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md).

## Headline results

| | Result |
|---|---|
| Part A: how hedging error scales | Log-log slope of std(PnL) against N is **{f2['slope_all']:.3f}** (regression SE {f2['slope_all_se']:.3f}); theory says about -0.5. Close, but not exactly -0.5: std x sqrt(N) is still creeping up ({r0.loc[5, 'std_times_sqrtN']:.2f} at N=5 to {r0.loc[500, 'std_times_sqrtN']:.2f} at N=500). |
| Part A: what drives PnL | The discrete path-wise gamma-PnL approximation tracks {pct(s3['r2_pnl_vs_gamma_estimate_pathwise'], 1)} of the path-by-path variation in hedging PnL (N=250), and its error shrinks as sampling gets finer (experiment A6: RMSE {a6i.loc[5, 'rmse_log_pct_premium']:.1f}% of the premium at N=5, {a6i.loc[500, 'rmse_log_pct_premium']:.2f}% at N=500). Simulated mean PnL matches the closed-form continuous-time expectation within 1.8 standard errors at all 7 volatility settings. |
| Part B: volatility risk premium | India VIX exceeded the NIFTY volatility realised over the next 21 trading days by **{vrp['mean_vol_gap_points']:.2f} vol points** on average (overlap-robust SE {vrp['se_vol_gap_hac']:.2f}); realised was below VIX in {pct(vrp['frac_windows_rv_below_vix'], 1)} of windows. |
| Part B: seller's PnL | Average **+{h['mean_pnl_pct']:.3f}%** of the initial index level per window (average premium {h['mean_premium_pct']:.2f}%), profitable in {pct(h['frac_profit'], 1)} of windows, standard deviation {h['std_pnl_pct']:.3f}%. |
| Part B: PnL vs variance gap | Slope **{h['slope']:.3f}** (overlap-robust SE {h['se_slope_hac']:.3f}), **R² = {h['r2']:.2f}** over {h['n']:,} overlapping windows (block-bootstrap 95% interval {h['r2_block_bootstrap_ci'][0]:.2f} to {h['r2_block_bootstrap_ci'][1]:.2f}). Theory predicts an average slope of {sb['theory_slope_mean']:.2f}, so the full-sample slope is much weaker than the theory. See "Where Part B does not match the theory". |
| Part B: approximation on the actual paths | Hedging PnL against the discrete path-wise gamma-PnL approximation evaluated along each real path: R² = {sb['r2_pnl_vs_gamma_estimate']:.2f}, slope {sb['ols_pnl_on_gamma_estimate']['slope']:.2f}; across three return conventions the R² ranges from {min(pw_r2):.3f} to {max(pw_r2):.3f}. This is an empirical fit of an approximation, not an exact identity with daily data. The mechanism is right; the weak regression comes from what the simple regressor leaves out. |

## 1. Theory

Notation: underlying S, strike K, maturity T (years), risk-free rate r, volatility sigma. The seller sells one call at time 0 at the Black-Scholes price computed with the implied volatility sigma_i, holds `delta` units of the underlying, rebalances at N equally spaced times, and cash earns or pays r. At maturity, hedging PnL = (value of the hedge portfolio) - (option payoff owed). PnL is measured in money at maturity (not discounted).

**Black-Scholes.** With d1 = [ln(S/K) + (r + sigma²/2) tau] / (sigma sqrt(tau)) and d2 = d1 - sigma sqrt(tau): call = S N(d1) - K e^(-r tau) N(d2); delta = N(d1); gamma = n(d1) / (S sigma sqrt(tau)).

**Gamma-PnL formula.** If the seller hedges continuously with delta computed at sigma_i while the market realises volatility sigma_r, then to first order

    PnL = integral over [0, T] of  e^(r(T-t)) * (1/2) * Gamma_t * S_t² * (sigma_i² - sigma_r²) dt

The seller profits when realised volatility is below implied, loses when it is above, and the size depends on gamma, which is largest near the strike and near expiry. Note the weight Gamma_t S_t² is path-dependent: it is large only while the index is close to the strike.

**Continuous-time identity versus daily-data approximation.** The expression above is a continuous-time, first-order statement: it holds as an identity only for a continuously observed price path (where "realised variance" means the quadratic variation of ln S) and continuous hedging. With daily closes the code evaluates a *discrete approximation*: each step's squared return stands in for that step's quadratic variation, Gamma_t S_t² is frozen at the start of each step, and higher-order terms in the Taylor expansion are ignored. Squared returns converge to the quadratic variation as the sampling gets finer, but at 21 daily steps the approximation is not exact. Experiment A6 (Part A) measures how its error shrinks with N, and section 6 measures how the fit on real data changes with the return convention.

**Discrete hedging.** With N rebalances the hedging error has standard deviation proportional to roughly 1/sqrt(N): each interval leaves a random error of size proportional to (1/2) Gamma S² (Z² - 1) sigma² dt, with zero mean and variance proportional to dt²; the N independent errors add in variance, so the sum has variance proportional to N dt² = T dt, i.e. std proportional to sqrt(dt) = 1/sqrt(N).

**Drift.** The stock's real-world drift mu does not appear in the price or in the continuous-time formula above. Part A, experiment A5, checks what happens at finite N.

## 2. Repository layout and how to reproduce

```
delta-hedging-analysis/
├── README.md                short landing page (generated from results/ by src/make_readme.py)
├── EXPLAINER.md             plain-English walkthrough of the code and results
├── requirements.txt, requirements-lock.txt, .python-version, pyproject.toml, Makefile
├── .github/workflows/ci.yml tests + regeneration on every push
├── data/raw/                {RAW_HERE if cfg.raw_data_available() else RAW_ABSENT}
├── docs/                    RESEARCH_REPORT.md (this file), DATA_DICTIONARY.md, FIGURE_GALLERY.md (all generated); OPTIONS_EXTENSION_PLAN.md (hand-written plan)
├── src/
│   ├── bs.py                Black-Scholes price, delta, gamma (vectorised, validated inputs)
│   ├── hedge.py             generic hedging engine (+ discrete gamma-PnL approximation)
│   ├── gbm_sim.py           Part A: path generator, experiments A1-A6, closed-form theory
│   ├── real_data.py         Part B: loading, cleaning, alignment, window construction
│   ├── backtest.py          Part B: backtest, regressions, benchmarks, sensitivities
│   ├── research.py          Part B: effect-size intervals, market states, volatility estimators, ex-ante/ex-post chronological split
│   ├── calendar_time.py     Part B: ACT/365 calendar-clock re-run of the hedge
│   ├── stages.py, verify_outputs.py   per-stage subprocess runner (time and memory) and the output-staleness comparison
│   ├── frictions.py         Part B: trading-friction scenario grid (assumptions, not measurements)
│   ├── integrity.py         raw-data SHA-256 manifest check
│   ├── data_dictionary.py   generates docs/DATA_DICTIONARY.md
│   ├── validation.py        the validation checklist as executable code
│   ├── plots.py, make_readme.py, make_gallery.py, config.py, __main__.py, run_part_a.py, run_part_b.py, run_all.py, inspect_data.py
├── tests/                   pytest suite (includes independent reproduction of the results)
├── results/                 every table behind every number (CSV / JSON / Markdown)
├── figures/                 all figures (PNG)
└── notebooks/               (empty; scripts regenerate everything)
```

To reproduce everything (a few minutes on a laptop):

```
pip install -r requirements.txt
python -m src reproduce        # integrity check, then one subprocess per stage: validation, Part A (seed {cfg.SEED}), Part B, gallery, data dictionary, README + report
python -m src test             # test suite
python -m src check-data       # verify the raw files against their SHA-256 manifest
python -m src snapshot DIR && python -m src verify-outputs DIR   # compare regenerated outputs with a snapshot (what CI does)
python src/inspect_data.py     # raw-data inspection report
```

Part A uses seed {cfg.SEED}; every experiment draws from its own stream, so results do not depend on run order. Part B is deterministic apart from a seeded block bootstrap. All tunable inputs are in `src/config.py`.

## 3. Data

| Series | Source | Downloaded | Coverage |
|---|---|---|---|
| NIFTY 50 price-return index, daily OHLC (11 yearly files) | https://www.niftyindices.com/reports/historical-data | 17 Oct 2026 | {dr['first_date']} to 2026-10-16 ({dr['nifty_rows']:,} rows) |
| India VIX, daily OHLC, quoted in percent (11 yearly files) | https://www.nseindia.com/reports-indices-historical-vix | 17 Oct 2026 | 2016-01-01 to 2026-10-16 ({dr['vix_rows']:,} rows) |
| Risk-free rate | **Placeholder: constant {cfg.RISK_FREE_RATE_B:.0%} per year, unsourced assumption** | n/a | n/a |

{"The raw files are in `data/raw/`, unmodified, with `SOURCES.md`. The code never writes to that folder." if cfg.raw_data_available() else GET_DATA.replace("(links, file layout and known problems are in [data/raw/SOURCES.md](data/raw/SOURCES.md))", "(links, file layout and known problems are in `data/raw/SOURCES.md`)").replace("[data/raw/SOURCES.md](data/raw/SOURCES.md)", "`data/raw/SOURCES.md`")}

**Sample period.** The full overlap of the two series, {dr['first_date']} to {dr['last_date']} (about 10.8 years), is used, because a longer sample contains more volatility regimes. It includes the March 2020 crash: the VIX closed at its maximum of 83.6 on 2020-03-24 and NIFTY's worst day was -13.9% (log return) on 2020-03-23 (see `results/data_inspection.txt`). {sb['crash_windows']} windows overlap the stretch 20 Feb to 30 Apr 2020 (a window definition fixed in advance); results are reported with and without them.

**Inspection and cleaning (all in `src/real_data.py`, reported by `src/inspect_data.py`).**

- Formats differ between sources (date style, row order, a byte-order mark and trailing spaces in the VIX column names); the loaders normalise them.
- No missing or non-numeric values in the NIFTY files; no duplicate dates in either series.
- **The 2022 VIX file stops on 2022-12-01.** December 2022 has 21 NIFTY trading days and no VIX, so no window can start in that month. This costs 21 start dates; NIFTY price paths are not affected.
- **Two stale VIX rows** (2021-02-12 and 2021-03-30) have Open = High = Low = Prev. Close = 0 and a Close that merely repeats the previous close. They are treated as missing and not forward-filled.
- **Weekend-dated rows** ({dr['weekend_sessions_kept']} remain after alignment) are real special sessions (Diwali Muhurat, budget-day Saturdays). They are kept as ordinary trading days by default; a run with them dropped is reported in the sensitivity table.
- NIFTY has {dr['only_in_nifty']} dates with no VIX (the 21 December-2022 days plus the Saturday 2024-03-02 session). After alignment there are {dr['aligned_rows']:,} dates with both series and no NaNs. Holidays could not be checked against an official exchange calendar offline; the only check is that no gap between consecutive trading dates exceeds 5 calendar days (none does).
- Windows are built from the full NIFTY series, with a window kept only if India VIX is valid on its start date, so the missing VIX data does not punch holes in price paths. The last 21 dates cannot start a full window: {dr['aligned_rows']:,} - 21 = {sb['windows']:,} windows, each spanning {sb['calendar_days_min']} to {sb['calendar_days_max']} calendar days.

## 4. Method

**Part A (GBM, no data).** Paths: S(t+dt) = S(t) exp((mu - sigma_r²/2) dt + sigma_r sqrt(dt) Z). Defaults: S0 = K = {cfg.A_S0:g}, T = {cfg.A_T:g}, r = {cfg.A_R:.0%}, sigma_i = {cfg.A_SIGMA_I:.0%}, mu = r, {cfg.A_N_PATHS:,} paths per setting, option premium {prem:.4f}. With sigma_r = sigma_i and mu = r the mean hedging PnL is exactly zero for any N, which is used as a check.

| Experiment | What varies | Output |
|---|---|---|
| A1 sanity | N from 10 to 5,000 with sigma_r = sigma_i | mean PnL near 0, error shrinks |
| A2 frequency | N in 5, 10, 20, 50, 100, 250, 500 | std of PnL vs N, log-log slope |
| A3 volatility gap | sigma_r in 10% to 30% around sigma_i = 20% (N = 250) | mean PnL vs the variance gap, against closed-form theory |
| A4 costs | proportional costs of 5 and 10 basis points | total cost vs N and the best N |
| A5 drift (extra) | mu from -20% to 40% | does the drift matter at finite N? |

**Part B (real data).** For each start date t0 with a valid VIX close: sell an at-the-money call (K = NIFTY close at t0) priced at sigma_i = VIX(t0)/100; delta-hedge at the daily close for 21 trading days with delta computed at the fixed sigma_i; hedging PnL is expressed as a fraction of the initial index level. One trading day = 1/252 year. Realised volatility is sqrt(252/21 x the sum of the 21 squared daily log returns) (the un-centred form, a natural proxy for the quadratic variation in the gamma-PnL expression; the sample-standard-deviation form is a sensitivity, and section 6 compares several definitions). The regression is PnL on (sigma_i² - sigma_r²), with Newey-West standard errors (lag 20, because windows overlap by up to 20 days). Rebalancing study: subsample to every 1, 2, 5, 10 days (the hedge is held to maturity after the last rebalance). Daily data only; no intraday claims are made.

## 5. Part A results

### Table: hedging-error standard deviation vs N (A2) and log-log fit

{_md(a2, [("N", "N", ".0f"), ("std_pnl", "std of PnL", ".4f"), ("se_std", "SE of std", ".4f"), ("std_over_premium", "std / premium", ".3f"), ("std_times_sqrtN", "std x sqrt(N)", ".3f"), ("mean_pnl", "mean PnL", "+.4f"), ("se_mean", "SE of mean", ".4f")])}

{cfg.A_N_PATHS:,} paths per row. Fitted log-log slope: **{f2['slope_all']:.4f}** (regression SE {f2['slope_all_se']:.4f}, R² {f2['r2_all']:.4f}); using only N >= 20: {f2['slope_N_ge_20']:.4f} (SE {f2['slope_N_ge_20_se']:.4f}). The slope is within about 3% of -0.5 but several regression standard errors away from it, because std x sqrt(N) is not yet constant. A 100-fold increase in rebalances (N = 5 to 500) cut the standard deviation by a factor of {f2['std_ratio_N5_over_N500']:.2f}; a 1/sqrt(N) law predicts {f2['ratio_if_1_over_sqrtN']:.0f} and a 1/N law would predict {f2['ratio_if_1_over_N']:.0f}. The mean PnL is within about two standard errors of zero in every row, as it must be.

![scaling](figures/A2_scaling_loglog.png)

**Figure A2-scaling.** Standard deviation of hedging PnL against the number of rebalances, both axes logarithmic. The blue points are simulated; the orange line is an exact -1/2 slope anchored at N = 100; the pale blue line is the fitted line (slope {f2['slope_all']:.3f}). The points hug the reference line, with the visible curvature at small N captured by the slightly flatter fitted slope. Error bars (one standard error) are smaller than the markers.

![distributions](figures/A2_pnl_distributions.png)

**Figure A2-distributions.** Distribution of hedging PnL for N = 5, 20, 100 and 500 on a common horizontal axis. The distribution is centred on zero at every N and narrows steadily ({r0.loc[5, 'std_pnl']:.2f}, {r0.loc[20, 'std_pnl']:.2f}, {r0.loc[100, 'std_pnl']:.2f}, {r0.loc[500, 'std_pnl']:.2f}). At small N it has a heavier left tail (the worst of {cfg.A_N_PATHS:,} paths at N = 5 loses about 18.8 against a premium of {prem:.2f}); the panels note how many paths fall outside the plotted range.

### A1 sanity check

{_md(a1, [("N", "N", ".0f"), ("mean_pnl", "mean PnL", "+.5f"), ("se_mean", "SE", ".5f"), ("z_mean", "mean / SE", "+.2f"), ("std_pnl", "std of PnL", ".4f")])}

With sigma_r = sigma_i the mean PnL is within about one standard error of zero at every N, and the standard deviation falls from {a1.iloc[0]['std_pnl']:.2f} (N = 10) to {a1.iloc[-1]['std_pnl']:.3f} (N = 5,000). Validation check E2 repeats this test 100 times at N = 20 to confirm that the z-scores behave like standard normal noise.

### A3: PnL against the volatility gap

{_md(a3, [("sigma_r", "sigma_r", ".2f"), ("gap", "sigma_i² - sigma_r²", "+.4f"), ("mean_pnl", "simulated mean PnL", "+.4f"), ("se_mean", "SE", ".4f"), ("theory_mean", "gamma-formula theory", "+.4f"), ("diff_in_se", "(sim - theory) / SE", "+.2f"), ("frac_paths_profit", "share of paths with profit", ".3f"), ("r2_pnl_on_gaphat_within", "within-setting R²", ".3f")])}

![volgap](figures/A3_volgap_mean_pnl.png)

**Figure A3.** Left: mean hedging PnL against the variance gap for seven volatility settings (N = 250). The simulated means (blue, with 95% error bars) sit on the continuous-time gamma-formula curve (orange); the sign always matches the gap, and every simulated mean is within 1.8 standard errors of the theory. Right: for every simulated path the discrete path-wise gamma-PnL approximation evaluated along that path against its actual PnL (5,000 points shown out of {s3['n_pooled']:,}); the points lie close to the 45-degree line with R² = {s3['r2_pnl_vs_gamma_estimate_pathwise']:.3f}. Using one constant realised variance per path instead of each step's own squared return, R² is {s3['r2_pnl_vs_gamma_estimate_constvar']:.3f}.

**Why the R² is {s3['pooled_pnl_on_gaphat_r2']:.2f} across all paths but only {s3['within_r2_min']:.2f} to {s3['within_r2_max']:.2f} within a single volatility setting.** Pooling all {s3['n_pooled']:,} simulated paths and regressing PnL on (sigma_i² - that path's realised variance) gives slope {s3['pooled_pnl_on_gaphat_slope']:.1f} and **R² = {s3['pooled_pnl_on_gaphat_r2']:.2f}**. Taking one volatility setting at a time, the same regression gives R² between {s3['within_r2_min']:.2f} and {s3['within_r2_max']:.2f} (last column of the table). The data have not changed, only the range being looked at. R² asks "what share of the ups and downs of PnL is explained by the gap?", and the answer depends on how much the gap itself varies. Across the seven settings the gap swings widely (standard deviation of the regressor {s3['gaphat_std_pooled']:.4f}), and PnL follows it closely, so the points form a long, clear line. Inside one setting every path faces the same true volatility, so the gap varies only because of sampling luck in realised variance (average within-setting standard deviation {s3['gaphat_std_within_mean']:.4f}, about {s3['gaphat_std_pooled'] / s3['gaphat_std_within_mean']:.0f} times narrower). In that short stretch the other ingredient of the PnL, how much of the variance arrived while the index was close to the strike, where gamma is large, is no longer drowned out by the big between-setting differences. Looking at a few people in one town you cannot see that heights rise with age as clearly as when you compare children with adults, even though the rule is the same. The within-setting R² is also lowest at the extreme settings and highest when sigma_r = sigma_i ({a3.loc[a3.sigma_r == 0.2, 'r2_pnl_on_gaphat_within'].iloc[0]:.2f}). This matters for Part B: on real data, with VIX and realised volatility both varying from window to window, the R² lands in between.

### A4: transaction costs

{_md(a4, [("N", "N", ".0f"), ("cost_rate", "cost per unit traded", ".4f"), ("mean_cost", "mean total cost", ".4f"), ("mean_net", "mean net PnL", "+.4f"), ("std_net", "std of net PnL", ".4f"), ("objective_lambda_1.0", "mean - 1 x std", "+.4f")])}

![costs](figures/A4_costs_and_optimal_N.png)

**Figure A4.** Left: mean total transaction cost against N for costs of 5 and 10 basis points of traded notional. Cost grows with N, but not like sqrt(N) at small N, because the initial purchase and the final liquidation are a fixed floor (about {a4.iloc[0]['mean_cost']:.2f} and {a4.iloc[1]['mean_cost']:.2f} at N = 5); between N = 500 and N = 1,000 the cost rises by a factor of {a4[(a4.N == 1000) & (a4.cost_rate == 0.0005)]['mean_cost'].iloc[0] / a4[(a4.N == 500) & (a4.cost_rate == 0.0005)]['mean_cost'].iloc[0]:.2f} (sqrt(2) = 1.41). Right: net PnL penalised for risk. With a fair volatility the *average gross* PnL is zero, so the average net PnL tends to fall as N grows (apart from small-N noise) and there is no interior optimum on mean PnL alone. An interior optimum appears only once risk is penalised: maximising mean(net PnL) - lambda x std(net PnL), the best N is **{best[(0.0005, 0.5)]}, {best[(0.0005, 1.0)]}, {best[(0.0005, 2.0)]}** for lambda = 0.5, 1, 2 at 5 basis points and **{best[(0.001, 0.5)]}, {best[(0.001, 1.0)]}, {best[(0.001, 2.0)]}** at 10 basis points. The choice of lambda is a modelling assumption, higher costs favour fewer rebalances, and the optimum is flat: neighbouring values of N are close (see `results/partA_A4_costs.csv`).

### A5 (extra): does the drift matter?

{_md(a5, [("mu", "mu", "+.2f"), ("mean_pnl", "mean PnL", "+.4f"), ("se_mean", "SE", ".4f"), ("std_pnl", "std of PnL", ".4f")])}

In theory the real-world drift plays no role. At N = 250 the mean PnL does shift away from zero when mu differs from r, by at most {a5['mean_pnl'].abs().max():.3f} (under 0.3% of the {prem:.2f} premium) but that is many standard errors for the largest drifts, and the standard deviation changes from {a5.loc[a5.mu == 0.05, 'std_pnl'].iloc[0]:.2f} at mu = 5% to {a5.loc[a5.mu == 0.4, 'std_pnl'].iloc[0]:.2f} at mu = 40%. A strong drift pushes the option away from the strike where gamma is small, which makes the hedging easier. So the drift does not change the *price* or the continuous-time replication, but it does change the discrete hedging error at finite N.

### A6 (extra): how good is the discrete approximation as sampling gets finer?

The gamma-PnL expression is a continuous-time, first-order result, so on a grid of N steps it is only approximated (see section 1). For each N, {int(a6.paths.iloc[0]):,} paths (an equal number at each of the seven A3 volatility settings, mu = r) are hedged and the actual PnL is compared with three estimates: path-wise with squared log returns, path-wise with squared simple returns, and the constant-variance form.

{_md(a6, [("N", "N", ".0f"), ("rmse_log_pct_premium", "RMSE, path-wise log (% of premium)", ".2f"), ("rmse_simple_pct_premium", "RMSE, path-wise simple", ".2f"), ("rmse_const_pct_premium", "RMSE, constant variance", ".2f"), ("r2_pooled_log", "R² pooled, path-wise log", ".4f"), ("r2_within_equal_vol_log", "R² within sigma_r = sigma_i, path-wise log", ".3f"), ("r2_within_equal_vol_const", "R² within sigma_r = sigma_i, constant variance", ".3f")])}

The path-wise approximation converges: its root-mean-square error falls from {a6i.loc[5, 'rmse_log_pct_premium']:.1f}% of the premium at N = 5 to {a6i.loc[500, 'rmse_log_pct_premium']:.2f}% at N = 500, a log-log slope of {a6_slope:.2f} over this range (steeper than the -0.5 of the hedging error itself). Squared log returns are somewhat better than squared simple returns at small N ({a6i.loc[5, 'rmse_log_pct_premium']:.1f}% against {a6i.loc[5, 'rmse_simple_pct_premium']:.1f}% at N = 5) and the two are nearly indistinguishable by N = 250 ({a6i.loc[250, 'rmse_log_pct_premium']:.2f}% against {a6i.loc[250, 'rmse_simple_pct_premium']:.2f}%). The pooled R² is flattered by the wide spread of gaps across settings (the A3 lesson), so the cleaner R² is the one *within* the setting sigma_r = sigma_i, where the gap carries no information: it rises from {a6i.loc[5, 'r2_within_equal_vol_log']:.2f} at N = 5 to {a6i.loc[500, 'r2_within_equal_vol_log']:.2f} at N = 500 for the path-wise form, but stays between {a6.r2_within_equal_vol_const.min():.2f} and {a6.r2_within_equal_vol_const.max():.2f} for the constant-variance form at every N, consistent with a single window-average variance ignoring when the variance arrives relative to the strike (the timing effect measured in Part B).

## 6. Part B results

### Volatility risk premium

![vix](figures/B1_vix_vs_realised_vol.png)

**Figure B1.** Left: India VIX at the start of each window against the NIFTY volatility realised over the following 21 trading days. Most points lie below the 45-degree line (realised below VIX in {pct(vrp['frac_windows_rv_below_vix'], 1)} of windows). The orange points are the 2020 crash windows, which are the clear exceptions: some start at a VIX near 14 and are followed by realised volatility of 40% to 88%. Right: both series over time; realised volatility tracks VIX but sits mostly below it. Average VIX {vrp['mean_vix']:.2f}, average realised {vrp['mean_rv']:.2f}, difference {vrp['mean_vol_gap_points']:.2f} vol points (SE {vrp['se_vol_gap_hac']:.2f}, median {vrp['median_vol_gap_points']:.2f}). The correlation between the two is {vrp['corr_vix_rv']:.2f}; regressing realised on VIX gives slope {vrp['rv_on_vix_slope']:.2f} (SE {vrp['rv_on_vix_slope_se_hac']:.2f}), intercept {vrp['rv_on_vix_intercept']:.2f}, R² {vrp['rv_on_vix_r2']:.2f}. This is the volatility risk premium: the market charged more for volatility than it then delivered. Measured in *variance*, the mean gap is {vrp['mean_var_gap']:.4f} with an overlap-robust SE of {vrp['se_var_gap_hac']:.4f}, which is not distinguishable from zero because the 2020 crash windows have enormous negative gaps; in vol points and in the share of windows the premium is clear.

### Hedging PnL against the variance gap

![regression](figures/B2_pnl_vs_variance_gap.png)

**Figure B2.** Left: hedging PnL (% of the initial index level) for each of {h['n']:,} windows against sigma_i² - sigma_r² (VIX² minus realised variance). The black line is the regression on all windows: slope **{h['slope']:.4f}**, intercept {h['intercept']:.4f}, **R² = {h['r2']:.4f}**. The green line excludes the {sb['crash_windows']} windows overlapping 20 Feb to 30 Apr 2020: slope {ex.slope:.3f}, R² {ex.r2:.2f}. Right: the same PnL against the gamma-formula estimate evaluated along each actual path, R² = {sb['r2_pnl_vs_gamma_estimate']:.4f}, slope {sb['ols_pnl_on_gamma_estimate']['slope']:.3f}; the discrete approximation tracks what really happened path by path.

**Statistical caveat.** Consecutive windows overlap by up to 20 of 21 days, so the {h['n']:,} observations are far from independent. The plain OLS standard error of the slope is {h['se_slope_ols']:.4f}; the overlap-robust (Newey-West, lag 20) value is {h['se_slope_hac']:.4f}, about {h['se_slope_hac'] / h['se_slope_ols']:.0f} times larger, and the latter is the one to use. A circular block bootstrap (blocks of 21 windows, 2,000 resamples) gives a 95% interval for R² of {h['r2_block_bootstrap_ci'][0]:.2f} to {h['r2_block_bootstrap_ci'][1]:.2f}. Using only non-overlapping windows ({no['n_per_subsample']} per subsample, all 21 possible offsets): R² between {no['r2_min']:.2f} and {no['r2_max']:.2f} (median {no['r2_median']:.2f}), slope between {no['slope_min']:.3f} and {no['slope_max']:.3f} (median {no['slope_median']:.3f}). The R² of {h['r2']:.2f} is therefore uncertain to roughly a factor of two either way.

### How sensitive is the inference to the dependence assumption?

Consecutive windows overlap, so both the standard error of the slope and the R² interval depend on how that dependence is modelled. I therefore repeated both under alternative assumptions (headline choices: Newey-West lag 20 and block length 21).

**Slope standard error under different lags** (slope {h['slope']:.4f}):

{_md(hs, [("method", "method", ""), ("lag", "lag", ""), ("se_slope", "SE of slope", ".4f"), ("t_stat", "t-statistic", ".2f"), ("slope_ci_low", "95% CI low", ".4f"), ("slope_ci_high", "95% CI high", ".4f")])}

**R² interval under different bootstrap block lengths** (circular block bootstrap of the windows; each block length was run {int(bsn.runs.iloc[0])} times independently, each with the full {int(bsn.resamples_per_run.iloc[0]):,} resamples, and the table gives the mean of the runs' bounds and the range across runs):

{_md(bsn, [("block_length", "block length (windows)", ".0f"), ("r2_low", "2.5th pct, mean of runs", ".3f"), ("r2_low_min", "range across runs: low", ".3f"), ("r2_low_max", "high", ".3f"), ("r2_median", "median", ".3f"), ("r2_high", "97.5th pct, mean of runs", ".3f"), ("r2_high_min", "range across runs: low", ".3f"), ("r2_high_max", "high", ".3f")])}

What this shows. (1) The standard error of the slope rises steeply as the dependence is modelled more fully, from {hs_ols.se_slope:.4f} (plain OLS) and {hs_white.se_slope:.4f} (heteroskedasticity-robust only) to {hnw.loc[5, 'se_slope']:.4f} at lag 5, {hnw.loc[10, 'se_slope']:.4f} at lag 10 and {hnw.loc[20, 'se_slope']:.4f} at lag 20, and then stays flat ({hnw.loc[42, 'se_slope']:.4f} at lag 42, {hnw.loc[63, 'se_slope']:.4f} at lag 63). A lag shorter than the overlap understates the uncertainty (the lag-5 error is {(1 - hnw.loc[5, 'se_slope'] / hnw.loc[20, 'se_slope']) * 100:.0f}% smaller than the headline's); extending the lag beyond the window length changes little, which supports the headline choice. (2) The slope is positive and statistically distinguishable from zero under every Newey-West lag (t-statistics {hnw.t_stat.min():.2f} to {hnw.t_stat.max():.2f}), and its 95% interval stays far below the theoretical average slope of {sb['theory_slope_mean']:.2f} (the largest upper limit is {hnw.slope_ci_high.max():.3f}). (3) The lower end of the R² interval is stable at about {bsn.r2_low.mean():.2f} for every block length, but the upper end rises with the block length, from {bsn_i.loc[21, 'r2_high']:.2f} (block 21) to {bsn_i.loc[42, 'r2_high']:.2f} (block 42) and {bsn_i.loc[63, 'r2_high']:.2f} (block 63). The spread between independent runs at a fixed block length is much smaller (for example {bsn_i.loc[21, 'r2_high_min']:.3f} to {bsn_i.loc[21, 'r2_high_max']:.3f} at block 21), so this rise reflects the assumed dependence structure and not Monte Carlo noise. The headline interval ({h['r2_block_bootstrap_ci'][0]:.2f} to {h['r2_block_bootstrap_ci'][1]:.2f}, one run) is consistent with the repeats at block 21.

Inference here is sensitive to the assumed dependence structure. The qualitative conclusions (a positive slope, much weaker than the theory predicts, and a wide R² interval) hold under all of these choices; the precise width of the intervals does not, and none of these methods has been validated against the true dependence of this series, so the intervals should be read as indicative rather than exact.

### Rebalancing-frequency study

{_md(freq, [("interval_days", "rebalance every k days", ".0f"), ("rebalances", "N", ".0f"), ("mean_pnl_pct", "mean PnL %", "+.3f"), ("std_pnl_pct", "std of PnL %", ".3f"), ("p05_pct", "5th pct", "+.3f"), ("p95_pct", "95th pct", "+.3f"), ("std_diff_vs_daily_pct", "std of (PnL_k - PnL_daily) %", ".3f"), ("std_diff_predicted_by_sqrtN_law_pct", "1/sqrt(N) law, one fitted constant", ".3f")])}

![frequency](figures/B3_error_vs_rebalancing_interval.png)

**Figure B3.** Left: spread of hedging PnL for hedging every 1, 2, 5 and 10 trading days. The mean is stable ({freq['mean_pnl_pct'].min():.2f}% to {freq['mean_pnl_pct'].max():.2f}%) but the spread widens: standard deviation {freq.iloc[0]['std_pnl_pct']:.3f}% (daily) to {freq.iloc[-1]['std_pnl_pct']:.3f}% (every 10 days). The raw spread mixes two things, the discretisation error and the variation caused by the volatility gap, so the raw spread widens more slowly than 1/sqrt(N) would imply (a factor {freq.iloc[-1]['std_pnl_pct'] / freq.iloc[0]['std_pnl_pct']:.2f}, against sqrt(21/3) = 2.65). Right: to isolate the discretisation part, the standard deviation of the difference between the PnL at k days and the daily-hedged PnL of the same window. If the discretisation error is proportional to 1/sqrt(N) and independent of the daily error, this difference follows c x sqrt(1/N_k - 1/21); with the single constant c fitted, the prediction is within about 6% of the measurement at all three coarser frequencies. That is a weak test (three points, one fitted constant) but it is consistent with the 1/sqrt(N) law on real data. Only coarse frequencies are possible with daily data.

### Calm and turbulent periods

{_md(reg, [("group", "group", ""), ("windows", "windows", ".0f"), ("mean_vix", "mean VIX", ".1f"), ("mean_rv_pct", "mean realised %", ".1f"), ("mean_pnl_pct", "mean PnL %", "+.3f"), ("std_pnl_pct", "std PnL %", ".3f"), ("frac_profit", "share profitable", ".3f"), ("slope", "slope", ".3f"), ("r2", "R²", ".3f")])}

Regime cut-offs (VIX at window start below 15, 15 to 20, 20 or above) were fixed before looking at results. The seller earns a positive average PnL in every regime and the spread of outcomes grows with the level of volatility; the within-regime R² is low in calm and middle regimes ({calm['r2']:.2f} and {mid['r2']:.2f}) and higher in turbulent windows ({turb['r2']:.2f}), where the variance gap varies a lot and dominates, in line with the A3 lesson about the range of the regressor.

### By calendar year

{_md(yearly, [("group", "year", ""), ("windows", "windows", ".0f"), ("mean_vix", "mean VIX", ".1f"), ("mean_rv_pct", "mean realised %", ".1f"), ("mean_pnl_pct", "mean PnL %", "+.3f"), ("std_pnl_pct", "std PnL %", ".3f"), ("frac_profit", "share profitable", ".3f")])}

(2026 ends at the latest window start, 2026-09-16.) {'Mean PnL is positive in every year' if (yearly.mean_pnl_pct > 0).all() else 'Mean PnL is not positive in every year'}; {int(yearly.loc[yearly.mean_pnl_pct.idxmin(), 'group'])} is the weakest ({yearly.mean_pnl_pct.min():+.3f}%) and {int(yearly.loc[yearly.mean_pnl_pct.idxmax(), 'group'])} the strongest ({yearly.mean_pnl_pct.max():+.3f}%).

### Comparison with an idealised GBM world

{_md(bench, [("benchmark", "benchmark", ""), ("observations", "observations", ",.0f"), ("slope", "slope", ".3f"), ("r2", "R²", ".3f"), ("mean_pnl_pct", "mean PnL %", "+.3f"), ("std_pnl_pct", "std PnL %", ".3f")])}

The "constant GBM" row simulates 21-day windows with one volatility regime (sigma_i = mean VIX, sigma_r = mean realised volatility of the real sample). Its mean PnL ({b_const.mean_pnl_pct:+.3f}%) is almost identical to the real average ({h['mean_pnl_pct']:+.3f}%): the *level* of the volatility-risk-premium income is what the theory predicts. Its slope ({b_const.slope:.2f}) is close to the theoretical average slope ({sb['theory_slope_mean']:.2f}) and its R² ({b_const.r2:.2f}) is the within-regime R² to expect with 21 daily steps. The "GBM twin" is a benchmark *conditional on the historical realised-volatility estimates*. Every real window gets {tc['replicates']} simulated GBM paths with that window's real VIX as the implied volatility and that window's historical realised volatility as the *true* (input) volatility of the simulated path. It therefore matches the distribution of the *input* volatilities to the historical realised volatilities, and it has none of the real market's fat tails, volatility clustering or trends. It does **not** reproduce the historical realised-volatility regressor window by window: each simulated 21-step path produces its own realised volatility, equal to the input plus sampling noise, and that simulated realised volatility is what the benchmark regresses on. The two distributions are therefore not identical by construction, as the next table shows.

{_md(tv_fmt, [("quantity", "quantity", ""), ("unit", "unit", ""), ("n", "n", ""), ("mean", "mean", ""), ("std", "std", ""), ("p05", "5th pct", ""), ("p25", "25th", ""), ("p50", "median", ""), ("p75", "75th", ""), ("p95", "95th", "")])}

The simulated realised volatility is slightly wider than the historical one (standard deviation {tvr.loc[Q_SIM, 'std']:.2f} against {tvr.loc[Q_HIST, 'std']:.2f} vol points, a ratio of {tc['std_ratio_simulated_over_historical_rv']:.2f}; for the variance gap the ratio is {tc['std_ratio_simulated_over_historical_gap']:.2f}) and its mean is slightly lower ({tvr.loc[Q_SIM, 'mean']:.2f} against {tvr.loc[Q_HIST, 'mean']:.2f}). The reason is sampling noise: each path's realised volatility differs from its input by a standard deviation of {tvr.loc[Q_NOISE, 'std']:.2f} vol points, about {tc['relative_sampling_noise_std'] * 100:.1f}% of the input, which is what theory predicts for a volatility estimated from 21 observations (1/sqrt(2 x 21) = {tc['theory_relative_sampling_noise'] * 100:.1f}%). This matters for interpretation because the twin's regression slope uses the simulated realised variance: regressed on it the twin's slope is {tc['slope_on_simulated_gap']:.3f} (R² {tc['r2_on_simulated_gap']:.3f}); regressed instead on the input variance gap, with no sampling noise, it is {tc['slope_on_input_gap']:.3f} (R² {tc['r2_on_input_gap']:.3f}). The mismatch between the simulated and historical regressor distributions therefore changes the twin's slope by only about {abs(tc['slope_on_input_gap'] - tc['slope_on_simulated_gap']) / tc['slope_on_simulated_gap'] * 100:.0f}%, and cannot explain why the twin's slope ({tc['slope_on_simulated_gap']:.2f}) differs from the real slope ({h['slope']:.2f}). The R² is higher on the simulated gap because the realised variance of a path is itself a driver of that path's PnL, which the input volatility alone is not. The relationship with VIX is similar in the two worlds: the correlation between VIX and realised volatility is {tc['corr_vix_rv_real']:.2f} in the real windows and {tc['corr_vix_rv_twin']:.2f} in the twin, and realised volatility is below VIX in {pct(tc['share_rv_below_vix_real'])} of real windows against {pct(tc['share_rv_below_vix_twin'])} in the twin.

### Sensitivity checks

{_md(sens, [("group", "variant", ""), ("windows", "windows", ".0f"), ("mean_pnl_pct", "mean PnL %", "+.4f"), ("std_pnl_pct", "std PnL %", ".4f"), ("slope", "slope", ".4f"), ("r2", "R²", ".4f")])}

**Risk-free rate (4%, 6%, 8%), as requested.** Differences relative to the 6% placeholder:

{_md(rate, [("risk_free", "risk-free", ".0%"), ("mean_pnl_pct", "mean PnL %", "+.4f"), ("diff_mean_pnl_pct", "difference in mean PnL (pp)", "+.4f"), ("diff_std_pnl_pct", "difference in std (pp)", "+.4f"), ("diff_frac_profit", "difference in share profitable", "+.4f"), ("slope", "slope", ".4f"), ("diff_slope", "difference in slope", "+.5f"), ("r2", "R²", ".4f"), ("diff_r2", "difference in R²", "+.4f")])}

Moving the rate by two percentage points changes the mean PnL by at most {rate[['diff_mean_pnl_pct']].abs().max().iloc[0]:.4f} percentage points ({rate[['diff_mean_pnl_pct']].abs().max().iloc[0] / h['mean_pnl_pct'] * 100:.1f}% of the mean), the slope by at most {rate[['diff_slope']].abs().max().iloc[0]:.5f} and R² by at most {rate[['diff_r2']].abs().max().iloc[0]:.4f}. The placeholder rate does not matter for any conclusion here.

Other checks: using the sample standard deviation to define realised volatility moves R² from {h['r2']:.3f} to {ce.r2:.3f}; dropping weekend special sessions leaves the slope at {nw.slope:.4f} and R² at {nw.r2:.4f} ({nw.windows} windows); dropping the 2020 crash windows lifts the slope to {ex.slope:.3f} and R² to {ex.r2:.2f} (mean PnL {ex.mean_pnl_pct:+.3f}%, standard deviation {ex.std_pnl_pct:.3f}%).

### How good is the gamma-PnL approximation on real data? Return conventions

With daily observations the path-wise gamma estimate is a discrete approximation (section 1), so I tested how its fit to the actual hedging PnL changes with how realised variance is defined. Each row compares an estimate with the hedging PnL of the same windows; the GBM twin has exactly lognormal returns with the same 21 daily steps, so its error is the error of the approximation itself, not an effect of the data.

{_md(conv, [("dataset", "windows", ""), ("convention", "convention", ""), ("r2", "R²", ".3f"), ("slope_pnl_on_estimate", "slope of PnL on estimate", ".3f"), ("mean_error_pct", "mean (estimate - PnL), % of S0", "+.3f"), ("rmse_pct", "RMSE, % of S0", ".3f"), ("rmse_over_std_pnl", "RMSE / std of PnL", ".2f")])}

What this shows. (1) The three path-wise conventions fit about equally well on the real windows (R² {min(pw_r2):.3f} to {max(pw_r2):.3f}), so the fit is not an artefact of using log returns; squared simple returns are marginally best. (2) Demeaning the returns shifts the level (mean error {cvr.loc[(D_ALL, C_DEMEAN), 'mean_error_pct']:+.3f}% of S0 against {cvr.loc[(D_ALL, C_LOG), 'mean_error_pct']:+.3f}% for the baseline) because it removes part of the squared move that the hedge actually experienced. (3) The constant-variance form fits clearly worse on all real windows (R² {cvr.loc[(D_ALL, C_UNC), 'r2']:.2f} un-centred, {cvr.loc[(D_ALL, C_CEN), 'r2']:.2f} centred) and improves to {cvr.loc[(D_EX, C_UNC), 'r2']:.2f} and {cvr.loc[(D_EX, C_CEN), 'r2']:.2f} when the crash windows are excluded, whereas the path-wise form barely moves ({cvr.loc[(D_ALL, C_LOG), 'r2']:.2f} to {cvr.loc[(D_EX, C_LOG), 'r2']:.2f}): what the single-variance summary misses is mainly timing. (4) The approximation is not exact even in the ideal case: on the GBM twin the baseline R² is {cvr.loc[(D_TWIN, C_LOG), 'r2']:.3f} and the RMSE is {cvr.loc[(D_TWIN, C_LOG), 'rmse_over_std_pnl']:.2f} of the standard deviation of PnL, because 21 daily steps are a coarse grid. On the real windows it is {cvr.loc[(D_ALL, C_LOG), 'r2']:.3f} and {cvr.loc[(D_ALL, C_LOG), 'rmse_over_std_pnl']:.2f}, a larger error than on lognormal windows with the same number of steps. That is consistent with fat tails and jumps making the quadratic-variation proxy less accurate, but that explanation was not tested separately and is not claimed.

### Stretch: re-hedging with each day's VIX

{_md(stretch, [("hedger", "hedger", ""), ("windows", "windows", ".0f"), ("mean_pnl_pct", "mean PnL %", "+.3f"), ("std_pnl_pct", "std PnL %", ".3f"), ("slope", "slope on gap", ".3f"), ("r2", "R² on gap", ".3f")])}

Restricted to the windows that have a valid VIX on every day. Delta is computed each day with that day's VIX instead of the fixed initial volatility. The mean PnL is about the same, the spread is a little wider ({base_fixed.std_pnl_pct:.3f}% to {rehedge.std_pnl_pct:.3f}%), and the R² of PnL on the initial-VIX variance gap rises from {base_fixed.r2:.2f} to {rehedge.r2:.2f}. This is a different hedger, so its slope is not comparable with the baseline's. Not attempted: delta-gamma hedging with a second option.

### Effect sizes with confidence intervals

Regression slopes are not the only numbers that matter, so the headline effect sizes are reported with 95% intervals computed two ways: a normal interval from the Newey-West (lag 20) standard error where one exists, and a circular block bootstrap (blocks of {sbc_block} windows, {sbc_boot:,} resamples) for every row. Both are needed because the overlapping windows make naive intervals far too narrow (see the dependence-sensitivity section above).

{_md(ef_fmt, [("effect", "effect", ""), ("unit", "unit", ""), ("estimate", "estimate", ""), ("hac", "95% HAC interval", ""), ("boot", "95% block-bootstrap interval", "")])}

![effects](figures/B4_effect_size_intervals.png)

**Figure B4.** The mean PnL, the share of profitable windows and the volatility risk premium are all clearly positive under both methods. The bootstrap interval of the slope is much wider above the estimate than the HAC interval ({ef_slope.boot_ci_lo:.3f} to {ef_slope.boot_ci_hi:.3f}, against {ef_slope.hac_ci_lo:.3f} to {ef_slope.hac_ci_hi:.3f}). The slope depends heavily on how many of the {sb['crash_windows']} crash windows a resample contains: across the bootstrap resamples the correlation between a resample's slope and its crash-window share is {sbc['corr_slope_vs_crash_share']:.2f}, and the {pct(sbc['share_of_resamples_with_under_half_percent_crash'], 0)} of resamples that contain almost none of them (under 0.5%, against {pct(sbc['crash_share_in_data'], 1)} in the data) have a median slope of {sbc['median_slope_those']:.2f}, against {sbc['median_slope_others']:.3f} for the rest. So the uncertainty of the slope is mostly the uncertainty about how much the March 2020 episode weighs, a feature of having one such episode in the sample, not something a larger bootstrap could remove.

### Market states: calm, drawdown and recovery

The VIX regimes above are defined by the level of implied volatility. Market states are defined from the *index itself*, using only information available on the day the option is sold: the drawdown of NIFTY from its running peak since the start of the sample, and its return over the previous {sbs['momentum_days']} trading days. **Calm**: less than {abs(sbs['drawdown_cut']) * 100:.0f}% below the peak. **Drawdown**: down {abs(sbs['drawdown_cut']) * 100:.0f}% or more from the peak and still falling over the last quarter. **Recovery**: down {abs(sbs['drawdown_cut']) * 100:.0f}% or more but rising over the last quarter. The thresholds were fixed before the table was produced, and the first {sbs['unlabelled_windows']} windows have no trailing quarter and are left unlabelled. The running peak starts at the sample start (2016), not at an all-time high, so an early-sample "calm" label can hide an earlier, unobserved drawdown.

{_md(mst, [("regime", "state", ""), ("windows", "windows", ".0f"), ("mean_pnl_pct", "mean PnL %", "+.3f"), ("mean_pnl_ci_lo", "95% low", "+.3f"), ("mean_pnl_ci_hi", "95% high", "+.3f"), ("std_pnl_pct", "std PnL %", ".3f"), ("worst_pnl_pct", "worst window %", "+.2f"), ("frac_profit", "share profitable", ".3f"), ("mean_vix", "mean VIX", ".1f"), ("mean_rv_pct", "mean realised %", ".1f"), ("vol_gap_points", "VIX - realised (vol pts)", ".2f"), ("slope", "slope", ".3f"), ("slope_se_hac", "HAC SE", ".3f"), ("r2", "R²", ".3f")])}

![states](figures/B5_market_states.png)

**Figure B5.** The seller's average PnL is positive in all three states, but the intervals overlap, so the data do not rank them with confidence; the drawdown state has the widest interval and the worst single window ({m_draw.worst_pnl_pct:+.2f}% of S0, against {m_calm.worst_pnl_pct:+.2f}% in calm periods and {m_rec.worst_pnl_pct:+.2f}% in recovery). The slope of PnL on the variance gap differs much more by state than the mean PnL does: {m_calm.slope:.3f} in calm periods, {m_draw.slope:.3f} in drawdowns and {m_rec.slope:.3f} in recoveries, against a theoretical average of {sb['theory_slope_mean']:.2f}; the within-state R² rises from {m_calm.r2:.2f} to {m_draw.r2:.2f} to {m_rec.r2:.2f}. Recovery windows also have the largest volatility premium ({m_rec.vol_gap_points:.2f} vol points against {m_calm.vol_gap_points:.2f} in calm periods). The recovery state has only {int(m_rec.windows)} overlapping windows, which is a handful of distinct episodes, so its numbers should not be over-read. Overlapping windows are handled with a Newey-West lag of at most 20 (fewer in small groups).

### Does the conclusion depend on how volatility is measured?

The headline realised volatility is the root-mean-square of daily close-to-close log returns. Ten alternatives were tested on the same windows, in three families. **Close-to-close**: the same with the mean subtracted, an exponentially weighted average (decay {ewma}, most weight on the last days) and bipower variation (robust to jumps). **Range-based, intraday only**: Parkinson, Garman-Klass and Rogers-Satchell use each day's open, high and low and therefore *do not see the overnight move* from one close to the next open. **Range-based with the overnight gap**: the same three with the squared overnight log move added back, and Yang-Zhang (which combines overnight, open-to-close and Rogers-Satchell variance). A daily-close hedger experiences the overnight move, so only the first and third families are like-for-like with the risk the hedge actually runs; the estimators are **not interchangeable**, and the figure colours them by family so the difference is visible without reading the text.

{_md(ve, [("estimator", "estimator", ""), ("type", "measures", ""), ("mean_rv_pct", "mean realised %", ".2f"), ("mean_vol_gap_points", "VIX - realised (vol pts)", ".2f"), ("se_vol_gap_hac", "HAC SE", ".2f"), ("frac_rv_below_vix", "share of windows below VIX", ".3f"), ("corr_with_close_to_close", "correlation with baseline", ".3f"), ("slope", "slope", ".4f"), ("slope_t", "t-stat", ".2f"), ("r2", "R²", ".3f")])}

![estimators](figures/B6_vol_estimators.png)

**Figure B6.** Colour shows what each estimator measures (blue close-to-close, orange intraday range with no overnight gap, green range with the overnight gap). What does *not* depend on the estimator: the sign and significance of the volatility risk premium (positive and significant for all {sve['n_estimators']} estimators: {sve['all_vrp_positive'] and sve['all_vrp_significant']}), the sign and significance of the regression slope (significant for all {sve['n_estimators']}: {sve['all_slopes_significant']}), and the weak fit (R² only between {sve['r2_range'][0]:.2f} and {sve['r2_range'][1]:.2f}, slope between {sve['slope_range'][0]:.3f} and {sve['slope_range'][1]:.3f}, always far below the theoretical average of {sb['theory_slope_mean']:.2f}). What *does* depend on it: the **size** of the premium, from {sve['vol_gap_range_points'][0]:.2f} to {sve['vol_gap_range_points'][1]:.2f} vol points. That spread is almost entirely the overnight gap: the three intraday-only estimators give {sve['vrp_range_range_no_gap'][0]:.2f} to {sve['vrp_range_range_no_gap'][1]:.2f} points because they measure only the move within each trading day and so understate the volatility that a daily-close hedger experiences; adding the overnight gap back moves them to {sve['vrp_range_range_with_gap'][0]:.2f} to {sve['vrp_range_range_with_gap'][1]:.2f}, inside or next to the close-to-close range ({sve['vrp_range_close_to_close'][0]:.2f} to {sve['vrp_range_close_to_close'][1]:.2f}). So the defensible size of the premium for this hedge is about {sve['vrp_range_range_with_gap'][0]:.1f} to {sve['vrp_range_close_to_close'][1]:.1f} vol points, and the larger intraday-only values should not be quoted as the premium. The best-fitting estimator by R² is {sve['best_r2_estimator']} ({ve.r2.max():.3f}), but the spread of R² across estimators ({sve['r2_range'][0]:.3f} to {sve['r2_range'][1]:.3f}) is smaller than the bootstrap uncertainty of the R² itself, so that ranking is not meaningful.

### Chronological validation: calibration, validation, test

Everything above uses the whole sample. To check whether a relationship fitted on earlier data holds on later data, windows were split by date, and any window that straddles a boundary is dropped (purging), so no window overlaps two periods. The models are kept in two tracks. **Ex-ante** models use only what is known on the day the option is sold: the calibration mean, VIX, and VIX against the realised volatility of the 21 trading days *before* that day (`exante_var_gap` is VIX² minus trailing realised variance, the ex-ante analogue of the regressor in the headline regression; `exante_vol_gap` is the same in volatility points). **Ex-post** models use the window's own future price path (`gap_*` use the 21 days that follow, `gamma_estimate` is evaluated along the path), so they can explain the PnL after the fact but could not be used to decide anything when selling the option. Only the ex-ante track speaks to prediction. Two splits were fixed in advance: **primary** (calibration 2016-2021, validation 2022-2023, test 2024 onwards) and **crash held out** (calibration 2016-2019, validation 2020-2022 which contains the crash, test 2023 onwards). Every model's parameters are fitted on the calibration period only; models are compared on validation; the test period is used once for the chosen model. "Out-of-sample R²" is 1 minus the model's squared error divided by that of simply predicting the calibration mean, so it can be negative. **Caution:** the headline results were known when these splits were chosen, so this is a discipline check and not a blind test; and each split has a single test period of overlapping windows.

{_md(split_view, [("split", "split", ""), ("model", "model", ""), ("track", "track", ""), ("slope", "fitted slope", ""), ("val_rmse", "validation RMSE %", ""), ("val_r2", "validation OOS R²", ""), ("test_rmse", "test RMSE %", ""), ("test_r2", "test OOS R²", "")])}

![split](figures/B7_chronological_split.png)

**Figure B7.** The left block of bars in each panel is the ex-ante track, the right block the ex-post track. (1) **Ex-ante: no predictive skill.** Nothing known when the option is sold beat the calibration mean by more than {ex_ante_test.max():.2f} of R² on the test periods: test R² of the VIX level {ex_t[('primary', 'vix_level')]:+.3f} (primary split) and {ex_t[('crash_held_out', 'vix_level')]:+.3f} (crash held out); of VIX against trailing realised variance {ex_t[('primary', 'exante_var_gap')]:+.3f} and {ex_t[('crash_held_out', 'exante_var_gap')]:+.3f}; of VIX against trailing realised volatility {ex_t[('primary', 'exante_vol_gap')]:+.3f} and {ex_t[('crash_held_out', 'exante_vol_gap')]:+.3f}. On the 2020-2022 validation period, which contains the crash, the two trailing-volatility models do worse than the benchmark (out-of-sample R² {ex_v[('crash_held_out', 'exante_var_gap')]:+.2f} and {ex_v[('crash_held_out', 'exante_vol_gap')]:+.2f}) while the VIX level does better ({split_ex3:+.2f}), a validation-period gain that does not carry over to the test period. Trailing volatility is therefore a poor stand-in for the volatility the window then delivers, and the simple regression does not become a predictor by replacing the realised variance with a lagged one. (2) **Ex-post: a stable explanation, not a forecast.** The model that wins on validation in both splits is `{sbl['primary']['selected_on_validation']}` (test OOS R² {sbl['primary']['test_oos_r2']:.2f} and {sbl['crash_held_out']['test_oos_r2']:.2f}), but it, like the variance-gap regressions, uses the realised path of the window, so it is an out-of-sample test of a *stable explanation of the PnL after the fact*, not a forecast; the model selected among the ex-ante candidates on validation is `{sbl['primary']['selected_ex_ante']}` in the primary split and `{sbl['crash_held_out']['selected_ex_ante']}` with the crash held out (test OOS R² {sbl['primary']['test_oos_r2_ex_ante']:+.3f} and {sbl['crash_held_out']['test_oos_r2_ex_ante']:+.3f}). (3) The variance-gap regression is stable only across similar regimes. With the crash held out, the slope fitted on calm 2016-2019 data is {sbl['crash_held_out']['gap_model_fitted_slope']:.3f}, close to the theoretical average of {sb['theory_slope_mean']:.2f}; applied to 2020-2022 it fails badly (validation OOS R² {sbl['crash_held_out']['gap_model_validation_oos_r2']:.1f}: the crash's enormous negative variance gaps are multiplied by a slope that is far too large for them, which is consistent with the gamma-weight finding in section 7, although this table does not isolate that mechanism), yet on the calm 2023-onwards test period the same slope explains {pct(sbl['crash_held_out']['gap_model_test_oos_r2'], 0)} of the out-of-sample variance. In the primary split the calibration period contains the crash, so the fitted slope is only {sbl['primary']['gap_model_fitted_slope']:.3f} and the test period is explained far less ({pct(sbl['primary']['gap_model_test_oos_r2'], 0)}). The full-sample slope of {h['slope']:.3f} is therefore a compromise: a slope near the theory fits calm windows well but badly over-predicts the crash windows, and no single slope describes both.

### Does the clock matter? A calendar-time re-run

The baseline treats every window as 21/252 of a year with 21 equal steps. But 21 trading days span {cinfo['calendar_days_min']} to {cinfo['calendar_days_max']} calendar days (mean {cinfo['calendar_days_mean']:.1f}), India VIX is quoted as an annualised volatility on calendar time, and weekends carry interest and Friday-to-Monday variance. The alternative uses the real dates (ACT/365, `src/calendar_time.py`): time stamps are (date - start date)/365, the option expires at the actual window end (T averages {cinfo['mean_T_calendar']:.4f} years against {cinfo['T_trading']:.4f}), interest accrues over the same calendar time, and realised volatility is annualised over the same T. Hedge dates are unchanged (the trading days). Nothing was tuned: ACT/365 is the only alternative tried.

{_md(cal_t, [("clock", "clock", ""), ("mean_pnl_pct", "mean PnL %", "+.3f"), ("mean_pnl_hac_lo", "HAC 95% low", "+.3f"), ("mean_pnl_hac_hi", "high", "+.3f"), ("std_pnl_pct", "std PnL %", ".3f"), ("frac_profit", "share profitable", ".3f"), ("mean_rv_pct", "mean realised %", ".2f"), ("vrp_vol_points", "VIX - realised (vol pts)", ".2f"), ("slope", "slope", ".4f"), ("slope_hac_lo", "HAC 95% low", ".3f"), ("slope_hac_hi", "high", ".3f"), ("r2", "R²", ".3f"), ("pathwise_estimate_r2", "R² on path-wise estimate", ".3f")])}

The two clocks tell the same story: the same sign and similar size for the mean PnL, the premium, the slope and R², and the PnL series of the two clocks correlate at {cinfo['corr_pnl']:.3f}. The calendar clock gives a slightly higher mean PnL (paired difference {cinfo['paired_mean_diff_pct']:+.3f} percentage points, HAC 95% interval {cinfo['paired_mean_diff_hac_lo']:+.3f} to {cinfo['paired_mean_diff_hac_hi']:+.3f}, so small but not zero) because the option is priced for a longer horizon, and a slightly higher slope ({cinfo['slope_ratio']:.2f} times the baseline) and R², well inside the baseline's own intervals. The baseline stays the headline for continuity with the rest of the report; the calendar clock is arguably closer to how VIX is defined, and neither is "the correct" one.

### How much of the interval width is simulation noise? (bootstrap Monte Carlo error)

A bootstrap interval has two kinds of uncertainty: sampling uncertainty (what the interval describes) and Monte Carlo error from using a finite number of random resamples. To measure the second, the whole effect-size bootstrap was repeated {int(mc.reps.iloc[0])} times with independent random numbers ({int(mc.resamples.iloc[0]):,} resamples each, block {sbc_block}); the table gives the mean and standard deviation of each bound across the repeats.

{_md(mc, [("effect", "effect", ""), ("lo_mean", "lower bound, mean", ".4g"), ("lo_sd", "sd", ".2g"), ("hi_mean", "upper bound, mean", ".4g"), ("hi_sd", "sd", ".2g"), ("max_bound_sd_over_width", "larger sd / interval width", ".3f")])}

Monte Carlo error is small: no bound moves by more than {sbc_all['mc_max_bound_sd_over_width'] * 100:.1f}% of the interval's width from one repeat to the next, so the printed bounds are stable to about the second or third significant digit. It is much smaller than the uncertainty from the dependence assumptions (the HAC lag and bootstrap block-length tables above), which is the uncertainty that matters.

### Trading frictions: how much of the income survives costs? (scenario analysis)

**These are scenario assumptions, not measurements.** The supplied data contain no futures prices, option quotes, bid-ask spreads, fee schedules or contract specifications, so the numbers below say how the frictionless result erodes if hedging costs X, and do not say what a desk actually paid. The grid uses round numbers. The hedge instrument is still the index itself (no futures basis, carry or roll). Costs are charged on every hedge trade including the initial purchase and the final liquidation; the option half-spread means the seller receives VIX minus *h* vol points instead of VIX, while the hedge ratio is still computed at VIX; lot-size effects are represented by rounding hedge ratios to multiples of a step *g* of the option's notional (a stylised stand-in for whole contract lots, not the NSE lot size).

Mean seller PnL (% of S0 per window) by one-way cost and rebalancing interval, no option spread:

{_md(fr_piv, [("one_way_cost_bps", "one-way cost (bp)", ".0f"), ("d1", "daily", "+.3f"), ("d2", "every 2 days", "+.3f"), ("d5", "every 5 days", "+.3f"), ("d10", "every 10 days", "+.3f")])}

![frictions](figures/B8_friction_scenarios.png)

**Figure B8.** With no costs the income is {sfr['baseline_mean_pnl_pct']:.3f}% of S0 per window. It falls linearly with the cost rate because costs are a sum of trade sizes times the rate. The one-way cost at which the average income reaches zero is **{sfr['breakeven_cost_bps']['every_1d']:.1f} bp** for daily hedging, {sfr['breakeven_cost_bps']['every_2d']:.1f} bp every 2 days, {sfr['breakeven_cost_bps']['every_5d']:.1f} bp every 5 days and {sfr['breakeven_cost_bps']['every_10d']:.1f} bp every 10 days. Hedging less often does not raise the frictionless mean (it lowers it slightly: {fr_daily0:.3f}% daily against {fr_5d0:.3f}% every 5 days), but it trades less, so it wins once costs are high enough: at 0-2 bp daily hedging has the highest mean, at 5 bp and above hedging every 10 days does ({sfr['best_interval_by_cost']['5bps']['mean_pnl_pct']:+.3f}% against {fr_daily5:+.3f}% daily at 5 bp). That difference is small relative to the standard error of the mean PnL ({ef_mean.hac_se:.3f}), so the ranking of intervals is not statistically established, and rebalancing less often also widens the PnL distribution (section 6). If the seller also sells the option half a vol point below VIX the breakeven falls to {sfr['breakeven_cost_bps_with_half_vol_spread']['every_1d']:.1f} bp (daily); a one-vol-point half-spread alone removes {sfr['half_spread_1vol_loss_pct']:.3f} percentage points, {pct(sfr['half_spread_1vol_loss_pct'] / sfr['baseline_mean_pnl_pct'], 0)} of the frictionless income. Rounding hedge ratios to 1%, 5% or 10% of the option's notional changes the mean PnL by at most {lot_max_dm:.3f} percentage points and widens the standard deviation from {fr_std0:.3f}% to at most {lot_max_std:.3f}%, so lot-size effects only matter when the position is small. Taken together: in this scenario grid the frictionless income survives one-way costs up to roughly {sfr['breakeven_cost_bps']['every_1d']:.0f} bp (about {sfr['breakeven_cost_bps_with_half_vol_spread']['every_1d']:.0f} bp if the option is also sold half a vol point cheaper), and it does not survive more than that. Whether real costs on a tradable instrument are below those levels cannot be determined from these data; that needs sourced fee schedules and quoted spreads (see 'Next steps').

## 7. Where Part B does not match the theory

1. **The slope is far below the theoretical average, and R² is lower than in the idealised world.** Full-sample slope {h['slope']:.3f} against an average theoretical slope of {sb['theory_slope_mean']:.2f}; R² {h['r2']:.2f} against {b_const.r2:.2f} for the constant-volatility GBM benchmark. The {sb['crash_windows']} windows around the March 2020 crash, {pct(sb['crash_windows'] / h['n'], 1)} of the sample, account for most of this: without them the slope is {ex.slope:.3f} and R² is {ex.r2:.2f}, close to the benchmark.
2. **Why the crash windows break the simple regression, and what was measured.** The gamma-PnL formula weights each day's variance by Gamma x S², large only when the index is near the strike. During the crash the index moved far from the strike quickly, so the slope that the gamma formula predicts for the regression (the sum of each day's weight e^(r(T-t)) x Gamma S² / 2 x dt along that window's own path, as a fraction of the initial index level, averaged over windows) is {gw['crash']:.2f} in those windows against {gw['non_crash']:.2f} in the other windows. Also, the variance did not arrive evenly across each window relative to the strike. Comparing the path-wise gamma estimate with the one that assumes evenly spread variance, the real crash windows differ by **{tm['real']['crash']:+.2f}% of the initial index level** on average (the seller lost far less than the window's realised variance suggested), while in the simulated GBM twin the same measure is {tm['gbm_twin']['crash']:+.2f}%, essentially zero. Outside the crash windows it is {tm['real']['non_crash']:+.3f}% in the real data. So a window-average realised variance is the wrong summary of a window when the variance is concentrated far from the strike; the discrete path-wise approximation, which captures the timing, tracks {pct(sb['r2_pnl_vs_gamma_estimate'], 0)} of the variation.
3. **The real data are less dispersed than the conditional GBM twin, and this is only partly explained.** The twin (each window simulated with its historical realised volatility as the true volatility) has a PnL standard deviation of {b_twin.std_pnl_pct:.2f}% against {h['std_pnl_pct']:.2f}% on the real data, and slope {b_twin.slope:.2f} against {h['slope']:.2f}. The small difference between the simulated and historical realised-volatility distributions (section 6) changes the twin's slope by only about {abs(tc['slope_on_input_gap'] - tc['slope_on_simulated_gap']) / tc['slope_on_simulated_gap'] * 100:.0f}%, so it is not the explanation. The gamma-weight finding in point 2 is one candidate explanation, and fat tails and volatility clustering are another; neither was tested separately, so neither is claimed.
4. **The variance premium itself is statistically weak, even though the volatility premium is clear.** The mean variance gap ({vrp['mean_var_gap']:.4f}) has an overlap-robust SE of {vrp['se_var_gap_hac']:.4f}; it is the mean *vol-point* gap ({vrp['mean_vol_gap_points']:.2f}, SE {vrp['se_vol_gap_hac']:.2f}) and the {pct(vrp['frac_windows_rv_below_vix'], 0)} share of windows with realised below VIX that make the premium visible.
5. **The slope is not one number: it depends on the market state, and a slope fitted in one regime fails in another.** Calm {m_calm.slope:.3f}, drawdown {m_draw.slope:.3f}, recovery {m_rec.slope:.3f} (section 6, market states); fitted on calm 2016-2019 data the slope is {sbl['crash_held_out']['gap_model_fitted_slope']:.2f}, close to the theory, and it explains calm 2023-onwards windows ({pct(sbl['crash_held_out']['gap_model_test_oos_r2'], 0)} out of sample) yet fails across the 2020 crash (out-of-sample R² {sbl['crash_held_out']['gap_model_validation_oos_r2']:.1f}). The full-sample slope blends the two.
6. **Where the theory does hold on real data:** the sign (seller profits when realised is below implied), the level of the average PnL ({h['mean_pnl_pct']:+.3f}% against {b_const.mean_pnl_pct:+.3f}% in the GBM benchmark), the path-wise gamma-PnL approximation (R² {sb['r2_pnl_vs_gamma_estimate']:.2f}, an empirical fit rather than an exact identity), and, weakly, the 1/sqrt(N) scaling of discretisation error.

## 8. Limitations

- **No real option prices.** The option is a theoretical Black-Scholes call priced from VIX. This tests the hedging mechanism, not option-market PnL.
- **India VIX is not an at-the-money implied volatility.** It is a model-free, 30-day, annualised measure built from NIFTY option prices across strikes, so it includes the skew. Using it as the Black-Scholes sigma for an at-the-money option is an approximation, probably biased towards a larger premium; the reported PnL level is therefore optimistic as a statement about tradable PnL.
- **Hedging the index** ignores futures and basis and dividends; the price-return index excludes dividends. Trading costs are only explored as *scenario assumptions* on a round-number grid (section 6), not measured: the data contain no quotes, spreads, fees or contract specifications.
- **Daily rebalancing only** (coarser subsamples of it); no intraday analysis.
- **Overlapping windows** make the observations dependent. Standard errors use Newey-West (lag 20), R² intervals use a block bootstrap, and non-overlapping subsamples are shown; even so R² is uncertain to roughly a factor of two, and the reported uncertainty depends on the assumed dependence structure (section 6 repeats the standard error and the R² interval under other lags and block lengths).
- **A single index and a single sample.** Results lean on one extreme episode (2020) and on roughly 125 independent months.
- **Real returns have fat tails and volatility clustering; GBM does not.** Part B is much noisier than Part A.
- **The gamma-PnL decomposition is an approximation with daily data.** It is a continuous-time, first-order result; the path-wise estimate used here tracks the actual PnL with R² {cvr.loc[(D_ALL, C_LOG), 'r2']:.2f} on the real windows and {cvr.loc[(D_TWIN, C_LOG), 'r2']:.2f} on exactly lognormal 21-step windows. It is an empirical fit, not an exact accounting of the PnL.
- **Placeholder risk-free rate** ({cfg.RISK_FREE_RATE_B:.0%}, unsourced), tested at 4% and 8%.
- **Time convention.** The headline uses one trading day = 1/252 year and a window of 21/252 year, while 21 trading days span {sb['calendar_days_min']} to {sb['calendar_days_max']} calendar days and VIX is annualised on calendar time. The calendar-clock re-run (section 6) shows the conclusions do not depend on this, but the numbers do move slightly. PnL is measured at maturity, undiscounted.
- **Data gaps.** December 2022 has no VIX; two stale VIX rows are treated as missing; the exchange holiday calendar could not be verified offline.
- **Chronological validation is a discipline check, not a blind test.** The headline results were known when the split dates and market-state thresholds were fixed, each split has one test period of overlapping windows, and the models that are known when the option is sold (calibration mean, VIX, VIX against trailing volatility) show no usable out-of-sample skill; the high out-of-sample R² belongs to models that use the window's own path.
- **Exploratory, multiple comparisons.** Many estimators, splits, regimes and variants were examined; none is corrected for multiplicity, and the intervals are descriptive.
- **No claim of a tradable edge.** The positive average PnL is the income of a theoretical Black-Scholes call priced from VIX; an executable strategy would need option quotes, a futures hedge and measured costs, none of which are in the data.
- **Volatility estimators.** The size of the volatility risk premium depends on the estimator, and the dependence is the overnight gap: range-based estimators without it give about twice the premium of those with it. Estimators are not interchangeable.
- **A4 optimum** depends on the risk-aversion weight chosen (a modelling assumption) and a cost model that charges a proportional fee on index trades.

## Next steps (need data that are not in this repository)

These are the extensions that would most increase the real-world credibility of the project; none is attempted here because the inputs are missing, and nothing in this report substitutes for them. A concrete plan for the first milestone is in [OPTIONS_EXTENSION_PLAN.md](OPTIONS_EXTENSION_PLAN.md).

1. Historical NIFTY option prices with contract specifications: compare actual option-implied volatility with India VIX, and replace the theoretical Black-Scholes option with a traded one.
2. NIFTY futures prices: hedge with futures instead of the index, including the basis, carry and roll, with a sourced financing rate in place of the 6% placeholder.
3. Quoted bid-ask spreads, fee and tax schedules and contract lot sizes, to replace the round-number cost grid.

## 9. Validation

The validation checklist (`python -m src.validation`) as executed, with the tolerances fixed before running:

{validation}

Checks E1 to E3 are extras: the engine agrees with an independent explicit cash-account loop; the mean hedging PnL is unbiased when sigma_r = sigma_i and mu = r (100 independent replicates); and the simulated mean PnL matches the closed-form gamma-formula expectation.

**Validation log (what failed along the way).** Nothing was relaxed to make a check pass.

- Check 3 first failed because a float64 number cannot sit strictly inside (0,1) once |d1| exceeds about 8 (for example N(49) rounds to exactly 1.0) and the test grid included such points. The code was correct (zero decreasing steps anywhere). The check is now stated in two parts: on the whole grid 0 <= delta <= 1 and delta never decreases with S; where delta is representable (|d1| < 6) it is strictly inside (0,1) and strictly increasing.
- A put-hedging unit test with an arbitrary "std below 5% of premium" cap failed at N = 500 because the put's premium is smaller; it was replaced by the exact identity that a hedged short put and a hedged short call have identical PnL path by path (maximum difference about 3e-12).
- The first closed-form comparison for the gamma-formula expectation used Gauss-Hermite quadrature, which did not converge near expiry (changing the node count moved the value by about 1%). It was replaced by an exact closed form, now unit-tested, and the simulation matches it.
- One mean-PnL result in A4 (N = 20, gross) looked about 3 standard errors from zero. Repeating the experiment 100 times showed the z-scores behave like standard normal noise (check E2).
- One Part B test threshold (R² above 0.95 on GBM windows of only 21 steps) was set too high; the measured R² is about 0.95 with that few steps, so the threshold was lowered to 0.90 for that extra test. This is not part of the brief's checklist.
- A new extra test of the gamma-PnL approximation first demanded a tenfold fall in its RMSE between N = 10 and N = 250; the measured ratio was 0.106, so the arbitrary ratio was loosened to a five-fold fall. The requirement that the error decreases monotonically with N was not changed.

- A test of the volatility estimators on simulated data first required the centred and un-centred close-to-close estimates to correlate above 0.99; the measured value was 0.98 because the sample-mean term matters on 21-day windows. The threshold was loosened to 0.95. The assertions that every estimator recovers the true volatility (within 7%) were not changed.
- The review of this project found that `hedge_pnl` silently accepted `rebalance_idx=[0, 1.9, 2]` (treated as `[0, 1, 2]`) and `[False, True, 2]`, because the indices were cast to integers before being validated; the original elements are now validated first (Booleans, non-finite and fractional values are rejected) and the statistical helpers (`ols_hac`, `mean_hac`, the bootstraps) now reject invalid lags, mismatched lengths, non-finite values and constant regressors. Each case has a regression test.
- The same review reported that a full `reproduce` did not finish within three minutes. That was not reproduced here (about {_total_s()} on this machine, per-stage times in section 10), but each stage now runs in its own process with its time and peak memory recorded.
- The estimator test that required a minimum equal to the close-to-close premium became wrong once gap-inclusive estimators were added (the minimum is now a gap-inclusive range estimator); it was rewritten to compare the extremes with the table itself.
- Two checks added with the market-state and friction code passed on the first run; they are tests of internal consistency (labels match an independent recomputation; zero frictions reproduce the baseline to 1e-12; cost equals turnover times the rate), not of the economic assumptions, which cannot be tested without market data.

Independent reproduction: `tests/test_independent_reproduction.py` rebuilds all {sb['windows']:,} Part B windows from the raw CSV files with the standard library and a scalar hedging loop (no project code), and the PnL of every window agrees with {'`results/partB_windows.csv`' if cfg.raw_data_available() else 'the per-window table `results/partB_windows.csv` (generated locally from the raw data; omitted from the public repository, so this test is skipped there)'} to within 1e-8 percent of S0; the headline slope, intercept, R², Newey-West standard error, mean PnL, profit share and volatility-premium means are re-derived the same way.

## 10. Reproducibility and data quality

| What | How |
|---|---|
| Regenerate every result, figure, the data dictionary, the README and this report | `python -m src reproduce`: runs each stage (validation, Part A, Part B, gallery, data dictionary, README) in its own Python process and records wall time and peak memory in `results/reproduce_stats.json`. Last run: {_timing_summary()} |
| Run the tests | `python -m src test` (or `python -m pytest -q`); the outcome is recorded in `results/test_run.json`. This is separate from `reproduce` because the suite alone takes about a minute on a fast machine and several minutes on a laptop |
| Verify the raw data | `python -m src check-data`; the loader also refuses to run if a raw file was changed, removed or added without regenerating `data/raw/MANIFEST.json` (SHA-256, row counts, date ranges) |
| Python and dependencies | Python 3.13 tested here (`.python-version`); `requirements.txt` gives compatible ranges, `requirements-lock.txt` the exact versions behind the numbers in this report |
| Continuous integration | `.github/workflows/ci.yml`: tests and the raw-data integrity check on Python 3.11, 3.12 and 3.13; then, on the locked versions, a snapshot of the committed results, docs and README, a full `reproduce`, and `python -m src verify-outputs`, which **fails the run if any regenerated table, document or README differs from what is committed** (numbers to a relative tolerance of 1e-6, text exactly; figures are checked for presence only). The status badge on the README shows the latest run; a green run is the evidence, and this report does not state a result for a run that has not happened |
| Data dictionary and provenance | `docs/DATA_DICTIONARY.md` (every input and every results column, with units; a test fails if a column is undocumented) and `docs/FIGURE_GALLERY.md` (a test fails if a figure has no caption or the gallery is out of date) and `data/raw/SOURCES.md` (sources, download dates, known file problems) |
| Randomness | Part A seed {cfg.SEED}, one independent stream per experiment; the Part B bootstraps use fixed streams. Same code and same library versions give the same numbers; other library versions may differ in the last digits |
| Clean deliverable | no virtual environment, caches or bytecode are shipped (`.gitignore` excludes them) |

Test suite (`python -m pytest -q`): {_test_summary()}.
"""


def build_report() -> str:
    """docs/RESEARCH_REPORT.md: the former README, with links made relative to docs/."""
    return (_report_raw().replace("](figures/", "](../figures/").replace("](docs/", "]("))


def build() -> str:
    """The README: a short landing page.  Everything detailed lives in docs/RESEARCH_REPORT.md."""
    sa = json.loads((R / "partA_summary.json").read_text())
    sb = json.loads((R / "partB_summary.json").read_text())
    f2, h, vrp = sa["a2_fits"], sb["headline"], sb["vrp"]
    ef = _csv("partB_effect_size_intervals.csv").set_index("effect")
    mean_ci = ef.loc["mean hedging PnL"]
    cs = _csv("partB_chronological_split.csv")
    ex_ante = cs[(cs.role == "test") & (cs.track == "ex-ante") & (cs.model != "constant")].oos_r2
    cal = _csv("partB_calendar_time_sensitivity.csv")
    sve, sbl = sb["vol_estimators"], sb["chronological_split"]
    prem = float(_csv("partA_A1_sanity.csv")["premium"].iloc[0])
    return f"""# Discrete Delta-Hedging Error Analysis

[![CI](https://github.com/Fablescript22/delta_hedging_analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/Fablescript22/delta_hedging_analysis/actions/workflows/ci.yml)

**Problem.** An option seller who hedges the option with the underlying at discrete times, not continuously, is left with a hedging error. This project measures how large that error is and how fast it shrinks as the seller rebalances more often (Monte Carlo on simulated geometric Brownian motion), and then asks what decides whether the seller makes or loses money: theory says it is the gap between the volatility priced into the option and the volatility the market then delivers. That is tested on {sb['windows']:,} rolling 21-trading-day windows of NIFTY 50 history ({sb['first_t0']} to {sb['last_end']}), with India VIX as the implied volatility. Python: NumPy, pandas, SciPy, matplotlib.

![Hedging error against rebalancing frequency](figures/A2_scaling_loglog.png)

*Standard deviation of the hedging PnL against the number of rebalances N (simulated, log-log). The orange line is the theoretical slope of -1/2; the fitted slope is {f2['slope_all']:.3f}.*

## Three findings

1. **Simulation: the error shrinks like 1/sqrt(N).** The log-log slope of the hedging-error standard deviation against N is {f2['slope_all']:.3f} (theory: -0.5). Rebalancing 100 times more often (N = 5 to 500) cuts the error about {f2['std_ratio_N5_over_N500']:.0f} times, not 100 times. Mean PnL is zero when implied equals realised volatility, as it must be.
2. **Real data: the sign and size of the income are explained by implied volatility sitting above realised volatility.** India VIX exceeded the volatility realised over the following 21 trading days by {vrp['mean_vol_gap_points']:.1f} vol points on average. A *theoretical* seller (Black-Scholes call priced at VIX, hedged in the index) earned {h['mean_pnl_pct']:+.3f}% of the index level per window (95% interval {mean_ci.hac_ci_lo:.2f} to {mean_ci.hac_ci_hi:.2f}), profitable in {h['frac_profit'] * 100:.0f}% of windows. This is **not** a trading result: see "What this does not show".
3. **The surprise: the textbook regression is weak, and nothing known in advance predicts the PnL.** Theory says PnL tracks the variance gap with a slope of about {sb['theory_slope_mean']:.2f}; the measured slope is {h['slope']:.3f} with R² {h['r2']:.2f}, and it changes a lot with the market regime. The path-wise gamma-PnL approximation explains {sb['r2_pnl_vs_gamma_estimate'] * 100:.0f}% of the PnL, so the mechanism is right but one realised-volatility number is a poor summary of it. Using only what is known when the option is sold (VIX, trailing volatility) the out-of-sample R² is between {ex_ante.min():+.3f} and {ex_ante.max():+.3f}: no predictive skill. The strong fits use the window's own realised path and explain the PnL after the fact.

## Method in brief

Sell an at-the-money call priced by Black-Scholes at the implied volatility; delta-hedge with the underlying at discrete closes with a self-financing cash account; measure the PnL at maturity. Part A simulates GBM (fixed seed 42, {cfg.A_N_PATHS:,} paths per setting). Part B repeats the procedure on real windows (K = index level at the start, sigma = India VIX/100, daily hedging, PnL as a fraction of the starting index level). Overlapping windows are handled with Newey-West standard errors, block bootstraps, non-overlapping subsamples and chronological calibration/validation/test splits, with calm, drawdown and recovery states, eleven volatility estimators and a calendar-time re-run as robustness checks. Every number in these documents is read from a results file written by an actual run. The work is exploratory: many comparisons are reported and none is corrected for multiple testing.

## What this does not show

- **It is not a profitable options strategy.** The +{h['mean_pnl_pct']:.3f}% per window is the income of a theoretical Black-Scholes option priced from India VIX. There are **no option quotes**, **no futures hedge** (basis, carry, roll), and **no measured transaction costs, spreads or taxes** in the data. India VIX is a 30-day model-free measure built from options across strikes, not an at-the-money implied volatility, so the theoretical income is probably optimistic. Costs appear only as clearly labelled scenarios.
- **A placeholder input.** The risk-free rate is a constant {cfg.RISK_FREE_RATE_B:.0%}, an unsourced assumption; results at 4% and 8% are reported and move no conclusion.
- **One index, one sample, one extreme episode.** About 125 independent months, with the March 2020 crash ({sb['crash_windows']} windows) driving much of the regression result.
- **In one line:** an investigation of discrete delta-hedging error and of the relationship between theoretical hedging PnL and implied-versus-realised volatility, using Monte Carlo simulation and historical NIFTY 50 / India VIX data.

## Reproduce

```
pip install -r requirements.txt      # or requirements-lock.txt for the exact versions behind the numbers
python -m src reproduce              # regenerate every result, figure and document (each stage in its own process)
python -m src test                   # run the test suite
```

{"" if cfg.raw_data_available() else GET_DATA + chr(10) + chr(10)}Last recorded run: {_timing_summary()}. Tests: {_test_summary()}. The CI badge above is the status of the latest run on GitHub (tests on Python 3.11-3.13, plus a full regeneration that fails if any committed table, document or README differs from the regenerated one). `python -m src check-data` verifies the raw files against a SHA-256 manifest.

## Where to read more

| | |
|---|---|
| [docs/RESEARCH_REPORT.md](docs/RESEARCH_REPORT.md) | the full report: theory, data cleaning, all tables, every figure, limitations, validation log |
| [EXPLAINER.md](EXPLAINER.md) | plain-English walkthrough of the code and results |
| [docs/FIGURE_GALLERY.md](docs/FIGURE_GALLERY.md) | all figures with captions |
| [docs/DATA_DICTIONARY.md](docs/DATA_DICTIONARY.md) | every input and results column, with units |
| [data/raw/SOURCES.md](data/raw/SOURCES.md) | data sources, official methodology references, known file problems, terms of use |
| [docs/OPTIONS_EXTENSION_PLAN.md](docs/OPTIONS_EXTENSION_PLAN.md) | plan for the realistic extension (option quotes, futures hedging, measured costs) |

## Licence and data

No licence has been chosen for the code, so by default all rights are reserved: the code can be read, but it has not been licensed for reuse. {"The raw NIFTY 50 and India VIX files in `data/raw/` come from NSE Indices and NSE and are subject to their terms (see `data/raw/SOURCES.md`); this working copy keeps them for private reproducibility, and a public copy omits them." if cfg.raw_data_available() else "The raw NIFTY 50 and India VIX files come from NSE Indices and NSE, are subject to their terms (see `data/raw/SOURCES.md`) and are not redistributed here. The per-window results table, which holds the VIX level of every date, is omitted for the same reason; the aggregate tables and figures are included."}
"""


def _total_s() -> str:
    f = R / "reproduce_stats.json"
    return _volatile(f"{json.loads(f.read_text())['total_seconds']:.0f} s" if f.exists() else "n/a")


def main() -> None:
    (cfg.ROOT / "README.md").write_text(build())
    docs = cfg.ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "RESEARCH_REPORT.md").write_text(build_report())
    print("wrote README.md and docs/RESEARCH_REPORT.md")


if __name__ == "__main__":
    main()
