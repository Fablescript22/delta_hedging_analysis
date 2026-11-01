"""Figures.  Colours: first slots of a validated categorical palette (blue, orange, aqua), light surface.

Style rules followed: one axis per chart, thin marks, 2px-ish lines, >= 8px markers with a surface-coloured
ring, hairline solid gridlines, legend whenever there are two or more series, text in ink colours
(never in the series colour).
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

from . import config as cfg  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e0"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
LW = 1.6        # ~2 px at 100 dpi
MS = 6.5        # ~9 px marker


def _style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
        "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "legend.frameon": False, "figure.dpi": 100, "savefig.dpi": 160,
    })


def _save(fig, name: str):
    cfg.FIG_DIR.mkdir(exist_ok=True)
    path = cfg.FIG_DIR / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plain(x, _pos=None):
    return f"{x:g}"


# ------------------------------------------------------------------------------------------------ A2
def fig_a2_scaling(df, fits):
    _style()
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    N, sd, se = df["N"].to_numpy(float), df["std_pnl"].to_numpy(), df["se_std"].to_numpy()
    grid = np.geomspace(N.min() * 0.85, N.max() * 1.15, 50)
    # reference: exact -1/2 slope anchored at the N=100 point
    i = int(np.where(N == 100)[0][0])
    ref = sd[i] * np.sqrt(N[i] / grid)
    ax.plot(grid, ref, color=ORANGE, lw=LW, label="reference: slope -0.5 (anchored at N=100)")
    fit = np.exp(fits["intercept_all"]) * grid ** fits["slope_all"]
    ax.plot(grid, fit, color=BLUE, lw=LW, alpha=0.55, label=f"fitted line, slope {fits['slope_all']:.3f}")
    ax.errorbar(N, sd, yerr=se, fmt="none", ecolor=BLUE, elinewidth=1.0, capsize=0)
    ax.plot(N, sd, "o", color=BLUE, ms=MS, mec=SURFACE, mew=1.5, label="simulated std of hedging PnL")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xticks(list(N)); ax.xaxis.set_major_formatter(FuncFormatter(_plain)); ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_yticks([0.3, 0.5, 1, 2, 3]); ax.yaxis.set_major_formatter(FuncFormatter(_plain)); ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("rebalances N (log scale)"); ax.set_ylabel("std of hedging PnL, money units (log scale)")
    ax.set_title("Hedging error falls like 1/sqrt(N)")
    ax.legend(loc="upper right")
    return _save(fig, "A2_scaling_loglog.png")


def fig_a2_distributions(samples, premium):
    _style()
    Ns = sorted(samples)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), sharex=True)
    lim = 8.0
    bins = np.linspace(-lim, lim, 81)
    for ax, N in zip(axes.ravel(), Ns):
        full = samples[N]
        x = full[np.abs(full) <= lim]                     # tails beyond +/-lim are dropped, and counted below
        ax.hist(x, bins=bins, density=True, color=BLUE, alpha=0.9, edgecolor=SURFACE, linewidth=0.4)
        ax.axvline(0, color=INK2, lw=0.8)
        ax.set_title(f"N = {N}   (std {full.std(ddof=1):.2f})", fontsize=10)
        n_out = int((np.abs(full) > lim).sum())
        if n_out:
            ax.text(0.01, 0.93, f"{n_out} paths beyond +/-{lim:g}\nnot shown (worst {full.min():.1f})",
                    transform=ax.transAxes, ha="left", va="top", fontsize=8.5, color=INK2)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
    for ax in axes[1]:
        ax.set_xlabel("hedging PnL (money units)")
    fig.suptitle(f"Distribution of hedging PnL, 20,000 paths each (premium {premium:.2f})", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    return _save(fig, "A2_pnl_distributions.png")


# ------------------------------------------------------------------------------------------------ A3
def fig_a3_volgap(df, summary, scatter):
    _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    gap = df["gap"].to_numpy()
    xs = np.linspace(gap.min() * 1.05, gap.max() * 1.05, 2)
    a.plot(df["gap"], df["theory_mean"], "-", color=ORANGE, lw=LW, label="gamma formula (continuous-time theory)")
    a.errorbar(gap, df["mean_pnl"], yerr=df["se_mean"] * 1.96, fmt="none", ecolor=BLUE, elinewidth=1.0)
    a.plot(gap, df["mean_pnl"], "o", color=BLUE, ms=MS, mec=SURFACE, mew=1.5, label="simulated mean (N=250, 20,000 paths)")
    a.axhline(0, color=INK2, lw=0.8); a.axvline(0, color=INK2, lw=0.8)
    a.set_xlabel(r"variance gap  $\sigma_i^2-\sigma_r^2$"); a.set_ylabel("mean hedging PnL (money units)")
    a.set_title("Seller wins when realised vol < implied")
    a.legend(loc="upper left")

    lim = np.percentile(np.abs(np.concatenate([scatter["pnl"], scatter["est_path"]])), 99.5)
    b.scatter(scatter["est_path"], scatter["pnl"], s=5, color=BLUE, alpha=0.25, linewidths=0, rasterized=True)
    b.plot([-lim, lim], [-lim, lim], color=ORANGE, lw=LW, label="45-degree line (perfect match)")
    b.set_xlim(-lim, lim); b.set_ylim(-lim, lim)
    b.set_xlabel("gamma-formula estimate along each path"); b.set_ylabel("actual hedging PnL of that path")
    b.set_title(f"Path by path, R² = {summary['r2_pnl_vs_gamma_estimate_pathwise']:.3f}")
    b.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, "A3_volgap_mean_pnl.png")


# ------------------------------------------------------------------------------------------------ A4
def fig_a4_costs(df, opt, lam: float = 1.0):
    _style()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.0, 4.4))
    cols = {0.0005: BLUE, 0.001: ORANGE}
    col_obj = f"objective_lambda_{lam}"
    for c, g in df.groupby("cost_rate"):
        lab = f"cost {c*1e4:.0f} bp per unit traded"
        a.plot(g["N"], g["mean_cost"], "-o", color=cols[c], lw=LW, ms=MS, mec=SURFACE, mew=1.5, label=lab)
        b.plot(g["N"], g[col_obj], "-o", color=cols[c], lw=LW, ms=MS, mec=SURFACE, mew=1.5, label=lab)
        best = g.loc[g[col_obj].idxmax()]
        b.plot([best["N"]], [best[col_obj]], "o", ms=12, mfc="none", mec=INK, mew=1.4)
        b.annotate(f"best N = {int(best['N'])}", (best["N"], best[col_obj]), textcoords="offset points",
                   xytext=(0, 16) if c == 0.0005 else (-10, -24), ha="center" if c == 0.0005 else "right",
                   fontsize=9, color=INK)
    for ax in (a, b):
        ax.set_xscale("log"); ax.set_xticks(sorted(df["N"].unique())); ax.xaxis.set_major_formatter(FuncFormatter(_plain))
        ax.xaxis.set_minor_locator(plt.NullLocator()); ax.set_xlabel("rebalances N (log scale)")
    a.set_ylabel("mean total transaction cost (money units)"); a.set_title("Costs rise with rebalancing frequency")
    b.set_ylabel(rf"mean(net PnL) $-\ {lam:g}\,\times$ std(net PnL)"); b.set_title("Risk-adjusted net PnL peaks at a middle N")
    a.legend(loc="upper left"); b.legend(loc="lower right")
    fig.tight_layout()
    return _save(fig, "A4_costs_and_optimal_N.png")


# ================================================================================================ Part B
def fig_b_vix_vs_rv(out):
    _style()
    ws, bt, crash = out["ws"], out["bt"], out["crash"]
    vix, rv = ws.vix_pct, bt.rv * 100
    vrp = out["summary"]["vrp"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.5), gridspec_kw={"width_ratios": [1, 1.25]})
    a.scatter(vix[~crash], rv[~crash], s=7, color=BLUE, alpha=0.35, linewidths=0, label="21-day windows", rasterized=True)
    a.scatter(vix[crash], rv[crash], s=11, color=ORANGE, alpha=0.8, linewidths=0, label="windows overlapping Feb 20 - Apr 30 2020",
              rasterized=True)
    lim = 90
    a.plot([0, lim], [0, lim], color=INK2, lw=1.2, label="realised = VIX")
    a.set_xlim(5, lim); a.set_ylim(0, lim)
    a.set_xlabel("India VIX at window start (% p.a.)"); a.set_ylabel("NIFTY realised vol over the next 21 trading days (% p.a.)")
    a.set_title(f"Realised vol sits below VIX in {vrp['frac_windows_rv_below_vix']:.0%} of windows")
    a.legend(loc="lower right", fontsize=8.5)

    b.plot(ws.t0, vix, color=BLUE, lw=1.1, label="India VIX at window start")
    b.plot(ws.t0, rv, color=ORANGE, lw=1.1, label="NIFTY realised vol over the following 21 days")
    b.set_ylabel("% p.a."); b.set_title(f"Average gap {vrp['mean_vol_gap_points']:.1f} vol points (volatility risk premium)")
    b.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, "B1_vix_vs_realised_vol.png")


def fig_b_regression(out):
    _style()
    ws, bt, crash, s = out["ws"], out["bt"], out["crash"], out["summary"]
    x, y = bt.gap, bt.pnl * 100
    fig, (a, b) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    a.scatter(x[~crash], y[~crash], s=7, color=BLUE, alpha=0.35, linewidths=0, label="21-day windows", rasterized=True)
    a.scatter(x[crash], y[crash], s=12, color=ORANGE, alpha=0.85, linewidths=0, label="windows overlapping Feb 20 - Apr 30 2020",
              rasterized=True)
    h = s["headline"]
    xs = np.linspace(x.min(), x.max(), 2)
    a.plot(xs, 100 * (h["intercept"] + h["slope"] * xs), color=INK, lw=LW, label=f"all windows: slope {h['slope']:.3f}, R² {h['r2']:.2f}")
    # fit excluding the crash windows
    A = np.polyfit(x[~crash], y[~crash], 1)
    xs2 = np.linspace(x[~crash].min(), x[~crash].max(), 2)
    a.plot(xs2, np.polyval(A, xs2), color=AQUA, lw=LW,
           label=f"excluding 2020 crash windows: slope {A[0]/100:.3f}, R² {np.corrcoef(x[~crash], y[~crash])[0,1]**2:.2f}")
    a.axhline(0, color=INK2, lw=0.8); a.axvline(0, color=INK2, lw=0.8)
    a.set_xlabel(r"variance gap  $\sigma_i^2-\sigma_r^2$  (VIX$^2$ minus realised variance)")
    a.set_ylabel("hedging PnL (% of initial index level)")
    a.set_title("Real windows: PnL vs variance gap")
    a.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8.5, ncol=1)

    est = bt.est * 100
    b.scatter(est[~crash], y[~crash], s=7, color=BLUE, alpha=0.35, linewidths=0, rasterized=True, label="21-day windows")
    b.scatter(est[crash], y[crash], s=12, color=ORANGE, alpha=0.85, linewidths=0, rasterized=True, label="2020 crash windows")
    lo, hi = min(est.min(), y.min()), max(est.max(), y.max())
    b.plot([lo, hi], [lo, hi], color=INK2, lw=1.2, label="45-degree line")
    b.set_xlabel("gamma-formula estimate along the actual path (% of S0)"); b.set_ylabel("actual hedging PnL (% of S0)")
    b.set_title(f"The formula works path by path: R² {s['r2_pnl_vs_gamma_estimate']:.2f}")
    b.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "B2_pnl_vs_variance_gap.png")


def fig_b_frequency(out):
    _style()
    f = out["freq"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.5, 4.3))
    k = f["interval_days"].to_numpy()
    a.fill_between(k, f["p05_pct"], f["p95_pct"], color=BLUE, alpha=0.12, linewidth=0, label="5th to 95th percentile of PnL")
    a.plot(k, f["mean_pnl_pct"], "-o", color=ORANGE, lw=LW, ms=MS, mec=SURFACE, mew=1.5, label="mean PnL")
    ax2_vals = f["std_pnl_pct"]
    a.plot(k, ax2_vals, "-o", color=BLUE, lw=LW, ms=MS, mec=SURFACE, mew=1.5, label="std of PnL")
    a.axhline(0, color=INK2, lw=0.8)
    a.set_xticks(k); a.set_xlabel("rebalance every k trading days"); a.set_ylabel("hedging PnL (% of S0)")
    a.set_title("Coarser hedging, wider outcomes"); a.legend(loc="upper left", fontsize=8.5)

    b.plot(k, f["std_diff_vs_daily_pct"], "-o", color=BLUE, lw=LW, ms=MS, mec=SURFACE, mew=1.5, label="measured")
    b.plot(k, f["std_diff_predicted_by_sqrtN_law_pct"], "-o", color=ORANGE, lw=LW, ms=MS, mec=SURFACE, mew=1.5,
           label="1/sqrt(N) law (one fitted constant)")
    b.set_xticks(k); b.set_xlabel("rebalance every k trading days")
    b.set_ylabel("std of (PnL at k days  -  PnL hedged daily), % of S0")
    b.set_title("Discretisation error vs the daily hedge"); b.legend(loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return _save(fig, "B3_error_vs_rebalancing_interval.png")


# ------------------------------------------------------------------------------------------------ extended Part B
YELLOW = "#eda100"


def _errbar(ax, y, est, lo, hi, color, label=None, dy=0.0, cap=0.12):
    ax.hlines(y + dy, lo, hi, color=color, lw=LW)
    ax.plot([est], [y + dy], "o", color=color, ms=MS, mec=SURFACE, mew=1.5, label=label, zorder=3)


def fig_b_effect_sizes(out):
    """Six small panels, one effect each: estimate with its HAC interval (blue) and block-bootstrap interval (orange)."""
    _style()
    df = out["effects"]
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 5.4))
    for ax, (_, r) in zip(axes.ravel(), df.iterrows()):
        if not np.isnan(r.hac_se):
            _errbar(ax, 1, r.estimate, r.hac_ci_lo, r.hac_ci_hi, BLUE, "HAC (Newey-West) 95% interval", dy=0.12)
            _errbar(ax, 0, r.estimate, r.boot_ci_lo, r.boot_ci_hi, ORANGE, "block-bootstrap 95% interval", dy=-0.12)
        else:
            _errbar(ax, 0, r.estimate, r.boot_ci_lo, r.boot_ci_hi, ORANGE, "block-bootstrap 95% interval", dy=-0.12)
        ax.set_ylim(-0.7, 1.7)
        ax.set_yticks([])
        ax.grid(axis="y", visible=False)
        ax.set_title(r.effect.replace(" (", "\n("), fontsize=9.5)
        ax.set_xlabel(r.unit.replace(" per ", "\nper "), fontsize=8)
        ax.annotate(f"{r.estimate:.3g}", (r.estimate, 0.5), ha="center", va="center", fontsize=8.5, color=INK2)
    h, l = axes[0, 0].get_legend_handles_labels()
    seen = dict(zip(l, h))
    fig.legend(seen.values(), seen.keys(), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(fig, "B4_effect_size_intervals.png")


def fig_b_market_states(out):
    _style()
    df = out["market_states"].dropna(subset=["mean_pnl_pct"])
    fig, (a, b) = plt.subplots(1, 2, figsize=(9.6, 3.9))
    x = np.arange(len(df))
    a.bar(x, df.mean_pnl_pct, width=0.55, color=BLUE)
    a.errorbar(x, df.mean_pnl_pct, yerr=[df.mean_pnl_pct - df.mean_pnl_ci_lo, df.mean_pnl_ci_hi - df.mean_pnl_pct],
               fmt="none", ecolor=INK2, elinewidth=LW, capsize=4)
    a.set_xticks(x, [f"{s}\n(n={int(n)})" for s, n in zip(df.regime, df.windows)])
    a.set_ylabel("mean hedging PnL, % of S0 per window")
    a.set_title("Seller's PnL by market state\n(95% HAC interval)", fontsize=10)
    a.axhline(0, color=INK2, lw=0.8)
    b.bar(x, df.slope, width=0.55, color=ORANGE)
    b.errorbar(x, df.slope, yerr=1.96 * df.slope_se_hac, fmt="none", ecolor=INK2, elinewidth=LW, capsize=4)
    b.axhline(out["summary"]["theory_slope_mean"], color=INK2, lw=1.2, ls="--")
    b.annotate("theory average slope", (-0.4, out["summary"]["theory_slope_mean"]), ha="left", va="bottom",
               fontsize=8.5, color=INK2)
    b.set_xticks(x, [f"{s}\nR² {r2:.2f}" for s, r2 in zip(df.regime, df.r2)])
    b.set_ylabel("slope of PnL on variance gap")
    b.set_title("PnL-vs-gap slope by market state\n(95% HAC interval)", fontsize=10)
    fig.tight_layout()
    return _save(fig, "B5_market_states.png")


def fig_b_estimators(out):
    """Premium and fit by estimator.  Colour = what the estimator measures, so the overnight-gap effect is
    visible in the figure itself: intraday-only range estimators (orange) sit far from everything that sees the
    close-to-close move (blue) or adds the overnight gap back (aqua)."""
    _style()
    df = out["est_table"].iloc[::-1].reset_index(drop=True)
    colour = {"close-to-close": BLUE, "range, intraday only (no overnight gap)": ORANGE, "range + overnight gap": AQUA}
    cols = [colour[t] for t in df["type"]]
    y = np.arange(len(df))
    fig, (a, b) = plt.subplots(1, 2, figsize=(10.4, 5.0), sharey=True)
    for yi, lo, hi, c in zip(y, df.mean_vol_gap_points - 1.96 * df.se_vol_gap_hac,
                             df.mean_vol_gap_points + 1.96 * df.se_vol_gap_hac, cols):
        a.hlines(yi, lo, hi, color=c, lw=LW)
    a.scatter(df.mean_vol_gap_points, y, c=cols, s=MS ** 2 * 1.2, zorder=3, edgecolors=SURFACE, linewidths=1.5)
    a.axvline(0, color=INK2, lw=0.8)
    a.set_yticks(y, df.estimator)
    a.set_xlabel("mean VIX minus realised vol, vol points (95% HAC interval)")
    a.set_title("Volatility risk premium by estimator", fontsize=10)
    b.barh(y, df.r2, height=0.55, color=cols)
    for yi, v in zip(y, df.r2):
        b.annotate(f"{v:.3f}", (v, yi), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8.5, color=INK2)
    b.set_xlim(0, df.r2.max() * 1.2)
    b.set_xlabel("R² of PnL on variance gap")
    b.set_title("Fit of the simple regression by estimator", fontsize=10)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=c, label=t) for t, c in colour.items()], loc="lower center", ncol=3, fontsize=8.5,
               title="what the estimator measures", title_fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return _save(fig, "B6_vol_estimators.png")


def fig_b_chronological(out):
    """Two tracks, visibly separated: models whose inputs are known when the option is sold (ex-ante, left) and
    models that use the window's own realised path (ex-post, right).  Only the left block speaks to prediction."""
    _style()
    df = out["split_table"]
    models = ["constant", "vix_level", "exante_var_gap", "exante_vol_gap",
              "gap_close_to_close", "gap_bipower", "gap_parkinson", "gamma_estimate"]
    n_ex = 4
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.6), sharey=True)
    for ax, sname in zip(axes, ("primary", "crash_held_out")):
        sub = df[df.split == sname]
        x = np.arange(len(models))
        ax.axvspan(-0.5, n_ex - 0.5, color=BLUE, alpha=0.07, lw=0)
        ax.axvspan(n_ex - 0.5, len(models) - 0.5, color=ORANGE, alpha=0.07, lw=0)
        for off, role, col in ((-0.19, "validation", BLUE), (0.19, "test", ORANGE)):
            v = [sub[(sub.model == m) & (sub.role == role)].rmse_pct.iloc[0] for m in models]
            ax.bar(x + off, v, width=0.36, color=col, label=role)
        ax.set_yscale("log")
        ax.set_xticks(x, [m.replace("gap_", "gap:\n").replace("exante_", "ex-ante\n").replace("_", "\n") for m in models], fontsize=7.5)
        ax.set_title(f"split '{sname.replace('_', ' ')}'", fontsize=10, pad=18)
        ax.text((n_ex - 1) / 2, 1.01, "EX-ANTE: known when selling", transform=ax.get_xaxis_transform(), ha="center",
                va="bottom", fontsize=8, color=INK2)
        ax.text(n_ex + (len(models) - n_ex - 1) / 2, 1.01, "EX-POST: uses the window's path", transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", fontsize=8, color=INK2)
    axes[0].set_ylabel("RMSE of predicted PnL, % of S0 (log scale)")
    axes[0].legend(title="period (fitted on calibration only)", loc="upper right", fontsize=8, title_fontsize=8)
    fig.tight_layout()
    return _save(fig, "B7_chronological_split.png")


def fig_b_frictions(out):
    _style()
    g = out["friction_tbl"]
    g = g[(g.option_half_spread_vol_pts == 0) & g.delta_step.isna()]
    cols = {1: BLUE, 2: ORANGE, 5: AQUA, 10: YELLOW}
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for k, c in cols.items():
        s = g[g.rebalance_every == k].sort_values("one_way_cost_bps")
        ax.plot(s.one_way_cost_bps, s.mean_pnl_pct, "-o", color=c, lw=LW, ms=MS, mec=SURFACE, mew=1.5, label=f"rebalance every {k} day{'s' if k > 1 else ''}")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xlabel("one-way hedging cost, basis points of traded notional (scenario assumption)")
    ax.set_ylabel("mean seller PnL, % of S0 per window")
    ax.set_title("How trading costs erode the volatility-premium income", fontsize=10.5)
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, "B8_friction_scenarios.png")
