"""Independent reproduction of the key generated outputs.

These tests deliberately do NOT import the project's data loader, hedging engine or regression code.  They rebuild
the numbers from the raw CSV files with the standard library (``csv``, ``math``) plus plain NumPy linear algebra,
using a scalar, loop-based implementation, and compare against the files in ``results/``.  If the optimised,
vectorised pipeline and this slow, obvious one ever disagree, one of them is wrong.

They skip (not fail) when ``results/`` has not been generated yet.
"""
import csv
import glob
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RAW, RES = ROOT / "data" / "raw", ROOT / "results"
W, DAYS = 21, 252.0                     # Part B constants (21-day windows, 252 days/yr); the rate is per window, see _read_tbill

_RAW_PRESENT = bool(list(RAW.glob("*.csv")))
needs_results = pytest.mark.skipif(
    not (RES / "partB_windows.csv").exists(),
    reason=("run `python -m src.run_all` first" if _RAW_PRESENT
            else "raw NIFTY 50 / India VIX files and results/partB_windows.csv are not in the public repository (see data/raw/SOURCES.md)"))


def _n(x):                                  # standard normal CDF from math.erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _read_nifty():
    out = {}
    for f in glob.glob(str(RAW / "NIFTY 50_Historical_PR_*.csv")):
        with open(f, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                out[datetime.strptime(row["Date"].strip(), "%d %b %Y").date()] = float(row["Close"])
    return dict(sorted(out.items()))


def _read_vix():
    out = {}
    for f in glob.glob(str(RAW / "hist_india_vix_*.csv")):
        with open(f, newline="", encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                row = {k.strip(): v.strip() for k, v in row.items()}
                if float(row["Open"]) == 0 and float(row["High"]) == 0 and float(row["Low"]) == 0:
                    continue                                  # stale placeholder rows are treated as missing
                out[datetime.strptime(row["Date"], "%d-%b-%Y").date()] = float(row["Close"])
    return out


def _read_tbill():
    """(sorted auction dates, implicit yields in percent) from the flattened RBI file, standard library only."""
    with open(RAW / "RBI_91day_tbill_auctions.csv", newline="") as fh:
        rows = sorted((datetime.strptime(r["Date"], "%Y-%m-%d").date(), float(r["Cutoff_Yield_Percent"]))
                      for r in csv.DictReader(fh))
    return [d for d, _ in rows], [y for _, y in rows]


def _rate_asof(t0, dates, yields):
    """Continuously compounded rate from the latest auction on or before t0 (written out, no project code)."""
    import bisect
    i = bisect.bisect_right(dates, t0) - 1
    assert i >= 0
    y = yields[i] / 100.0
    return (365.0 / 91.0) * math.log(1.0 + y * 91.0 / 365.0), yields[i], (t0 - dates[i]).days


def _short_call_pnl(path, sigma, R):
    """Scalar daily delta hedge of one short ATM call at continuously compounded rate R; PnL at maturity in money."""
    K, T = path[0], W / DAYS

    def d1(s, tau):
        return (math.log(s / K) + (R + 0.5 * sigma**2) * tau) / (sigma * math.sqrt(tau))

    d = d1(K, T)
    cash = K * _n(d) - K * math.exp(-R * T) * _n(d - sigma * math.sqrt(T))      # premium received
    held = 0.0
    for i in range(W):
        new = _n(d1(path[i], (W - i) / DAYS))
        cash -= (new - held) * path[i]
        held = new
        cash *= math.exp(R / DAYS)
    return cash + held * path[W] - max(path[W] - K, 0.0)


@pytest.fixture(scope="module")
def rebuilt():
    nifty, vix = _read_nifty(), _read_vix()
    ad, ay = _read_tbill()
    dates = list(nifty)
    closes = [nifty[d] for d in dates]
    rows = []
    for i in range(len(dates) - W):
        t0, tend = dates[i], dates[i + W]
        if t0 not in vix or (tend - t0).days > 45:
            continue
        path = closes[i:i + W + 1]
        rate, yld, stale = _rate_asof(t0, ad, ay)
        pnl = _short_call_pnl(path, vix[t0] / 100.0, rate) / path[0] * 100.0
        rv = math.sqrt(DAYS / W * sum(math.log(path[j + 1] / path[j]) ** 2 for j in range(W)))
        rows.append((t0.isoformat(), tend.isoformat(), vix[t0], rv, pnl, rate, yld, stale))
    return rows


@pytest.fixture(scope="module")
def saved():
    with open(RES / "partB_windows.csv", newline="") as fh:
        return list(csv.DictReader(fh))


@needs_results
def test_every_saved_window_matches_independent_loop(rebuilt, saved):
    assert len(rebuilt) == len(saved) == 2629
    assert [r[0] for r in rebuilt] == [s["t0"] for s in saved]
    assert [r[1] for r in rebuilt] == [s["t_end"] for s in saved]
    pnl_saved = np.array([float(s["pnl_pct"]) for s in saved])
    pnl_ind = np.array([r[4] for r in rebuilt])
    rv_saved = np.array([float(s["realised_vol"]) for s in saved])
    rv_ind = np.array([r[3] for r in rebuilt])
    assert np.max(np.abs(pnl_saved - pnl_ind)) < 1e-8          # percent of S0
    assert np.max(np.abs(rv_saved - rv_ind)) < 1e-12
    for col, k in (("risk_free_rate", 5), ("tbill_yield_pct", 6), ("rate_stale_days", 7)):    # the rate itself, rebuilt independently
        assert np.max(np.abs(np.array([float(s[col]) for s in saved]) - np.array([r[k] for r in rebuilt]))) < 1e-9


@needs_results
def test_headline_statistics_match_independent_regression(rebuilt):
    summ = json.load(open(RES / "partB_summary.json"))["headline"]
    vix = np.array([r[2] for r in rebuilt]) / 100.0
    rv = np.array([r[3] for r in rebuilt])
    y = np.array([r[4] for r in rebuilt]) / 100.0
    x = vix**2 - rv**2
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    r2 = 1 - resid @ resid / ((y - y.mean()) @ (y - y.mean()))
    assert beta[1] == pytest.approx(summ["slope"], rel=1e-8)
    assert beta[0] == pytest.approx(summ["intercept"], rel=1e-8)
    assert r2 == pytest.approx(summ["r2"], rel=1e-8)
    assert 100 * y.mean() == pytest.approx(summ["mean_pnl_pct"], rel=1e-8)
    assert float((y > 0).mean()) == pytest.approx(summ["frac_profit"], abs=1e-12)
    # Newey-West (Bartlett, lag 20) slope SE rebuilt from first principles, with the same n/(n-2) scaling
    n, lag = len(y), summ["lag"]
    xtx_inv = np.linalg.inv(X.T @ X)
    g0 = (X * resid[:, None]).T @ (X * resid[:, None])
    s = g0.copy()
    for k in range(1, lag + 1):
        gk = (X[k:] * resid[k:, None]).T @ (X[:-k] * resid[:-k, None])
        s += (1 - k / (lag + 1)) * (gk + gk.T)
    se = math.sqrt((xtx_inv @ s @ xtx_inv * n / (n - 2))[1, 1])
    assert se == pytest.approx(summ["se_slope_hac"], rel=1e-8)


@needs_results
def test_vrp_numbers_match_independent_means(rebuilt):
    vrp = json.load(open(RES / "partB_summary.json"))["vrp"]
    vix = np.array([r[2] for r in rebuilt])
    rv = np.array([r[3] for r in rebuilt]) * 100
    assert vix.mean() == pytest.approx(vrp["mean_vix"], rel=1e-10)
    assert rv.mean() == pytest.approx(vrp["mean_rv"], rel=1e-10)
    assert (rv < vix).mean() == pytest.approx(vrp["frac_windows_rv_below_vix"], abs=1e-12)


@needs_results
def test_part_a_premium_and_slope_match_closed_form():
    S0 = K = 100.0
    sig, r, T = 0.20, 0.05, 1.0
    d1 = (math.log(S0 / K) + (r + sig**2 / 2) * T) / (sig * math.sqrt(T))
    premium = S0 * _n(d1) - K * math.exp(-r * T) * _n(d1 - sig * math.sqrt(T))
    assert premium == pytest.approx(10.4505835722, abs=1e-9)             # textbook Black-Scholes value
    with open(RES / "partA_A1_sanity.csv", newline="") as fh:
        assert all(float(row["premium"]) == pytest.approx(premium, abs=1e-9) for row in csv.DictReader(fh))
    with open(RES / "partA_A2_frequency.csv", newline="") as fh:
        rows = list(csv.DictReader(fh))
    lx = np.log([float(r["N"]) for r in rows])
    ly = np.log([float(r["std_pnl"]) for r in rows])
    slope = np.polyfit(lx, ly, 1)[0]
    assert slope == pytest.approx(json.load(open(RES / "partA_summary.json"))["a2_fits"]["slope_all"], rel=1e-9)
    assert -0.55 < slope < -0.45                                             # theory: -1/2


@needs_results
def test_part_a_std_matches_independent_scalar_hedger():
    """Re-simulate a small GBM sample with plain numpy + a scalar hedging loop: std(PnL) must scale like 1/sqrt(N)."""
    rng = np.random.default_rng(2026)
    sig, r, T, S0 = 0.20, 0.05, 1.0, 100.0
    stds = {}
    for N in (10, 40):
        pnls = []
        for _ in range(600):
            dt = T / N
            z = rng.standard_normal(N)
            path = [S0]
            for zi in z:
                path.append(path[-1] * math.exp((r - 0.5 * sig**2) * dt + sig * math.sqrt(dt) * zi))
            d1 = lambda s, tau: (math.log(s / S0) + (r + 0.5 * sig**2) * tau) / (sig * math.sqrt(tau))
            cash = S0 * _n(d1(S0, T)) - S0 * math.exp(-r * T) * _n(d1(S0, T) - sig * math.sqrt(T))
            held = 0.0
            for i in range(N):
                new = _n(d1(path[i], T - i * dt))
                cash -= (new - held) * path[i]
                held = new
                cash *= math.exp(r * dt)
            pnls.append(cash + held * path[-1] - max(path[-1] - S0, 0.0))
        stds[N] = np.std(pnls, ddof=1)
    assert stds[10] / stds[40] == pytest.approx(2.0, rel=0.2)             # sqrt(40/10) = 2
