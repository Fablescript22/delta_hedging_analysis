"""Part B data layer: load, clean, align and window the NIFTY 50 and India VIX files.

Raw files in ``data/raw/`` are only ever READ.  Every cleaning decision is made
here, in code, and is reported (see ``load_all(...).report``) so nothing is hidden.

Quirks of the supplied files (found by ``src/inspect_data.py``):
  * NIFTY files: quoted CSV, Windows line endings, dates like "30 Dec 2016", newest row first.
  * VIX files: UTF-8 BOM, column names with trailing spaces ("Close "), dates like "01-JAN-2016",
    oldest row first.
  * VIX has two rows (2021-02-12, 2021-03-30) with Open = High = Low = Prev.Close = 0 and a Close
    that is just the previous close repeated -> stale placeholders, treated as MISSING (not filled).
  * The VIX file for 2022 stops on 2022-12-01, so Dec 2022 has NIFTY but no VIX.
  * A few weekend-dated rows are genuine special sessions (Diwali Muhurat, Budget-day Saturday ...).
"""
from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

NIFTY_GLOB = "NIFTY 50_Historical_PR_*.csv"
VIX_GLOB = "hist_india_vix_*.csv"


# ----------------------------------------------------------------------------- loading
def _read_files(raw_dir: Path, pattern: str, date_fmt: str) -> pd.DataFrame:
    files = sorted(glob.glob(str(Path(raw_dir) / pattern)))
    if not files:
        raise FileNotFoundError(f"no files matching {pattern!r} in {raw_dir}")
    frames = []
    for f in files:
        df = pd.read_csv(f, encoding="utf-8-sig", dtype=str)   # utf-8-sig drops the BOM
        df.columns = [c.strip() for c in df.columns]
        df["Date"] = pd.to_datetime(df["Date"].str.strip(), format=date_fmt, errors="raise")
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    for c in out.columns:
        if c not in ("Index Name", "Date"):
            out[c] = pd.to_numeric(out[c].str.strip(), errors="coerce")
    dup = out["Date"].duplicated()
    if dup.any():
        raise ValueError(f"duplicate dates in {pattern}: {out.loc[dup, 'Date'].dt.date.tolist()[:5]}")
    return out.sort_values("Date").set_index("Date")


def load_nifty(raw_dir: Path = cfg.RAW_DIR) -> pd.DataFrame:
    """NIFTY 50 price-return index, ascending by date, columns Open/High/Low/Close."""
    df = _read_files(raw_dir, NIFTY_GLOB, "%d %b %Y")
    names = set(df["Index Name"].unique())
    if names != {"NIFTY 50"}:
        raise ValueError(f"unexpected index names in NIFTY files: {names}")
    return df[["Open", "High", "Low", "Close"]]


def load_vix(raw_dir: Path = cfg.RAW_DIR) -> pd.DataFrame:
    """India VIX in PERCENT points (e.g. 14.06), ascending by date.

    Adds a boolean ``stale`` column: True for placeholder rows whose OHLC is all zero.  Their
    ``Close`` is set to NaN rather than trusted or forward-filled.
    """
    df = _read_files(raw_dir, VIX_GLOB, "%d-%b-%Y")
    df = df[["Open", "High", "Low", "Close", "Prev. Close"]].copy()
    df["stale"] = (df["Open"] == 0) & (df["High"] == 0) & (df["Low"] == 0)
    df.loc[df["stale"], "Close"] = np.nan
    return df


# ----------------------------------------------------------------------------- alignment
@dataclass
class RealData:
    nifty: pd.DataFrame            # full NIFTY series (used for price paths)
    vix: pd.DataFrame              # VIX with stale rows NaN'd
    aligned: pd.DataFrame          # inner join on dates with a valid VIX close
    report: dict = field(default_factory=dict)
    tbill: pd.DataFrame | None = None   # RBI 91-day T-bill auctions (yield_pct, rate); None for synthetic fixtures


def load_all(raw_dir: Path = cfg.RAW_DIR, drop_weekend_sessions: bool = cfg.DROP_WEEKEND_SESSIONS,
             verify_integrity: bool = True) -> RealData:
    """Load, clean and align.  The shipped raw folder is checked against data/raw/MANIFEST.json first
    (other folders, e.g. test fixtures, are not)."""
    if verify_integrity and Path(raw_dir).resolve() == Path(cfg.RAW_DIR).resolve():
        from . import integrity
        integrity.assert_ok(raw_dir)
    nifty = load_nifty(raw_dir)
    vix = load_vix(raw_dir)
    from . import rates
    tbill = rates.load_tbill(raw_dir)
    if drop_weekend_sessions:
        nifty = nifty[nifty.index.dayofweek < 5]
        vix = vix[vix.index.dayofweek < 5]

    stale_dates = vix.index[vix["stale"]]
    both = nifty.index.intersection(vix.index)
    only_nifty = nifty.index.difference(vix.index)
    only_vix = vix.index.difference(nifty.index)

    aligned = pd.DataFrame(
        {
            "nifty_close": nifty.loc[both, "Close"],
            "vix_pct": vix.loc[both, "Close"],
        }
    )
    n_before = len(aligned)
    aligned = aligned.dropna()
    aligned["sigma_implied"] = aligned["vix_pct"] / 100.0      # VIX is quoted in percent

    report = dict(
        nifty_rows=len(nifty),
        vix_rows=len(vix),
        vix_stale_dates=[d.date().isoformat() for d in stale_dates],
        common_dates=len(both),
        aligned_rows=len(aligned),
        dropped_for_missing_vix=n_before - len(aligned),
        only_in_nifty=len(only_nifty),
        only_in_nifty_first_last=(
            (only_nifty.min().date().isoformat(), only_nifty.max().date().isoformat()) if len(only_nifty) else None
        ),
        only_in_vix=len(only_vix),
        first_date=aligned.index.min().date().isoformat(),
        last_date=aligned.index.max().date().isoformat(),
        weekend_sessions_kept=int((aligned.index.dayofweek >= 5).sum()),
    )
    return RealData(nifty=nifty, vix=vix, aligned=aligned, report=report, tbill=tbill)


# ----------------------------------------------------------------------------- validation (checklist #7)
def validate_aligned(aligned: pd.DataFrame) -> list[tuple[str, bool, str]]:
    """Return [(check name, passed, detail)] for the real-data sanity checks."""
    lo, hi = cfg.VIX_PLAUSIBLE_RANGE_PCT
    out = []
    n_nan = int(aligned.isna().sum().sum())
    out.append(("no NaNs after alignment", n_nan == 0, f"{n_nan} NaN cells in {len(aligned)} rows"))
    inc = bool(aligned.index.is_monotonic_increasing and aligned.index.is_unique)
    out.append(("dates strictly increasing and unique", inc, f"{aligned.index[0].date()} -> {aligned.index[-1].date()}"))
    vmin, vmed, vmax = aligned["vix_pct"].min(), aligned["vix_pct"].median(), aligned["vix_pct"].max()
    out.append(
        (
            "VIX quoted in percent and in plausible range",
            bool(lo <= vmin and vmax <= hi and 8.0 <= vmed <= 40.0),
            f"min={vmin:.2f} median={vmed:.2f} max={vmax:.2f} (allowed {lo}-{hi})",
        )
    )
    s_min, s_max = aligned["sigma_implied"].min(), aligned["sigma_implied"].max()
    out.append(
        (
            "sigma_implied = VIX/100 lies in (0.05, 1.0)",
            bool(0.05 <= s_min and s_max <= 1.0),
            f"min={s_min:.4f} max={s_max:.4f}",
        )
    )
    return out


# ----------------------------------------------------------------------------- windows
@dataclass
class WindowSet:
    t0: pd.DatetimeIndex           # window start dates
    t_end: pd.DatetimeIndex        # window end dates
    paths: np.ndarray              # (M, W+1) NIFTY closes from t0 to t_end
    sigma_implied: np.ndarray      # (M,) VIX_t0 / 100
    vix_pct: np.ndarray            # (M,)
    calendar_days: np.ndarray      # (M,) calendar days spanned
    window: int                    # trading days per window (W)
    n_excluded_long_span: int
    r: np.ndarray | None = None               # (M,) continuously compounded risk-free rate in force at t0 (None: constant fallback)
    tbill_yield_pct: np.ndarray | None = None  # (M,) the 91-day T-bill auction yield it was built from, percent
    rate_stale_days: np.ndarray | None = None  # (M,) days between t0 and that auction


MAX_CALENDAR_SPAN = 45             # guard: a 21-trading-day window should span ~27-35 calendar days


def build_windows(data: RealData, window: int = cfg.WINDOW_TRADING_DAYS) -> WindowSet:
    """Rolling windows of ``window`` trading days, stride 1 day (so windows OVERLAP).

    Paths come from the FULL NIFTY series (so the missing Dec-2022 VIX does not punch a hole in
    the price paths); a window is kept only if India VIX is valid on its start date, since
    sigma_implied = VIX at t0.
    """
    if isinstance(window, bool) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window must be a positive integer number of trading days")
    close = data.nifty["Close"].to_numpy()
    dates = data.nifty.index
    pos = pd.Series(np.arange(len(dates)), index=dates)
    starts = pos.loc[data.aligned.index].to_numpy()
    starts = starts[starts + window < len(dates)]          # window must fit

    idx = starts[:, None] + np.arange(window + 1)[None, :]
    paths = close[idx]
    t0 = dates[starts]
    t_end = dates[starts + window]
    cal = (t_end - t0).days.to_numpy()
    keep = cal <= MAX_CALENDAR_SPAN
    n_long = int((~keep).sum())

    vix_pct = data.aligned.loc[t0[keep], "vix_pct"].to_numpy()
    r = y = stale = None
    if data.tbill is not None:
        from . import rates
        r, y, stale = rates.rate_asof(t0[keep], data.tbill)
    return WindowSet(
        t0=t0[keep],
        t_end=t_end[keep],
        paths=paths[keep],
        sigma_implied=vix_pct / 100.0,
        vix_pct=vix_pct,
        calendar_days=cal[keep],
        window=window,
        n_excluded_long_span=n_long,
        r=r, tbill_yield_pct=y, rate_stale_days=stale,
    )
