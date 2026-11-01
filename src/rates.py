"""Risk-free rate for Part B: the RBI 91-day Treasury-bill auction cut-off yield, looked up as of each window's start.

Source: Reserve Bank of India, *Handbook of Statistics on the Indian Economy*, table "Auctions of 91-Day Government of
India Treasury Bills" (column "Implicit Yield at Cut-off Price", actual/365 day count).  The original spreadsheets are
kept in ``data/raw/rbi_xlsx/`` (working copy only) and flattened into one file, ``data/raw/RBI_91day_tbill_auctions.csv``,
by ``python -m src.rates`` (conflicting duplicates between overlapping editions are an error).

Conventions
-----------
* **No look-ahead.**  A window that starts on date t uses the most recent auction held on or before t.
* **Yield to a continuously compounded rate.**  The implicit yield y is a simple annualised rate on a 91-day bill,
  price = 100 / (1 + y * 91/365).  Black-Scholes needs a continuously compounded rate r, so
  r = (365/91) * ln(1 + y * 91/365).  (About 0.07 percentage points lower than y at y = 6%.)
* **Auctions with no accepted bids** (yield printed as "-") are skipped; the previous auction is used.
* **Staleness.**  The weekly auction series has gaps (holidays, skipped auctions, and the end of the data).  A window
  whose latest auction is older than ``MAX_STALE_DAYS`` is an error, not a silent forward-fill; one older than
  ``FLAG_STALE_DAYS`` is flagged in the window table.
"""
from __future__ import annotations

import csv
import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

TBILL_FILE = "RBI_91day_tbill_auctions.csv"
TBILL_GLOB = "RBI_91day_tbill*.csv"
XLSX_DIR = cfg.RAW_DIR / "rbi_xlsx"
BILL_DAYS, YEAR_DAYS = 91, 365
FLAG_STALE_DAYS = 28
MAX_STALE_DAYS = 120
PLAUSIBLE_YIELD_PCT = (0.5, 20.0)


class RateError(RuntimeError):
    pass


def continuous_rate(yield_pct) -> np.ndarray:
    """Continuously compounded annual rate (decimal) equivalent to an implicit 91-day T-bill yield in percent."""
    y = np.asarray(yield_pct, dtype=float) / 100.0
    if not np.all(np.isfinite(y)):
        raise ValueError("yields must be finite")
    return (YEAR_DAYS / BILL_DAYS) * np.log1p(y * BILL_DAYS / YEAR_DAYS)


def load_tbill(raw_dir: Path = cfg.RAW_DIR) -> pd.DataFrame | None:
    """Auction dates (index, ascending) with ``yield_pct`` and ``rate`` (continuous, decimal); None if the file is absent."""
    files = sorted(Path(raw_dir).glob(TBILL_GLOB))
    if not files:
        return None
    df = pd.read_csv(files[0])
    df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d")
    df = df.set_index("Date").sort_index()
    if df.index.has_duplicates:
        raise RateError("duplicate auction dates in the T-bill file")
    y = df["Cutoff_Yield_Percent"].astype(float)
    lo, hi = PLAUSIBLE_YIELD_PCT
    if not y.between(lo, hi).all():
        raise RateError(f"T-bill yields outside the plausible range {lo}-{hi}%")
    return pd.DataFrame({"yield_pct": y, "rate": continuous_rate(y.to_numpy())}, index=df.index)


def rate_asof(dates, tbill: pd.DataFrame, max_stale_days: int = MAX_STALE_DAYS):
    """(rate, yield_pct, stale_days) arrays for each date: the latest auction on or before it.

    Raises ``RateError`` if a date precedes the first auction or is more than ``max_stale_days`` after the latest one.
    """
    dates = pd.DatetimeIndex(dates)
    pos = tbill.index.searchsorted(dates, side="right") - 1
    if np.any(pos < 0):
        raise RateError(f"{int((pos < 0).sum())} window start date(s) precede the first T-bill auction "
                        f"({tbill.index[0].date()})")
    used = tbill.index[pos]
    stale = (dates - used).days.to_numpy()
    if stale.max() > max_stale_days:
        raise RateError(f"a window start is {int(stale.max())} days after the latest T-bill auction (limit {max_stale_days}); "
                        "add a newer RBI table to data/raw/rbi_xlsx and rebuild")
    return tbill["rate"].to_numpy()[pos], tbill["yield_pct"].to_numpy()[pos], stale


# ------------------------------------------------------------------------------------------- build the CSV
def _as_date(v):
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, str):
        for fmt in ("%d-%b-%Y", "%Y-%m-%d"):
            try:
                return dt.datetime.strptime(v.strip(), fmt).date()
            except ValueError:
                pass
    return None


def read_xlsx(path: Path) -> list[tuple[dt.date, float]]:
    import openpyxl
    ws = openpyxl.load_workbook(path, data_only=True).worksheets[0]
    rows = list(ws.iter_rows(values_only=True))
    hdr = next(r for r in rows if r[1] and str(r[1]).strip() == "Date of Auction")
    col = next(i for i, c in enumerate(hdr) if c and "Implicit Yield" in str(c))
    out = []
    for r in rows:
        d, y = _as_date(r[1]), r[col]
        if d is not None and isinstance(y, (int, float)) and not isinstance(y, bool):
            out.append((d, round(float(y), 4)))
    return out


def build_csv(xlsx_dir: Path = XLSX_DIR, out: Path = cfg.RAW_DIR / TBILL_FILE) -> pd.DataFrame:
    files = sorted(Path(xlsx_dir).glob("*.xlsx"))
    if not files:
        raise RateError(f"no .xlsx files in {xlsx_dir}")
    seen: dict[dt.date, tuple[float, str]] = {}
    for f in files:
        for d, y in read_xlsx(f):
            if d in seen and abs(seen[d][0] - y) > 1e-9:
                raise RateError(f"{d}: {f.name} says {y}, {seen[d][1]} says {seen[d][0]}")
            seen.setdefault(d, (y, f.name))
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["Date", "Cutoff_Yield_Percent", "Source_File"])
        for d in sorted(seen):
            w.writerow([d.isoformat(), f"{seen[d][0]:.4f}", seen[d][1]])
    return pd.read_csv(out)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    df = build_csv(*(Path(a) for a in argv))
    print(f"wrote {len(df)} auctions, {df['Date'].iloc[0]} to {df['Date'].iloc[-1]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
