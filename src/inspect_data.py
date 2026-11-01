"""Raw-data inspection report for the NIFTY 50 and India VIX CSV files.

Reads every CSV in data/raw/ WITHOUT modifying it and prints a report:
columns, row counts, date ranges, missing / non-numeric values, duplicate
dates, ordering, calendar gaps, OHLC sanity, and cross-series date mismatches.

Usage:  python src/inspect_data.py
"""
from __future__ import annotations

import glob
import os
import sys

import numpy as np
import pandas as pd

RAW = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "raw")


def read_group(pattern: str, date_fmt: str) -> tuple[pd.DataFrame, list[dict]]:
    """Read all files matching pattern; return stacked frame + per-file stats."""
    files = sorted(glob.glob(os.path.join(RAW, pattern)))
    frames, stats = [], []
    for f in files:
        # utf-8-sig strips the BOM that the VIX files carry
        df = pd.read_csv(f, encoding="utf-8-sig", dtype=str)
        raw_cols = list(df.columns)
        df.columns = [c.strip() for c in df.columns]
        n_rows = len(df)
        d = pd.to_datetime(df["Date"].str.strip(), format=date_fmt, errors="coerce")
        bad_dates = int(d.isna().sum())
        numeric_cols = [c for c in df.columns if c not in ("Index Name", "Date")]
        num = df[numeric_cols].apply(lambda s: pd.to_numeric(s.str.strip(), errors="coerce"))
        raw_blank = int(df[numeric_cols].isna().sum().sum())
        non_numeric = int(num.isna().sum().sum()) - raw_blank
        df2 = pd.concat([d.rename("Date"), num], axis=1)
        if "Index Name" in df.columns:
            df2["Index Name"] = df["Index Name"]
        stats.append(
            dict(
                file=os.path.basename(f),
                raw_columns=raw_cols,
                rows=n_rows,
                first=d.min(),
                last=d.max(),
                bad_dates=bad_dates,
                blank_cells=raw_blank,
                non_numeric_cells=non_numeric,
                order=("descending" if d.is_monotonic_decreasing else
                       "ascending" if d.is_monotonic_increasing else "unsorted"),
            )
        )
        frames.append(df2)
    return pd.concat(frames, ignore_index=True), stats


def describe_group(name: str, df: pd.DataFrame, stats: list[dict], price_col: str) -> pd.DataFrame:
    print(f"\n{'=' * 78}\n{name}\n{'=' * 78}")
    print(f"Files: {len(stats)}")
    print(f"Raw header of first file: {stats[0]['raw_columns']}")
    for s in stats:
        print(
            f"  {s['file']:<52} rows={s['rows']:>4}  {s['first'].date()} -> {s['last'].date()}  "
            f"order={s['order']:<10} bad_dates={s['bad_dates']} blank={s['blank_cells']} "
            f"non_numeric={s['non_numeric_cells']}"
        )
    all_same = len({tuple(s["raw_columns"]) for s in stats}) == 1
    print(f"Same header in every file: {all_same}")

    n_total = len(df)
    dup = int(df["Date"].duplicated().sum())
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"Total rows: {n_total}   duplicate dates: {dup}   "
          f"range: {df['Date'].min().date()} -> {df['Date'].max().date()}")
    print("NaN counts per column after parsing:")
    print(df.drop(columns=[c for c in ["Index Name"] if c in df.columns]).isna().sum().to_string())

    wk = df["Date"].dt.dayofweek
    print(f"Weekend-dated rows: {int((wk >= 5).sum())}")
    gaps = df["Date"].diff().dt.days
    big = df.loc[gaps > 5, "Date"]
    print(f"Calendar gaps > 5 days: {len(big)}")
    for i in big.index:
        print(f"    {df.loc[i - 1, 'Date'].date()} -> {df.loc[i, 'Date'].date()} ({int(gaps[i])} days)")

    per_year = df.groupby(df["Date"].dt.year).size()
    print("Rows per calendar year:", per_year.to_dict())

    ohlc = [c for c in ("Open", "High", "Low", "Close") if c in df.columns]
    if len(ohlc) == 4:
        bad_hl = int((df["High"] < df["Low"]).sum())
        bad_c = int(((df["Close"] > df["High"] + 1e-9) | (df["Close"] < df["Low"] - 1e-9)).sum())
        print(f"OHLC sanity: High<Low rows={bad_hl}; Close outside [Low,High] rows={bad_c}")
    print(f"{price_col} summary: min={df[price_col].min():.4f}  max={df[price_col].max():.4f}  "
          f"mean={df[price_col].mean():.4f}  zeros/non-positive={int((df[price_col] <= 0).sum())}")
    return df


def main() -> int:
    nifty, ns = read_group("NIFTY 50_Historical_PR_*.csv", "%d %b %Y")
    vix, vs = read_group("hist_india_vix_*.csv", "%d-%b-%Y")

    nifty = describe_group("NIFTY 50 (price return index)", nifty, ns, "Close")
    vix = describe_group("India VIX", vix, vs, "Close")

    # Extra NIFTY checks
    print("\n-- NIFTY extra checks --")
    print("Index Name values:", nifty["Index Name"].unique().tolist())
    r = np.log(nifty["Close"]).diff().dropna()
    print(f"Daily log-return: mean={r.mean():.6f} std={r.std():.6f} "
          f"min={r.min():.4f} ({nifty.loc[r.idxmin(), 'Date'].date()}) "
          f"max={r.max():.4f} ({nifty.loc[r.idxmax(), 'Date'].date()})")
    print(f"|return| > 7%: {int((r.abs() > 0.07).sum())} days")

    # Extra VIX checks
    print("\n-- VIX extra checks --")
    q = vix["Close"].quantile([0, .01, .5, .99, 1]).round(2).to_dict()
    print("VIX Close quantiles (0,1%,50%,99%,100%):", q)
    top = vix.nlargest(3, "Close")[["Date", "Close"]]
    print("Three highest VIX closes:", [(d.date().isoformat(), v) for d, v in zip(top.Date, top.Close)])

    # Cross-series alignment
    print("\n-- Cross-series alignment --")
    nd, vd = set(nifty["Date"]), set(vix["Date"])
    both = sorted(nd & vd)
    only_n = sorted(nd - vd)
    only_v = sorted(vd - nd)
    print(f"NIFTY dates: {len(nd)}  VIX dates: {len(vd)}  common: {len(both)}")
    print(f"Only in NIFTY: {len(only_n)}   Only in VIX: {len(only_v)}")
    for label, lst in (("Only-in-NIFTY", only_n), ("Only-in-VIX", only_v)):
        by_year = pd.Series([d.year for d in lst]).value_counts().sort_index().to_dict()
        print(f"  {label} by year: {by_year}")
        if 0 < len(lst) <= 60:
            print("   ", [d.date().isoformat() for d in lst])
        elif len(lst) > 60:
            print("    first 10:", [d.date().isoformat() for d in lst[:10]])
            print("    last 10: ", [d.date().isoformat() for d in lst[-10:]])
    print(f"Common range: {both[0].date()} -> {both[-1].date()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
