"""RBI 91-day T-bill rate: conversion, as-of lookup, staleness, and the flattening of the RBI spreadsheets."""
import csv
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config as cfg
from src import rates


def _tb(rows):
    idx = pd.to_datetime([d for d, _ in rows])
    y = np.array([v for _, v in rows], float)
    return pd.DataFrame({"yield_pct": y, "rate": rates.continuous_rate(y)}, index=idx)


def test_continuous_rate_formula():
    y = 6.0
    r = float(rates.continuous_rate(y))
    assert r == pytest.approx((365 / 91) * math.log(1 + 0.06 * 91 / 365))
    # a 91-day bill bought at 100/(1+y*91/365) grows to 100 in 91/365 years at continuous rate r
    price = 100 / (1 + 0.06 * 91 / 365)
    assert price * math.exp(r * 91 / 365) == pytest.approx(100.0)
    assert r < 0.06                                                    # continuous compounding is lower for the same growth
    assert float(rates.continuous_rate(0.0)) == 0.0
    with pytest.raises(ValueError):
        rates.continuous_rate([5.0, float("nan")])


def test_asof_uses_latest_auction_on_or_before_and_never_a_later_one():
    tb = _tb([("2020-01-01", 6.0), ("2020-01-08", 5.0), ("2020-01-15", 4.0)])
    r, y, stale = rates.rate_asof(pd.to_datetime(["2020-01-01", "2020-01-07", "2020-01-08", "2020-01-14", "2020-01-20"]), tb)
    assert list(y) == [6.0, 6.0, 5.0, 5.0, 4.0]                       # an auction on the same day counts, a later one never does
    assert list(stale) == [0, 6, 0, 6, 5]
    assert r[0] == pytest.approx(float(rates.continuous_rate(6.0)))


def test_asof_errors_before_first_auction_and_when_too_stale():
    tb = _tb([("2020-01-08", 6.0)])
    with pytest.raises(rates.RateError, match="precede the first"):
        rates.rate_asof(pd.to_datetime(["2020-01-07"]), tb)
    with pytest.raises(rates.RateError, match="days after the latest"):
        rates.rate_asof(pd.to_datetime(["2020-06-01"]), tb)
    r, _, stale = rates.rate_asof(pd.to_datetime(["2020-06-01"]), tb, max_stale_days=200)
    assert stale[0] == 145


def _write(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["Date", "Cutoff_Yield_Percent", "Source_File"])
        w.writerows([(d, y, "x") for d, y in rows])


def test_loader_validates_the_file(tmp_path):
    assert rates.load_tbill(tmp_path) is None                         # absent -> None (synthetic fixtures)
    _write(tmp_path / "RBI_91day_tbill_auctions.csv", [("2020-01-01", 6.0), ("2020-01-01", 6.1)])
    with pytest.raises(rates.RateError, match="duplicate"):
        rates.load_tbill(tmp_path)
    _write(tmp_path / "RBI_91day_tbill_auctions.csv", [("2020-01-01", 60.0)])
    with pytest.raises(rates.RateError, match="plausible"):
        rates.load_tbill(tmp_path)


openpyxl = pytest.importorskip("openpyxl")


def _xlsx(path, rows, fmt="date"):
    import datetime as dt
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append([None, "TABLE : AUCTIONS OF 91-DAY GOVERNMENT OF INDIA TREASURY BILLS"])
    ws.append([None, "Date of Auction ", "Date of Issue", "x", "Implicit Yield at Cut-off Price (percent)"][:5])
    for d, y in rows:
        d0 = dt.datetime.strptime(d, "%Y-%m-%d")
        ws.append([None, d0 if fmt == "date" else d0.strftime("%d-%b-%Y"), d0, 1, y])
    wb.save(path)


def test_flattening_merges_editions_skips_no_bid_rows_and_rejects_conflicts(tmp_path):
    (tmp_path / "x").mkdir()
    _xlsx(tmp_path / "x" / "a.xlsx", [("2020-01-01", 6.0), ("2020-01-08", "-"), ("2020-01-15", 5.0)])
    _xlsx(tmp_path / "x" / "b.xlsx", [("2020-01-15", 5.0), ("2020-01-22", 4.5)], fmt="text")     # overlap agrees; text dates
    df = rates.build_csv(tmp_path / "x", tmp_path / "out.csv")
    assert list(df["Date"]) == ["2020-01-01", "2020-01-15", "2020-01-22"]
    assert list(df["Cutoff_Yield_Percent"]) == [6.0, 5.0, 4.5]
    _xlsx(tmp_path / "x" / "c.xlsx", [("2020-01-15", 5.1)])
    with pytest.raises(rates.RateError, match="2020-01-15"):
        rates.build_csv(tmp_path / "x", tmp_path / "out2.csv")


@pytest.mark.skipif(not (cfg.RAW_DIR / rates.TBILL_FILE).exists() or not rates.XLSX_DIR.exists(),
                    reason="RBI T-bill file / spreadsheets not present (see data/raw/SOURCES.md)")
def test_shipped_tbill_file_is_exactly_what_the_spreadsheets_flatten_to(tmp_path):
    out = tmp_path / "again.csv"
    again = rates.build_csv(rates.XLSX_DIR, out)
    assert out.read_text() == (cfg.RAW_DIR / rates.TBILL_FILE).read_text()
    tb = rates.load_tbill()
    assert len(tb) == len(again) == 564 and tb.index[0] == pd.Timestamp("2015-07-02") and tb.index[-1] == pd.Timestamp("2026-06-24")
    gaps = np.diff(tb.index.values).astype("timedelta64[D]").astype(int)
    assert gaps.max() <= 42                                            # no hidden hole in the weekly series
    assert 2.5 < tb["yield_pct"].min() and tb["yield_pct"].max() < 8
