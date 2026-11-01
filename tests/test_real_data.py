import numpy as np
import pytest

from src import config as cfg
from src import real_data as rd


@pytest.fixture(scope="module")
def data():
    return rd.load_all()


def test_raw_files_are_untouched_readable_and_complete():
    assert len(list(cfg.RAW_DIR.glob(rd.NIFTY_GLOB))) == 11
    assert len(list(cfg.RAW_DIR.glob(rd.VIX_GLOB))) == 11


def test_loaders_return_ascending_unique_dates(data):
    for df in (data.nifty, data.vix, data.aligned):
        assert df.index.is_monotonic_increasing and df.index.is_unique


def test_stale_vix_rows_are_detected_and_not_filled(data):
    assert data.report["vix_stale_dates"] == ["2021-02-12", "2021-03-30"]
    assert data.vix.loc["2021-02-12", "Close"] != data.vix.loc["2021-02-12", "Close"]    # NaN
    assert "2021-02-12" not in data.aligned.index.strftime("%Y-%m-%d")


def test_december_2022_has_no_vix_and_no_window_starts_there(data):
    assert not ((data.aligned.index >= "2022-12-02") & (data.aligned.index <= "2022-12-31")).any()
    ws = rd.build_windows(data)
    assert not ((ws.t0 >= "2022-12-02") & (ws.t0 <= "2022-12-31")).any()


def test_vix_scaling_to_decimal(data):
    assert np.allclose(data.aligned["sigma_implied"], data.aligned["vix_pct"] / 100)
    assert data.aligned["sigma_implied"].between(0.05, 1.0).all()


def test_windows_shape_and_content(data):
    ws = rd.build_windows(data)
    M = len(ws.t0)
    assert ws.paths.shape == (M, cfg.WINDOW_TRADING_DAYS + 1)
    assert np.all(ws.paths > 0) and np.all(np.isfinite(ws.paths))
    assert ws.calendar_days.min() >= 20 and ws.calendar_days.max() <= rd.MAX_CALENDAR_SPAN
    # first price of each window is the NIFTY close on t0
    assert np.allclose(ws.paths[:, 0], data.nifty.loc[ws.t0, "Close"].to_numpy())
    # last price is the NIFTY close on t_end, exactly 21 trading rows later
    assert np.allclose(ws.paths[:, -1], data.nifty.loc[ws.t_end, "Close"].to_numpy())
    assert np.allclose(ws.sigma_implied, data.aligned.loc[ws.t0, "vix_pct"].to_numpy() / 100)


def test_window_dates_follow_the_nifty_calendar(data):
    ws = rd.build_windows(data)
    pos = {d: i for i, d in enumerate(data.nifty.index)}
    gaps = np.array([pos[e] - pos[s] for s, e in zip(ws.t0, ws.t_end)])
    assert np.all(gaps == cfg.WINDOW_TRADING_DAYS)
