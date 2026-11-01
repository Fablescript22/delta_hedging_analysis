"""Behaviour when the raw NSE files are absent (the public repository ships without them)."""
import pandas as pd
import pytest

from src import config as cfg
from src import data_dictionary as dd
from src import integrity as ig
from src import make_readme

needs_results = pytest.mark.skipif(not (cfg.RESULTS_DIR / "partB_summary.json").exists(), reason="results not generated")


@pytest.fixture()
def no_raw(tmp_path, monkeypatch):
    monkeypatch.setattr(cfg, "RAW_DIR", tmp_path)          # an empty data folder
    return tmp_path


def test_raw_data_available_reflects_the_folder(no_raw):
    assert cfg.raw_data_available() is False
    (no_raw / "x.csv").write_text("a\n1\n")
    assert cfg.raw_data_available() is True


def test_integrity_check_explains_what_to_do_when_there_are_no_files(tmp_path):
    problems = ig.verify(tmp_path)
    assert len(problems) == 1
    assert "not part of the public repository" in problems[0] and "SOURCES.md" in problems[0]


def test_integrity_check_still_lists_individual_problems_when_some_files_exist(tmp_path):
    (tmp_path / "NIFTY 50_Historical_PR_extra.csv").write_text("Index Name,Date\n")
    problems = "\n".join(ig.verify(tmp_path))
    assert "missing file" in problems and "unexpected file" in problems


def test_loader_stops_with_the_download_message_when_data_is_absent(no_raw):
    from src import real_data as rd
    monkey_dir = no_raw
    with pytest.raises(ig.IntegrityError, match="not part of the public repository"):
        rd.load_all(monkey_dir)


@needs_results
def test_readme_and_report_adapt_to_the_missing_data(no_raw):
    readme, report = make_readme.build(), make_readme.build_report()
    assert "Getting the data" in readme and "does not include the raw NIFTY 50" in readme
    assert "this working copy keeps them" not in readme and "per-window results table" in readme
    assert "Getting the data" in report and "The raw files are in `data/raw/`, unmodified" not in report
    assert "CSV files are not in the public repository" in report


@needs_results
def test_readme_and_report_with_data_present_do_not_carry_the_public_notice(monkeypatch, tmp_path):
    (tmp_path / "x.csv").write_text("a\n1\n")
    monkeypatch.setattr(cfg, "RAW_DIR", tmp_path)
    readme, report = make_readme.build(), make_readme.build_report()
    assert "Getting the data" not in readme and "this working copy keeps them" in readme
    assert "The raw files are in `data/raw/`, unmodified" in report


@needs_results
def test_data_dictionary_is_identical_with_or_without_the_windows_table(monkeypatch):
    full = dd.build()
    monkeypatch.setattr(dd, "results_files",
                        lambda: [p for p in sorted(cfg.RESULTS_DIR.glob("*.csv")) if p.name != dd.WINDOWS_FILE])
    assert dd.build() == full
    assert dd.missing() == []


@needs_results
def test_static_windows_columns_match_the_real_table_and_are_described():
    assert all(dd.describe(dd.WINDOWS_FILE, c) for c in dd.WINDOWS_COLUMNS)
    f = cfg.RESULTS_DIR / dd.WINDOWS_FILE
    if f.exists():
        assert tuple(pd.read_csv(f, nrows=0).columns) == dd.WINDOWS_COLUMNS


# ---------------------------------------------------------------- python -m src make-public
from src import make_public as mp  # noqa: E402


def test_exclusion_rules():
    assert mp.excluded("data/raw/NIFTY 50_Historical_PR_01012016to31122016.csv")
    assert mp.excluded("data/raw/hist_india_vix_-01-01-2016-to-31-12-2016.csv")
    assert mp.excluded("results/partB_windows.csv")
    assert mp.excluded("src/__pycache__/x.pyc") and mp.excluded(".venv/lib/x.py") and mp.excluded(".git/config")
    for kept in ("data/raw/MANIFEST.json", "data/raw/SOURCES.md", "results/partB_summary.json", "src/backtest.py",
                 "tests/test_bs.py", "README.md", ".github/workflows/ci.yml", ".gitignore", "results/reproduce_stats.json"):
        assert not mp.excluded(kept), kept


def test_make_public_copies_everything_except_the_excluded_files(tmp_path):
    dest = mp.make_public(tmp_path / "pub")
    rel = {p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file()}
    assert not [r for r in rel if r.endswith(".csv") and r.startswith("data/")]
    assert "results/partB_windows.csv" not in rel
    assert {"data/raw/MANIFEST.json", "data/raw/SOURCES.md", "src/make_public.py", "README.md", ".gitignore",
            ".github/workflows/ci.yml", "docs/RESEARCH_REPORT.md", "results/partB_summary.json"} <= rel
    assert not [r for r in rel if "explainer_im_7" in r or "__pycache__" in r or r.startswith(".git/")]
    # the working tree is untouched
    assert (cfg.RESULTS_DIR / "partB_summary.json").exists()


def test_make_public_refuses_unsafe_destinations(tmp_path):
    with pytest.raises(ValueError, match="outside the project"):
        mp.make_public(cfg.ROOT / "inside")
    with pytest.raises(ValueError, match="outside the project"):
        mp.make_public(cfg.ROOT.parent)
    full = tmp_path / "full"
    full.mkdir()
    (full / "x").write_text("1")
    with pytest.raises(ValueError, match="not empty"):
        mp.make_public(full)


def test_make_public_command_needs_exactly_one_argument(capsys):
    assert mp.main([]) == 2 and mp.main(["a", "b"]) == 2
