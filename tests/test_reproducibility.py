"""Reproducibility plumbing: integrity manifest, data dictionary, documented dependencies, entry points."""
import json
import shutil
from pathlib import Path

import pytest

from src import __main__ as entry
from src import config as cfg
from src import data_dictionary as dd
from src import integrity as ig

ROOT = Path(__file__).resolve().parents[1]


def test_shipped_raw_data_matches_manifest():
    assert ig.verify() == []


def test_manifest_covers_all_22_files_with_sane_metadata():
    m = json.loads(ig.MANIFEST.read_text())["files"]
    assert len(m) == 22 and len({f["file"] for f in m}) == 22
    assert sum(f["rows"] for f in m) == 5326
    assert all(len(f["sha256"]) == 64 for f in m)


def _copy_raw(tmp_path):
    d = tmp_path / "raw"
    shutil.copytree(cfg.RAW_DIR, d)
    return d


def test_integrity_detects_modified_file(tmp_path):
    d = _copy_raw(tmp_path)
    f = sorted(d.glob("hist_india_vix_*2019*.csv"))[0]
    f.write_bytes(f.read_bytes().replace(b"1", b"2", 1))
    problems = ig.verify(d)
    assert len(problems) == 1 and "changed file" in problems[0] and f.name in problems[0]


def test_integrity_detects_missing_and_unexpected_files(tmp_path):
    d = _copy_raw(tmp_path)
    gone = sorted(d.glob("NIFTY*2020*.csv"))[0]
    gone.unlink()
    (d / "NIFTY 50_Historical_PR_extra.csv").write_text("Index Name,Date\n")
    problems = "\n".join(ig.verify(d))
    assert "missing file" in problems and gone.name in problems and "unexpected file" in problems


def test_integrity_detects_missing_manifest(tmp_path):
    assert "manifest not found" in ig.verify(cfg.RAW_DIR, tmp_path / "nope.json")[0]


def test_loader_refuses_tampered_default_data(monkeypatch):
    monkeypatch.setattr(ig, "verify", lambda *a, **k: ["changed file: x.csv"])
    from src import real_data as rd
    with pytest.raises(ig.IntegrityError):
        rd.load_all()


def test_data_dictionary_describes_every_results_column():
    assert dd.missing() == []


def test_data_dictionary_file_is_current():
    if not (cfg.RESULTS_DIR / "partB_summary.json").exists():
        pytest.skip("results not generated")
    assert dd.DOC.read_text() == dd.build()


def test_dependencies_documented_and_pinned():
    req = (ROOT / "requirements.txt").read_text()
    lock = (ROOT / "requirements-lock.txt").read_text()
    for pkg in ("numpy", "pandas", "scipy", "matplotlib", "pytest"):
        assert pkg in req and f"{pkg}==" in lock.lower().replace("_", "-")
    assert (ROOT / ".python-version").read_text().strip()
    assert "-text" in (ROOT / ".gitattributes").read_text()      # raw bytes must survive git checkouts


def test_ci_workflow_runs_tests_and_integrity():
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "python -m src test" in ci and "python -m src check-data" in ci and "python -m src reproduce" in ci


def test_entry_point_rejects_unknown_commands(capsys):
    assert entry.main(["bogus"]) == 2
    assert entry.main([]) == 2
    assert entry.main(["check-data"]) == 0


def test_caches_and_virtualenvs_are_git_ignored():
    gi = (ROOT / ".gitignore").read_text()
    for pat in ("__pycache__/", "*.pyc", ".pytest_cache/", ".venv/", "venv/"):
        assert pat in gi
