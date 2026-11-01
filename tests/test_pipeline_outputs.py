"""Staleness protection: committed README / gallery / tables must equal what the code regenerates, the
snapshot-compare tool must catch real differences, and the reproduce pipeline must run stage by stage."""
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
import pytest

from src import __main__ as entry
from src import config as cfg
from src import make_gallery, make_readme, stages, verify_outputs as vo

have_results = (cfg.RESULTS_DIR / "partB_summary.json").exists()
needs_results = pytest.mark.skipif(not have_results, reason="results not generated")


# ---------------------------------------------------------------------------------- documents are current
@needs_results
def test_committed_readme_equals_regenerated_readme():
    committed = (cfg.ROOT / "README.md").read_text()
    assert vo.mask_volatile(committed) == vo.mask_volatile(make_readme.build()), \
        "README.md is stale: run `python -m src reproduce`"


@needs_results
def test_research_report_equals_regenerated_report():
    committed = (cfg.ROOT / "docs" / "RESEARCH_REPORT.md").read_text()
    assert vo.mask_volatile(committed) == vo.mask_volatile(make_readme.build_report()), \
        "docs/RESEARCH_REPORT.md is stale: run `python -m src reproduce`"


@needs_results
def test_figure_gallery_is_current_and_covers_every_figure():
    assert (cfg.ROOT / "docs" / "FIGURE_GALLERY.md").read_text() == make_gallery.build()
    listed = {n for g in make_gallery.GROUPS.values() for n in g}
    on_disk = {p.name for p in cfg.FIG_DIR.glob("*.png")}
    assert on_disk <= listed, f"figures with no caption in the gallery: {sorted(on_disk - listed)}"
    assert listed <= on_disk, f"gallery captions for missing figures: {sorted(listed - on_disk)}"


@needs_results
@pytest.mark.parametrize("doc", ["README.md", "docs/RESEARCH_REPORT.md", "docs/FIGURE_GALLERY.md", "docs/OPTIONS_EXTENSION_PLAN.md"])
def test_document_links_resolve(doc):
    import re
    path = cfg.ROOT / doc
    for target in re.findall(r"\]\(([^)#]+?)(?:#[^)]*)?\)", path.read_text()):
        if target.startswith(("http://", "https://", "mailto:")):
            continue
        assert (path.parent / target).resolve().exists(), f"{doc} links to a missing file: {target}"


def test_readme_is_a_short_landing_page_with_the_required_parts():
    text = make_readme.build() if have_results else None
    if text is None:
        pytest.skip("results not generated")
    assert len(text) < 12_000                                       # a landing page, not a 70 KB report
    for needle in ("**Problem.**", "![Hedging error against rebalancing frequency]", "## Three findings", "## Method in brief",
                   "## What this does not show", "## Reproduce", "python -m src reproduce", "python -m src test",
                   "docs/RESEARCH_REPORT.md", "EXPLAINER.md", "docs/FIGURE_GALLERY.md", "docs/DATA_DICTIONARY.md",
                   "no option quotes", "no futures hedge", "not a profitable options strategy", "actions/workflows/ci.yml/badge.svg"):
        assert needle.lower() in text.lower(), needle


def test_documents_do_not_mention_employers_or_applications():
    for doc in ("README.md", "EXPLAINER.md", "docs/RESEARCH_REPORT.md", "docs/OPTIONS_EXTENSION_PLAN.md", "data/raw/SOURCES.md"):
        text = (cfg.ROOT / doc).read_text().lower()
        for word in ("internship", "application", "applicant", "recruit", "interview", "hiring"):
            assert word not in text.replace("applications of", "").replace("application of", ""), (doc, word)


# ---------------------------------------------------------------------------------- compare tool
def test_mask_volatile_removes_only_marked_regions():
    t = "a <!--volatile-->12 s<!--/volatile--> b <!--volatile-->x<!--/volatile-->"
    assert vo.mask_volatile(t) == "a <!--volatile--> b <!--volatile-->"


def test_compare_json_tolerates_rounding_but_catches_changes():
    base = {"a": 1.0, "b": [1, 2.0, {"c": 3.0}], "s": "x", "n": None}
    assert vo.compare_json(base, json.loads(json.dumps(base))) == []
    near = {"a": 1.0 + 1e-9, "b": [1, 2.0, {"c": 3.0}], "s": "x", "n": None}
    assert vo.compare_json(base, near) == []
    far = {"a": 1.01, "b": [1, 2.0, {"c": 3.0}], "s": "x", "n": None}
    assert len(vo.compare_json(base, far)) == 1
    assert vo.compare_json(base, {**base, "extra": 1})
    assert vo.compare_json({"a": [1, 2]}, {"a": [1]})


def test_compare_csv_numeric_and_text(tmp_path):
    a = pd.DataFrame({"x": [1.0, 2.0, float("nan")], "s": ["u", "v", "w"]})
    a.to_csv(tmp_path / "a.csv", index=False)
    a.to_csv(tmp_path / "same.csv", index=False)
    assert vo.compare_csv(tmp_path / "a.csv", tmp_path / "same.csv") == []
    b = a.copy(); b.loc[1, "x"] = 2.5
    b.to_csv(tmp_path / "num.csv", index=False)
    assert "row 1" in vo.compare_csv(tmp_path / "a.csv", tmp_path / "num.csv")[0]
    c = a.copy(); c.loc[0, "s"] = "z"
    c.to_csv(tmp_path / "txt.csv", index=False)
    assert "text differs" in vo.compare_csv(tmp_path / "a.csv", tmp_path / "txt.csv")[0]
    a.iloc[:2].to_csv(tmp_path / "short.csv", index=False)
    assert "shape" in vo.compare_csv(tmp_path / "a.csv", tmp_path / "short.csv")[0]


@needs_results
def test_snapshot_verifies_itself_and_detects_tampering(tmp_path):
    snap = vo.snapshot(tmp_path / "snap")
    assert vo.verify(snap) == []
    # make a copy of the repo outputs and change one number, one document and remove a table
    root = tmp_path / "root"
    for sub in ("results", "docs", "figures"):
        shutil.copytree(cfg.ROOT / sub, root / sub)
    shutil.copy(cfg.ROOT / "README.md", root / "README.md")
    f = root / "results" / "partB_effect_size_intervals.csv"
    df = pd.read_csv(f); df.loc[0, "estimate"] *= 1.5; df.to_csv(f, index=False)
    (root / "README.md").write_text((root / "README.md").read_text() + "\nstale\n")
    (root / "results" / "partB_yearly.csv").unlink()
    problems = "\n".join(vo.verify(snap, root))
    assert "partB_effect_size_intervals" in problems and "README.md" in problems and "partB_yearly" in problems


@needs_results
def test_timing_files_and_volatile_text_do_not_cause_differences(tmp_path):
    snap = vo.snapshot(tmp_path / "snap")
    root = tmp_path / "root"
    for sub in ("results", "docs", "figures"):
        shutil.copytree(cfg.ROOT / sub, root / sub)
    shutil.copy(cfg.ROOT / "README.md", root / "README.md")
    (root / "results" / "reproduce_stats.json").write_text('{"total_seconds": 999}')
    text = (root / "README.md").read_text()
    (root / "README.md").write_text(text.replace("<!--volatile-->", "<!--volatile-->different ", 1))
    assert vo.verify(snap, root) == []
    # the same holds for volatile regions inside docs/ (the research report quotes timings too)
    rep = root / "docs" / "RESEARCH_REPORT.md"
    rep.write_text(rep.read_text().replace("<!--volatile-->", "<!--volatile-->x ", 1))
    assert vo.verify(snap, root) == []
    rep.write_text(rep.read_text() + "\nstale line\n")
    assert any("RESEARCH_REPORT" in p for p in vo.verify(snap, root))


# ---------------------------------------------------------------------------------- stages
def test_stage_registry_matches_modules_and_order():
    import importlib
    assert list(stages.STAGES)[-1] == "docs"               # the README is built last, after every table it quotes
    for name, (mod, _) in stages.STAGES.items():
        assert callable(importlib.import_module(mod).main), name
    assert list(stages.STAGES)[:3] == ["validation", "part_a", "part_b"]


def test_stage_runner_records_stats(tmp_path):
    out = tmp_path / "s.json"
    code = stages.main(["data_dictionary", str(out)]) if have_results else 0
    if have_results:
        st = json.loads(out.read_text())
        assert code == 0 and st["stage"] == "data_dictionary" and st["seconds"] >= 0 and st["exit_code"] == 0
        assert st["peak_rss_mb"] is None or st["peak_rss_mb"] > 0


def test_stage_runner_rejects_unknown_stage():
    assert stages.main(["nope", "x.json"]) == 2


def test_entry_point_commands_and_usage(capsys):
    assert entry.main([]) == 2 and entry.main(["bogus"]) == 2
    assert entry.main(["snapshot"]) == 2 and entry.main(["check-data", "x"]) == 2
    assert {"reproduce", "test", "check-data", "snapshot", "verify-outputs", "make-public"} <= set(entry.COMMANDS)


def test_ci_verifies_outputs_against_committed_versions():
    ci = (cfg.ROOT / ".github" / "workflows" / "ci.yml").read_text()
    assert "snapshot" in ci and "verify-outputs" in ci
    assert ci.index("snapshot") < ci.index("python -m src reproduce") < ci.index("verify-outputs")
