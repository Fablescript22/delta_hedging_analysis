"""Build the public copy of the project:  python -m src make-public DEST

The public repository does not redistribute the NSE raw CSV files (their terms of use restrict copying and
republishing) or the per-window table ``results/partB_windows.csv`` (it carries the India VIX level of every
date).  This command copies the working tree to DEST without those files, then regenerates the data
dictionary, figure gallery, README and report there in "no raw data" mode, so every statement in the public
copy matches what the public copy contains.  The working tree is not modified.

DEST must not exist or must be empty, and must not be inside the project.  The copy has no git history:
``git init`` in it and push to a NEW repository (the private repository's history contains the CSV files).
"""
from __future__ import annotations

import fnmatch
import shutil
import subprocess
import sys
from pathlib import Path

from . import config as cfg

# paths (relative to the project root, forward slashes) that never go into the public copy
EXCLUDE_GLOBS = (
    "data/raw/*.csv",                  # NSE / niftyindices source files and the RBI T-bill file (terms of use)
    "data/raw/rbi_xlsx/*",             # the RBI spreadsheets it is built from
    "results/partB_windows.csv",       # holds the India VIX level of every date
    "results/test_run.json",           # rewritten below by a test run of the public copy
    ".git/*", ".venv/*", "venv/*", "env/*", ".pytest_cache/*", ".mypy_cache/*", ".ruff_cache/*",
    "*/__pycache__/*", "__pycache__/*", "*.pyc", "*.egg-info/*", ".ipynb_checkpoints/*", ".DS_Store",
    "explainer_im_7.md",               # private study notes (also git-ignored)
)
REGENERATE = ("data_dictionary", "gallery", "docs")      # "docs" is run a second time after the tests (the README quotes their outcome)


def excluded(rel: str) -> bool:
    rel = rel.lower()
    return any(fnmatch.fnmatch(rel, g.lower()) or fnmatch.fnmatch(rel.split("/")[-1], g.lower()) for g in EXCLUDE_GLOBS)


def files_to_copy(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file() and not excluded(p.relative_to(root).as_posix()))


def make_public(dest, root=None) -> Path:
    root = Path(root or cfg.ROOT).resolve()
    dest = Path(dest).resolve()
    if dest == root or root in dest.parents or dest in root.parents:
        raise ValueError("DEST must be a new folder outside the project (and not a parent of it)")
    if dest.exists() and any(dest.iterdir()):
        raise ValueError(f"{dest} exists and is not empty")
    for src in files_to_copy(root):
        target = dest / src.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
    return dest


def regenerate(dest: Path, stages=REGENERATE) -> None:
    """Rebuild generated documents in the copy (cwd = the copy, so the copy's own src/ and data/ are used)."""
    for stage in stages:
        out = dest / f".stage_{stage}.json"
        code = subprocess.call([sys.executable, "-m", "src.stages", stage, str(out)], cwd=dest)
        out.unlink(missing_ok=True)
        if code:
            raise RuntimeError(f"stage {stage} failed in the public copy (exit code {code})")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    try:
        dest = make_public(argv[0])
    except ValueError as e:
        print(f"error: {e}")
        return 1
    n = len(files_to_copy(dest))
    print(f"copied {n} files to {dest}")
    regenerate(dest)
    print("regenerated data dictionary, figure gallery, README and report in public mode")
    print("running the test suite of the public copy (raw-data tests are skipped) ...", flush=True)
    code = subprocess.call([sys.executable, "-m", "src", "test"], cwd=dest)
    regenerate(dest, ("docs",))          # now quotes the recorded test outcome
    if code:
        print("WARNING: the public copy's test suite failed; do not publish it")
        return code
    print(f"ready: {dest}\nnext: git init, add, commit, and push to a NEW repository (see the instructions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
