"""Input-file integrity: a checked manifest of the raw data.

``data/raw/MANIFEST.json`` records, for each raw CSV, its SHA-256 hash, size, number of data rows and first/last
date.  ``verify()`` recomputes all of these and fails loudly if a file is missing, changed, or if an unexpected
CSV appeared in the folder (the loaders pick files up with a glob, so a stray file would otherwise be included
silently).

    python -m src.integrity            # verify (exit code 1 on any problem)
    python -m src.integrity --write    # regenerate the manifest after deliberately changing the raw data

The manifest proves that the files are the ones the results were produced from; it does not prove that the files are
correct (see data/raw/SOURCES.md for provenance and known problems).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from . import config as cfg

MANIFEST = cfg.RAW_DIR / "MANIFEST.json"
DATE_FORMATS = {"NIFTY": "%d %b %Y", "hist_india_vix": "%d-%b-%Y", "RBI_91day_tbill": "%Y-%m-%d"}


class IntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _describe(path: Path) -> dict:
    fmt = next(f for k, f in DATE_FORMATS.items() if path.name.startswith(k))
    with open(path, newline="", encoding="utf-8-sig") as fh:
        rows = [{k.strip(): v for k, v in r.items()} for r in csv.DictReader(fh)]   # VIX headers have trailing spaces
    dates = [datetime.strptime(r["Date"].strip(), fmt).date() for r in rows]
    return dict(file=path.name, bytes=path.stat().st_size, sha256=_sha256(path), rows=len(rows),
                first_date=min(dates).isoformat(), last_date=max(dates).isoformat())


def raw_files(raw_dir: Path = cfg.RAW_DIR) -> list[Path]:
    return sorted(p for p in Path(raw_dir).glob("*.csv"))


def build_manifest(raw_dir: Path = cfg.RAW_DIR) -> dict:
    files = [_describe(p) for p in raw_files(raw_dir)]
    if not files:
        raise IntegrityError(f"no CSV files in {raw_dir}")
    return dict(note="SHA-256 manifest of the untouched raw data files; regenerate with `python -m src.integrity --write`.",
                files=files)


def verify(raw_dir: Path = cfg.RAW_DIR, manifest: Path = MANIFEST) -> list[str]:
    """Return a list of problems (empty = all good)."""
    if not Path(manifest).exists():
        return [f"manifest not found: {manifest} (create it with `python -m src.integrity --write`)"]
    expected = {f["file"]: f for f in json.loads(Path(manifest).read_text())["files"]}
    found = {p.name: p for p in raw_files(raw_dir)}
    if not found:
        return [f"no raw data found in {raw_dir}. The NIFTY 50 and India VIX CSV files are not part of the public repository "
                "(their publishers' terms restrict redistribution), and neither is the RBI T-bill file derived from the RBI "
                "spreadsheets: obtain them as described in data/raw/SOURCES.md, put them in data/raw/, then rerun. `python -m src check-data` tells you whether they match the manifest."]
    problems = []
    for name in sorted(set(expected) - set(found)):
        problems.append(f"missing file: {name}")
    for name in sorted(set(found) - set(expected)):
        problems.append(f"unexpected file not in the manifest: {name}")
    for name in sorted(set(expected) & set(found)):
        exp, p = expected[name], found[name]
        if _sha256(p) != exp["sha256"]:
            got = _describe(p)
            detail = f"{got['rows']} rows (expected {exp['rows']}), {got['bytes']} bytes (expected {exp['bytes']})"
            problems.append(f"changed file: {name} (SHA-256 differs; {detail})")
    return problems


def assert_ok(raw_dir: Path = cfg.RAW_DIR) -> None:
    problems = verify(raw_dir)
    if problems:
        raise IntegrityError("raw-data integrity check failed:\n  - " + "\n  - ".join(problems))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--write", action="store_true", help="regenerate MANIFEST.json from the files now on disk")
    args = ap.parse_args(argv)
    if args.write:
        MANIFEST.write_text(json.dumps(build_manifest(), indent=2) + "\n")
        print(f"wrote {MANIFEST} ({len(raw_files())} files)")
        return 0
    problems = verify()
    if problems:
        print("INTEGRITY CHECK FAILED")
        for p in problems:
            print("  -", p)
        return 1
    n = len(json.loads(MANIFEST.read_text())["files"])
    print(f"raw-data integrity OK: {n} files match data/raw/MANIFEST.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
