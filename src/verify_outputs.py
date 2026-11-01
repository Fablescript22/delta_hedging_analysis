"""Check that regenerated outputs match a snapshot of the committed ones.

    python -m src snapshot DIR              # copy results/, docs/ and README.md into DIR
    python -m src verify-outputs DIR        # compare the current files with that snapshot

Numeric table cells and JSON numbers must agree to a relative tolerance (default 1e-6, so a different
BLAS build cannot fail the check, but a changed result does); text must match exactly.  Regions of the
README and of the docs between ``<!--volatile-->`` and ``<!--/volatile-->`` (wall-clock times, the test-run summary) are
ignored, as is ``results/reproduce_stats.json`` and ``results/test_run.json`` (run metadata).  Figures are checked for existence only, since PNG
bytes depend on the matplotlib build.  The CI job snapshots the committed files, regenerates everything and
fails if anything differs, so a stale README, table or document cannot be committed unnoticed.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

IGNORED = {"reproduce_stats.json", "test_run.json"}
VOLATILE = re.compile(r"<!--volatile-->.*?<!--/volatile-->", re.S)
RTOL, ATOL = 1e-6, 1e-9


def mask_volatile(text: str) -> str:
    return VOLATILE.sub("<!--volatile-->", text)


def _close(a, b, rtol, atol) -> bool:
    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if math.isnan(a) and math.isnan(b):
            return True
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return a == b


def compare_json(a, b, rtol=RTOL, atol=ATOL, path="$") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}: key sets differ ({sorted(set(a) ^ set(b))})"] if set(a) != set(b) else []
        for k in sorted(set(a) & set(b)):
            out += compare_json(a[k], b[k], rtol, atol, f"{path}.{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} vs {len(b)}"]
        return [m for i, (x, y) in enumerate(zip(a, b)) for m in compare_json(x, y, rtol, atol, f"{path}[{i}]")]
    return [] if _close(a, b, rtol, atol) else [f"{path}: {a!r} vs {b!r}"]


def compare_csv(pa: Path, pb: Path, rtol=RTOL, atol=ATOL) -> list[str]:
    a, b = pd.read_csv(pa), pd.read_csv(pb)
    if list(a.columns) != list(b.columns):
        return [f"{pa.name}: columns differ"]
    if a.shape != b.shape:
        return [f"{pa.name}: shape {a.shape} vs {b.shape}"]
    msgs = []
    for c in a.columns:
        if pd.api.types.is_numeric_dtype(a[c]) and pd.api.types.is_numeric_dtype(b[c]):
            x, y = a[c].to_numpy(float), b[c].to_numpy(float)
            same_nan = np.isnan(x) == np.isnan(y)
            ok = same_nan & (np.isnan(x) | np.isclose(x, y, rtol=rtol, atol=atol))
            if not ok.all():
                i = int(np.flatnonzero(~ok)[0])
                msgs.append(f"{pa.name}[{c}] row {i}: {x[i]!r} vs {y[i]!r} ({int((~ok).sum())} cells differ)")
        elif not a[c].astype(str).equals(b[c].astype(str)):
            msgs.append(f"{pa.name}[{c}]: text differs")
    return msgs


def snapshot(dest) -> Path:
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    shutil.copytree(cfg.RESULTS_DIR, dest / "results")
    shutil.copytree(cfg.ROOT / "docs", dest / "docs")
    shutil.copy(cfg.ROOT / "README.md", dest / "README.md")
    (dest / "figures.txt").write_text("\n".join(sorted(p.name for p in cfg.FIG_DIR.glob("*.png"))) + "\n")
    return dest


def verify(base, current_root=None, rtol=RTOL, atol=ATOL) -> list[str]:
    """Return a list of human-readable differences (empty = outputs match the snapshot)."""
    base, root = Path(base), Path(current_root or cfg.ROOT)
    problems: list[str] = []
    for sub in ("results", "docs"):
        old_files = {p.name for p in (base / sub).iterdir() if p.is_file()} - IGNORED
        new_files = {p.name for p in (root / sub).iterdir() if p.is_file()} - IGNORED
        for n in sorted(old_files - new_files):
            problems.append(f"{sub}/{n}: present in snapshot, missing now")
        for n in sorted(new_files - old_files):
            problems.append(f"{sub}/{n}: new file not in snapshot")
        for n in sorted(old_files & new_files):
            pa, pb = base / sub / n, root / sub / n
            if n.endswith(".csv"):
                problems += compare_csv(pa, pb, rtol, atol)
            elif n.endswith(".json"):
                problems += compare_json(json.loads(pa.read_text()), json.loads(pb.read_text()), rtol, atol, n)
            elif mask_volatile(pa.read_text()) != mask_volatile(pb.read_text()):
                problems.append(f"{sub}/{n}: text differs (outside <!--volatile--> regions)")
    if mask_volatile((base / "README.md").read_text()) != mask_volatile((root / "README.md").read_text()):
        problems.append("README.md: differs (outside <!--volatile--> regions)")
    old_figs = set((base / "figures.txt").read_text().split())
    new_figs = {p.name for p in (root / "figures").glob("*.png")}
    if old_figs - new_figs:
        problems.append(f"figures missing now: {sorted(old_figs - new_figs)}")
    return problems


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    problems = verify(argv[0])
    if problems:
        print(f"{len(problems)} difference(s) from the snapshot in {argv[0]}:")
        for p in problems[:50]:
            print("  -", p)
        return 1
    print(f"outputs match the snapshot in {argv[0]} (numeric rtol {RTOL}, text exact).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
