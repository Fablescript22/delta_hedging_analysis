"""One entry point:  python -m src <command>

    python -m src reproduce              # integrity check -> validation -> Part A -> Part B -> data dictionary -> README + gallery
    python -m src test                   # run the test suite and record the outcome
    python -m src check-data             # verify the raw files against data/raw/MANIFEST.json
    python -m src snapshot DIR           # copy results/, docs/ and README.md into DIR (a baseline to compare against)
    python -m src verify-outputs DIR     # fail if the current results/docs/README differ from the snapshot in DIR
    python -m src make-public DIR        # copy the project to DIR without the NSE raw files, for a public repository

``reproduce`` runs every stage in its own Python process (so large simulation arrays are freed between
stages and a slow stage is easy to identify) and writes per-stage wall time and peak memory to
``results/reproduce_stats.json``.
"""
from __future__ import annotations

import json
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from . import config as cfg
from .stages import STAGES

COMMANDS = ("reproduce", "test", "check-data", "snapshot", "verify-outputs", "make-public")
STATS_FILE = cfg.RESULTS_DIR / "reproduce_stats.json"
TEST_FILE = cfg.RESULTS_DIR / "test_run.json"


def _run_stage(name: str, tmp: str):
    print(f"\n=== stage: {name} ({STAGES[name][1]}) ===", flush=True)
    out = Path(tmp) / f"{name}.json"
    code = subprocess.call([sys.executable, "-m", "src.stages", name, str(out)], cwd=cfg.ROOT)
    return code, (json.loads(out.read_text()) if out.exists() else None)


def reproduce() -> int:
    from . import integrity
    if integrity.main([]) != 0:
        return 1
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    stats, worst = [], 0
    t_all = time.perf_counter()
    with tempfile.TemporaryDirectory() as tmp:
        for name in (n for n in STAGES if n != "docs"):
            code, st = _run_stage(name, tmp)
            if st is None:
                print(f"stage {name} did not finish (exit code {code})")
                return code or 1
            stats.append(st)
            worst = worst or code          # the validation stage exits non-zero if a check fails: keep going, report at the end
        total = round(time.perf_counter() - t_all, 1)
        # written BEFORE the README is built, because the README quotes these timings (inside a <!--volatile--> region)
        STATS_FILE.write_text(json.dumps(dict(total_seconds=total, python=platform.python_version(),
                                              platform=platform.platform(), stages=stats), indent=2))
        code, st = _run_stage("docs", tmp)
    print("\nstage             seconds   peak MB")
    for s in stats + ([st] if st else []):
        print(f"{s['stage']:<16}{s['seconds']:>9}   {s['peak_rss_mb']}")
    print(f"computation total{total:>8}")
    return worst or code


def run_tests() -> int:
    proc = subprocess.Popen([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"], cwd=cfg.ROOT,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    code = proc.wait()
    m = dict((w, int(n)) for n, w in re.findall(r"(\d+) (passed|failed|errors?|skipped)", "".join(lines[-3:])))
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    TEST_FILE.write_text(json.dumps(dict(
        passed=m.get("passed", 0), failed=m.get("failed", 0), skipped=m.get("skipped", 0), errors=m.get("error", 0) + m.get("errors", 0),
        python=platform.python_version(), exit_code=code), indent=2))
    return code


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__)
        return 2
    cmd, rest = argv[0], argv[1:]
    if cmd == "make-public":
        from . import make_public
        return make_public.main(rest)
    if cmd in ("snapshot", "verify-outputs"):
        if len(rest) != 1:
            print(__doc__)
            return 2
        from . import verify_outputs
        if cmd == "snapshot":
            print(f"snapshot written to {verify_outputs.snapshot(rest[0])}")
            return 0
        return verify_outputs.main(rest)
    if rest:
        print(__doc__)
        return 2
    if cmd == "check-data":
        from . import integrity
        return integrity.main([])
    if cmd == "test":
        return run_tests()
    return reproduce()


if __name__ == "__main__":
    sys.exit(main())
