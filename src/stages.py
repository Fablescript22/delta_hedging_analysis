"""Run one pipeline stage in its own Python process and record its wall time and peak memory.

    python -m src.stages <stage> <stats.json>

Each stage of ``python -m src reproduce`` runs in a fresh interpreter so large simulation arrays from one
stage can never be retained while the next one runs, and a stall in one stage is attributable to that stage.
Peak memory is the process's resident-set high-water mark (``resource.getrusage``); on Windows, where the
``resource`` module does not exist, it is recorded as null.
"""
from __future__ import annotations

import importlib
import json
import sys
import time

STAGES = {      # name -> (module, what it does)
    "validation": ("src.validation", "validation checklist"),
    "part_a": ("src.run_part_a", "Part A: Monte Carlo experiments A1-A6"),
    "part_b": ("src.run_part_b", "Part B: NIFTY 50 / India VIX backtest and research tables"),
    "gallery": ("src.make_gallery", "figure gallery (docs/FIGURE_GALLERY.md)"),
    "data_dictionary": ("src.data_dictionary", "data dictionary (docs/DATA_DICTIONARY.md)"),
    "docs": ("src.make_readme", "README"),
}


def peak_rss_mb():
    try:
        import resource
    except ImportError:                                  # Windows
        return None
    kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(kb / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)    # macOS reports bytes


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2 or argv[0] not in STAGES:
        print(__doc__)
        return 2
    name, stats_path = argv
    module = importlib.import_module(STAGES[name][0])
    t0 = time.perf_counter()
    code = module.main()
    code = int(code) if isinstance(code, int) else 0
    stats = dict(stage=name, seconds=round(time.perf_counter() - t0, 1), peak_rss_mb=peak_rss_mb(), exit_code=code)
    with open(stats_path, "w") as fh:
        json.dump(stats, fh)
    return code


if __name__ == "__main__":
    sys.exit(main())
