import sys
from pathlib import Path

import pytest

# make `import src...` work when pytest is run from anywhere
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config as cfg  # noqa: E402

# The public repository ships without the NSE raw CSV files (their terms restrict redistribution).  Tests that need
# those files are SKIPPED with a visible reason when they are absent; nothing else is skipped, and in the full
# working copy (files present) every test runs.
RAW_REASON = "raw NIFTY 50 / India VIX files not present in data/raw/ (see data/raw/SOURCES.md)"
RAW_MODULES = {"test_real_data.py", "test_calendar_time.py", "test_independent_reproduction.py"}
RAW_FIXTURES = {"real", "part_b", "data"}            # fixtures that load the raw data or run Part B on it
RAW_TESTS = {"test_shipped_raw_data_matches_manifest", "test_integrity_detects_modified_file",
             "test_integrity_detects_missing_and_unexpected_files", "test_entry_point_rejects_unknown_commands",
             "test_7_real_data"}


def pytest_collection_modifyitems(config, items):
    if cfg.raw_data_available():
        return
    skip = pytest.mark.skip(reason=RAW_REASON)
    for item in items:
        base = item.name.split("[")[0]
        if (item.path.name in RAW_MODULES or base in RAW_TESTS
                or RAW_FIXTURES & set(getattr(item, "fixturenames", ()))):
            item.add_marker(skip)
