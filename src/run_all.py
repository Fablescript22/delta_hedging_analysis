"""Regenerate EVERYTHING: validation checklist, Part A (fixed seed) and Part B (saved raw data).

    python -m src.run_all
"""
from __future__ import annotations

import sys

from . import run_part_a, run_part_b, validation


def main() -> int:
    code = validation.main()
    run_part_a.main()
    run_part_b.main()
    return code


if __name__ == "__main__":
    sys.exit(main())
