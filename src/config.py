"""Central configuration.  Every tunable number used by the experiments lives here.

Nothing in this file is a result; it only holds inputs and labelled assumptions.
"""
from __future__ import annotations

from pathlib import Path

# ------------------------------------------------------------------ paths
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"


def raw_data_available() -> bool:
    """True if any raw CSV is present in data/raw/.  The public repository ships without them (see data/raw/SOURCES.md)."""
    return any(RAW_DIR.glob("*.csv"))

# ------------------------------------------------------------------ randomness
SEED = 42  # fixed seed for every Part A experiment (each experiment spawns its own stream)

# ------------------------------------------------------------------ Part A (synthetic GBM)
A_S0 = 100.0
A_K = 100.0
A_T = 1.0                       # years
A_R = 0.05                      # synthetic-world risk-free rate (also the checklist's r = 5%)
A_SIGMA_I = 0.20                # implied vol at which the option is sold
A_MU = 0.05                     # real-world drift of the simulated stock (default mu = r)
A_N_PATHS = 20_000              # brief requires >= 10,000 per setting
A_N_GRID = (5, 10, 20, 50, 100, 250, 500)

# ------------------------------------------------------------------ Part B (real data)
TRADING_DAYS_PER_YEAR = 252     # used to annualise daily returns and to set dt
WINDOW_TRADING_DAYS = 21        # ~30 calendar days, the horizon India VIX is quoted for

# ---- Risk-free rate for Part B ------------------------------------------------
# Baseline: the RBI 91-day Treasury-bill auction cut-off yield (data/raw/RBI_91day_tbill_auctions.csv), looked up as of
# each window's start date (no look-ahead) and converted to a continuously compounded rate; see src/rates.py.
# RISK_FREE_RATE_B is NOT the baseline any more.  It is the single constant that earlier versions of this project
# assumed (6%, an unsourced placeholder), kept as one of the sensitivity cases and as the fallback for synthetic test
# fixtures that carry no rate of their own.
RISK_FREE_RATE_B = 0.06
RISK_FREE_SOURCE_B = ("RBI 91-day T-bill auction cut-off yield (Handbook of Statistics on the Indian Economy), "
                      "latest auction on or before each window's start, continuously compounded")
RISK_FREE_IS_PLACEHOLDER = False
RISK_FREE_SENSITIVITY_B = (0.04, 0.06, 0.08)   # constant-rate sensitivity cases, compared with the T-bill baseline

# ---- Data-cleaning decisions (see data/raw/SOURCES.md and README) -------------
VIX_PLAUSIBLE_RANGE_PCT = (5.0, 100.0)   # India VIX is quoted in percent; outside this = data error
DROP_WEEKEND_SESSIONS = False            # keep Muhurat / budget-day Saturday-Sunday sessions by default
