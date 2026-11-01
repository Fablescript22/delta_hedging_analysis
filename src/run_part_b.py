"""Regenerate every Part B table and figure from the saved raw data in data/raw/.

    python -m src.run_part_b
"""
from __future__ import annotations

from . import backtest, config as cfg, plots


def main() -> None:
    out = backtest.run_part_b(save=True)
    paths = [plots.fig_b_vix_vs_rv(out), plots.fig_b_regression(out), plots.fig_b_frequency(out),
             plots.fig_b_effect_sizes(out), plots.fig_b_market_states(out), plots.fig_b_estimators(out),
             plots.fig_b_chronological(out), plots.fig_b_frictions(out)]
    print(f"{len(out['ws'].t0)} windows; risk-free {cfg.RISK_FREE_RATE_B:.0%} (placeholder={cfg.RISK_FREE_IS_PLACEHOLDER}). Figures:")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
