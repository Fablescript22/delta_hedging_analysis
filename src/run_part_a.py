"""Regenerate every Part A table and figure from the fixed seed.

    python -m src.run_part_a
"""
from __future__ import annotations

from . import config as cfg
from . import gbm_sim, plots


def main() -> None:
    res = gbm_sim.run_part_a(save=True)
    premium = float(res["a1"]["premium"].iloc[0])
    paths = [
        plots.fig_a2_scaling(res["a2"], res["a2_fits"]),
        plots.fig_a2_distributions(res["a2_samples"], premium),
        plots.fig_a3_volgap(res["a3"], res["a3_summary"], res["a3_scatter"]),
        plots.fig_a4_costs(res["a4"], res["a4_optima"]),
    ]
    print(f"seed={cfg.SEED}; wrote tables to {cfg.RESULTS_DIR} and figures:")
    for p in paths:
        print("  ", p)


if __name__ == "__main__":
    main()
