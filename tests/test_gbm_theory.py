import numpy as np
import pytest

from src import bs, config as cfg
from src import gbm_sim as g


def test_closed_form_matches_gauss_hermite_at_moderate_tau():
    t, sr = 0.5, 0.3
    z, w = np.polynomial.hermite_e.hermegauss(100)
    w = w / np.sqrt(2 * np.pi)
    S = 100 * np.exp((0.05 - 0.5 * sr**2) * t + sr * np.sqrt(t) * z)
    gh = (bs.gamma(S, 100, 1 - t, 0.2, 0.05) * S**2 * w).sum()
    cf = g.expected_gamma_s2(t, sr, 0.2, mu=0.05, S0=100, K=100, T=1, r=0.05)
    assert cf == pytest.approx(gh, rel=1e-6)


def test_theory_is_zero_at_equal_vol_and_sign_follows_gap():
    assert g.theory_mean_pnl(0.2) == pytest.approx(0.0, abs=1e-12)
    assert g.theory_mean_pnl(0.15) > 0 > g.theory_mean_pnl(0.25)


def test_theory_converged_in_nodes():
    assert g.theory_mean_pnl(0.3, n_u=500) == pytest.approx(g.theory_mean_pnl(0.3, n_u=4000), rel=1e-9)


def test_simulation_reproducible_with_seed():
    a, _ = g.simulate_and_hedge(500, 20, sigma_r=0.2, rng=g.make_rng(1, 2))
    b, _ = g.simulate_and_hedge(500, 20, sigma_r=0.2, rng=g.make_rng(1, 2))
    assert np.array_equal(a["pnl"], b["pnl"])
