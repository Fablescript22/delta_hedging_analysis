"""Pytest wrappers around the validation checklist (src/validation.py)."""
import pytest

from src import validation as v


def _assert(check):
    assert check.passed, f"[{check.id}] {check.name}: {check.detail}"


def test_1_reference_price():
    _assert(v.check_1_reference_price())


def test_2_put_call_parity():
    _assert(v.check_2_put_call_parity())


def test_3_delta_bounds_and_monotonic():
    _assert(v.check_3_delta_bounds())


def test_4_finite_difference():
    _assert(v.check_4_finite_difference())


def test_5_large_n_hedging_error_near_zero():
    _assert(v.check_5_large_n_error())


def test_6_gbm_moments():
    _assert(v.check_6_gbm_moments())


@pytest.mark.parametrize("i", range(4))
def test_7_real_data(i):
    _assert(v.check_7_real_data()[i])


def test_engine_matches_independent_loop():
    _assert(v.check_extra_engine_vs_loop())


def test_mean_pnl_unbiased_under_q():
    _assert(v.check_extra_unbiased_mean())


def test_simulation_matches_gamma_theory():
    _assert(v.check_extra_gamma_theory())
