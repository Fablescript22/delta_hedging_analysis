import numpy as np
import pytest

from src import bs


def test_textbook_call_and_put_values():
    # S=K=100, r=5%, sigma=20%, T=1  ->  call 10.4506, put 5.5735 (standard reference values)
    assert bs.call_price(100, 100, 1, 0.2, 0.05) == pytest.approx(10.4506, abs=1e-4)
    assert bs.put_price(100, 100, 1, 0.2, 0.05) == pytest.approx(5.5735, abs=1e-4)


def test_atm_delta_and_gamma_reference():
    # d1 = (0.05 + 0.02)/0.2 = 0.35 -> N(0.35) = 0.636831 ; gamma = n(0.35)/(100*0.2) = 0.0187
    assert bs.delta(100, 100, 1, 0.2, 0.05) == pytest.approx(0.636831, abs=1e-6)
    assert bs.gamma(100, 100, 1, 0.2, 0.05) == pytest.approx(0.018762, abs=1e-5)


def test_put_delta_is_call_delta_minus_one():
    S = np.linspace(70, 140, 50)
    assert np.allclose(bs.delta(S, 100, 0.5, 0.3, 0.02, "put"), bs.delta(S, 100, 0.5, 0.3, 0.02) - 1)


def test_gamma_is_largest_near_the_strike_and_grows_toward_expiry():
    S = np.linspace(60, 140, 801)
    g = bs.gamma(S, 100, 0.25, 0.2, 0.05)
    assert abs(S[np.argmax(g)] - 100) < 5
    assert bs.gamma(100, 100, 0.01, 0.2, 0.05) > bs.gamma(100, 100, 0.25, 0.2, 0.05) > bs.gamma(100, 100, 1.0, 0.2, 0.05)


def test_expiry_limits():
    assert bs.call_price(110, 100, 0.0, 0.2, 0.05) == pytest.approx(10.0)
    assert bs.call_price(90, 100, 0.0, 0.2, 0.05) == 0.0
    assert bs.delta(110, 100, 0.0, 0.2, 0.05) == 1.0 and bs.delta(90, 100, 0.0, 0.2, 0.05) == 0.0
    assert bs.gamma(100, 100, 0.0, 0.2, 0.05) == 0.0


def test_broadcasting():
    S = np.array([[90.0], [100.0], [110.0]])
    tau = np.array([0.1, 0.5, 1.0, 2.0])
    assert bs.call_price(S, 100, tau, 0.2, 0.05).shape == (3, 4)


def test_bad_inputs_raise():
    with pytest.raises(ValueError):
        bs.call_price(100, 100, 1.0, 0.0, 0.05)
    with pytest.raises(ValueError):
        bs.call_price(-1, 100, 1.0, 0.2, 0.05)
    with pytest.raises(ValueError):
        bs.price(100, 100, 1.0, 0.2, 0.05, option="straddle")
