"""Black-Scholes pricing, delta and gamma (vectorised, no dividends).

All functions accept scalars or NumPy arrays for S, K, tau, sigma and r and
broadcast them against each other.

Notation
--------
S     spot of the underlying
K     strike
tau   time to maturity in years (T - t).  tau <= 0 means "at expiry"
sigma volatility (annualised, e.g. 0.20 for 20%)
r     continuously compounded risk-free rate

Closed forms
------------
d1 = [ln(S/K) + (r + sigma^2/2) tau] / (sigma sqrt(tau)),   d2 = d1 - sigma sqrt(tau)
Call  = S N(d1) - K e^{-r tau} N(d2)
Put   = K e^{-r tau} N(-d2) - S N(-d1)
Delta (call) = N(d1)         Delta (put) = N(d1) - 1
Gamma        = n(d1) / (S sigma sqrt(tau))        (same for call and put)
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # standard normal CDF, vectorised and fast

_SQRT_2PI = np.sqrt(2.0 * np.pi)


def _norm_pdf(x):
    return np.exp(-0.5 * np.square(x)) / _SQRT_2PI


def _prep(S, K, tau, sigma, r):
    S, K, tau, sigma, r = (np.asarray(a, dtype=float) for a in (S, K, tau, sigma, r))
    for name, a in (("S", S), ("K", K), ("tau", tau), ("sigma", sigma), ("r", r)):
        if not np.all(np.isfinite(a)):          # NaN / inf would otherwise slip through the <= 0 comparisons
            raise ValueError(f"{name} must be finite (no NaN or inf)")
    live = tau > 0
    if np.any(S <= 0) or np.any(K <= 0):
        raise ValueError("S and K must be strictly positive")
    sig_b, live_b = np.broadcast_arrays(sigma, live)
    if np.any((sig_b <= 0) & live_b):
        raise ValueError("sigma must be strictly positive whenever tau > 0")
    tau_safe = np.where(live, tau, 1.0)  # avoid 0-division; overwritten below
    sigma_safe = np.where(sigma > 0, sigma, 1.0)
    return S, K, tau, tau_safe, sigma_safe, r, live


def d1_d2(S, K, tau, sigma, r):
    """Return (d1, d2).  Values at tau <= 0 are meaningless and should not be used."""
    S, K, tau, tau_safe, sigma, r, _ = _prep(S, K, tau, sigma, r)
    vol_sqrt_t = sigma * np.sqrt(tau_safe)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau_safe) / vol_sqrt_t
    return d1, d1 - vol_sqrt_t


def call_price(S, K, tau, sigma, r):
    S, K, tau, tau_safe, sigma_s, r, live = _prep(S, K, tau, sigma, r)
    d1, d2 = d1_d2(S, K, tau, sigma, r)
    price = S * ndtr(d1) - K * np.exp(-r * tau_safe) * ndtr(d2)
    return np.where(live, price, np.maximum(S - K, 0.0))


def put_price(S, K, tau, sigma, r):
    S, K, tau, tau_safe, sigma_s, r, live = _prep(S, K, tau, sigma, r)
    d1, d2 = d1_d2(S, K, tau, sigma, r)
    price = K * np.exp(-r * tau_safe) * ndtr(-d2) - S * ndtr(-d1)
    return np.where(live, price, np.maximum(K - S, 0.0))


def price(S, K, tau, sigma, r, option: str = "call"):
    if option == "call":
        return call_price(S, K, tau, sigma, r)
    if option == "put":
        return put_price(S, K, tau, sigma, r)
    raise ValueError("option must be 'call' or 'put'")


def delta(S, K, tau, sigma, r, option: str = "call"):
    """Delta dV/dS.  At expiry (tau <= 0) returns the limiting step function."""
    S, K, tau, tau_safe, sigma_s, r, live = _prep(S, K, tau, sigma, r)
    d1, _ = d1_d2(S, K, tau, sigma, r)
    if option == "call":
        d = ndtr(d1)
        return np.where(live, d, (S > K).astype(float))
    if option == "put":
        d = ndtr(d1) - 1.0
        return np.where(live, d, -(S < K).astype(float))
    raise ValueError("option must be 'call' or 'put'")


def gamma(S, K, tau, sigma, r):
    """Gamma d2V/dS2 (identical for calls and puts).  Zero at expiry by convention."""
    S, K, tau, tau_safe, sigma_s, r, live = _prep(S, K, tau, sigma, r)
    d1, _ = d1_d2(S, K, tau, sigma, r)
    g = _norm_pdf(d1) / (S * sigma_s * np.sqrt(tau_safe))
    return np.where(live, g, 0.0)
