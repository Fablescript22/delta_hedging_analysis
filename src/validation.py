"""The validation checklist from the brief, as executable code.

Run:  python -m src.validation        (prints a table, writes results/validation_report.md)
Tests in tests/ call the same functions and assert ``passed``.

All tolerances below were fixed BEFORE the checks were run.  If a check fails the
fix belongs in the code under test, not in the tolerance.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from . import bs, config as cfg, real_data
from .gbm_sim import make_rng, simulate_and_hedge, simulate_gbm_paths
from .hedge import hedge_pnl


@dataclass
class Check:
    id: str
    name: str
    passed: bool
    detail: str


# ------------------------------------------------------------------ 1. reference price
def check_1_reference_price() -> Check:
    p = float(bs.call_price(100.0, 100.0, 1.0, 0.20, 0.05))
    tol = 0.005                                   # "about 10.45"
    return Check("1", "BS call price S=K=100, r=5%, sigma=20%, T=1 is ~10.45",
                 abs(p - 10.45) < tol, f"price = {p:.6f} (|price-10.45| < {tol})")


# ------------------------------------------------------------------ 2. put-call parity
def check_2_put_call_parity() -> Check:
    rng = np.random.default_rng(1)
    n = 5000
    S = rng.uniform(20, 300, n); K = rng.uniform(20, 300, n)
    tau = rng.uniform(0.01, 3.0, n); sig = rng.uniform(0.05, 0.9, n); r = rng.uniform(-0.01, 0.12, n)
    lhs = bs.call_price(S, K, tau, sig, r) - bs.put_price(S, K, tau, sig, r)
    rhs = S - K * np.exp(-r * tau)
    err = float(np.max(np.abs(lhs - rhs)))
    tol = 1e-10
    return Check("2", "Put-call parity C - P = S - K e^{-r tau}", err < tol,
                 f"max |error| over {n} random inputs = {err:.2e} (tol {tol:g})")


# ------------------------------------------------------------------ 3. delta bounds & monotonicity
def check_3_delta_bounds() -> Check:
    """Delta in (0,1) and increasing in S.

    NOTE (found on the first run, see README "validation log"): a float64 number cannot sit
    strictly inside (0,1) once |d1| is beyond ~8 (e.g. N(49) rounds to exactly 1.0).  The
    mathematical property is therefore tested in two parts:
      (a) on the WHOLE grid:  0 <= delta <= 1 and delta never decreases as S rises (0 violations);
      (b) wherever |d1| < 6 (delta representably inside (0,1)): STRICTLY inside (0,1) and STRICTLY increasing.
    """
    S = np.linspace(50, 200, 3001)
    viol_bounds = viol_mono = viol_strict = n_core = n_sat = 0
    for tau in (0.02, 0.1, 0.25, 1.0, 2.0):
        for sig in (0.1, 0.2, 0.5):
            d = bs.delta(S, 100.0, tau, sig, 0.05)
            d1, _ = bs.d1_d2(S, 100.0, tau, sig, 0.05)
            viol_bounds += int(np.sum((d < 0) | (d > 1)))
            viol_mono += int(np.sum(np.diff(d) < 0))
            core = np.abs(d1) < 6
            n_core += int(core.sum())
            n_sat += int(np.sum((d <= 0) | (d >= 1)))
            dc = d[core]
            viol_strict += int(np.sum((dc <= 0) | (dc >= 1))) + int(np.sum(np.diff(dc) <= 0))
    ok = viol_bounds == 0 and viol_mono == 0 and viol_strict == 0
    return Check("3", "Call delta in (0,1) and increasing in S", ok,
                 f"(a) full grid: {viol_bounds} bound violations, {viol_mono} decreasing steps; "
                 f"(b) |d1|<6 region ({n_core} pts): {viol_strict} strictness violations; "
                 f"{n_sat} grid points saturate to exactly 0/1 in float64 (|d1| large)")


# ------------------------------------------------------------------ 4. finite differences
def check_4_finite_difference() -> Check:
    rng = np.random.default_rng(4)
    n = 2000
    S = rng.uniform(60, 160, n); K = 100.0
    tau = rng.uniform(0.05, 2.0, n); sig = rng.uniform(0.1, 0.6, n); r = 0.05
    h = 1e-4 * S
    fd_delta = (bs.call_price(S + h, K, tau, sig, r) - bs.call_price(S - h, K, tau, sig, r)) / (2 * h)
    err_d = float(np.max(np.abs(fd_delta - bs.delta(S, K, tau, sig, r))))
    fd_gamma = (bs.delta(S + h, K, tau, sig, r) - bs.delta(S - h, K, tau, sig, r)) / (2 * h)
    err_g = float(np.max(np.abs(fd_gamma - bs.gamma(S, K, tau, sig, r))))
    tol_d, tol_g = 1e-6, 1e-6
    return Check("4", "Finite-difference dPrice/dS matches delta (and dDelta/dS matches gamma)",
                 err_d < tol_d and err_g < tol_g,
                 f"max |FD delta - delta| = {err_d:.2e} (tol {tol_d:g}); max |FD gamma - gamma| = {err_g:.2e} (tol {tol_g:g})")


# ------------------------------------------------------------------ 5. large-N hedging error ~ 0
def check_5_large_n_error(n_steps: int = 4000, n_paths: int = 10_000) -> Check:
    """sigma_r = sigma_i, N very large: mean PnL ~ 0 (within 3 SE) and std small vs the premium."""
    rng = make_rng(900)
    out, _ = simulate_and_hedge(n_paths, n_steps, sigma_r=cfg.A_SIGMA_I, rng=rng)
    pnl, prem = out["pnl"], float(out["premium"][0])
    mean, std = pnl.mean(), pnl.std(ddof=1)
    se = std / np.sqrt(n_paths)
    std_cap = 0.05 * prem                         # fixed a priori: std below 5% of the premium
    ok = abs(mean) < 3 * se and std < std_cap
    return Check("5", f"Part A with very large N (N={n_steps}): hedging error close to 0", ok,
                 f"mean PnL = {mean:+.5f} (SE {se:.5f}; |mean| < 3 SE required); std = {std:.4f} "
                 f"vs cap {std_cap:.4f} = 5% of premium {prem:.4f}")


# ------------------------------------------------------------------ 6. GBM moments
def check_6_gbm_moments(n_paths: int = 400_000, n_steps: int = 12) -> Check:
    mu, sig, T, S0 = 0.08, 0.25, 1.0, 100.0
    rng = make_rng(901)
    P = simulate_gbm_paths(n_paths, n_steps, S0, mu, sig, T, rng)
    L = np.log(P[:, -1] / S0)                      # terminal log-return over T
    m_exp, v_exp = (mu - 0.5 * sig**2) * T, sig**2 * T
    m_hat, v_hat = L.mean(), L.var(ddof=1)
    se_m = np.sqrt(v_exp / n_paths)
    tol_v = 4 * np.sqrt(2.0 / (n_paths - 1))       # 4 standard errors of a normal variance estimate
    ok_m = abs(m_hat - m_exp) < 4 * se_m
    ok_v = abs(v_hat / v_exp - 1) < tol_v
    # step-level too: every single increment should have mean (mu-s^2/2)dt, var s^2 dt
    inc = np.diff(np.log(P), axis=1)
    dt = T / n_steps
    ok_step = abs(inc.mean() - (mu - 0.5 * sig**2) * dt) < 4 * np.sqrt(sig**2 * dt / inc.size) and \
        abs(inc.var(ddof=1) / (sig**2 * dt) - 1) < 4 * np.sqrt(2.0 / (inc.size - 1))
    return Check("6", "Simulated GBM log-returns have the input mean and variance",
                 bool(ok_m and ok_v and ok_step),
                 f"terminal mean {m_hat:+.5f} vs {m_exp:+.5f} (4 SE = {4*se_m:.5f}); "
                 f"variance ratio {v_hat/v_exp:.4f} (tol +/-{tol_v:.4f}); per-step moments ok={bool(ok_step)}")


# ------------------------------------------------------------------ 7. real data
def check_7_real_data() -> list[Check]:
    data = real_data.load_all()
    res = real_data.validate_aligned(data.aligned)
    return [Check("7" + "abcd"[i], f"Real data: {name}", ok, detail) for i, (name, ok, detail) in enumerate(res)]


# ------------------------------------------------------------------ extra: engine vs an independent loop
def check_extra_engine_vs_loop() -> Check:
    """Independent step-by-step implementation (explicit cash account) must equal hedge_pnl."""
    rng = make_rng(902)
    P = simulate_gbm_paths(50, 37, 100.0, 0.05, 0.3, 0.5, rng)
    K, sig, r, T = 105.0, 0.25, 0.04, 0.5
    res = hedge_pnl(P, K, sig, r, T)
    n = P.shape[1] - 1
    dt = T / n
    ref = np.empty(P.shape[0])
    for m in range(P.shape[0]):
        S = P[m]
        C0 = float(bs.call_price(S[0], K, T, sig, r))
        d = float(bs.delta(S[0], K, T, sig, r))
        cash = C0 - d * S[0]
        for i in range(1, n):
            cash *= np.exp(r * dt)
            d_new = float(bs.delta(S[i], K, T - i * dt, sig, r))
            cash -= (d_new - d) * S[i]
            d = d_new
        cash *= np.exp(r * dt)
        ref[m] = cash + d * S[-1] - max(S[-1] - K, 0.0)
    err = float(np.max(np.abs(ref - res.pnl)))
    return Check("E1", "Hedging engine equals an independent explicit cash-account loop", err < 1e-9,
                 f"max |difference| over 50 paths = {err:.2e} (tol 1e-9)")


# ------------------------------------------------------------------ extra: hedging PnL is unbiased under Q
def check_extra_unbiased_mean(n_rep: int = 100, n_paths: int = 20_000, n_steps: int = 20) -> Check:
    """sigma_r = sigma_i and mu = r  =>  E[PnL] = 0 EXACTLY for any N (discounted hedge portfolio is a martingale).

    One experiment is just one z-score, so repeat it ``n_rep`` times with independent seeds.  The z-scores must
    look like draws from N(0,1): |mean z| < 3/sqrt(n_rep) and 0.8 < std z < 1.25 (fixed a priori).
    """
    zs = []
    for k in range(n_rep):
        out, _ = simulate_and_hedge(n_paths, n_steps, sigma_r=cfg.A_SIGMA_I, mu=cfg.A_R, rng=make_rng(777, k))
        p = out["pnl"]
        zs.append(p.mean() / (p.std(ddof=1) / np.sqrt(len(p))))
    zs = np.array(zs)
    ok = abs(zs.mean()) < 3 / np.sqrt(n_rep) and 0.8 < zs.std(ddof=1) < 1.25
    return Check("E2", f"Mean hedging PnL is unbiased when sigma_r = sigma_i and mu = r (N={n_steps})", bool(ok),
                 f"{n_rep} independent replicates of {n_paths} paths: mean z = {zs.mean():+.3f} "
                 f"(|.| < {3/np.sqrt(n_rep):.2f}), std z = {zs.std(ddof=1):.3f} (0.8-1.25), max |z| = {np.abs(zs).max():.2f}")


# ------------------------------------------------------------------ extra: simulation vs gamma-formula theory
def check_extra_gamma_theory(n_paths: int = 100_000, n_steps: int = 250) -> Check:
    """Mean simulated PnL must match the continuous-time gamma formula (closed-form expectation)."""
    from .gbm_sim import theory_mean_pnl
    parts, ok = [], True
    for sr in (0.10, 0.30):
        out, _ = simulate_and_hedge(n_paths, n_steps, sigma_r=sr, mu=cfg.A_MU, rng=make_rng(778, int(sr * 100)))
        p = out["pnl"]
        th, se = theory_mean_pnl(sr), p.std(ddof=1) / np.sqrt(n_paths)
        ok &= abs(p.mean() - th) < 4 * se
        parts.append(f"sigma_r={sr}: sim {p.mean():+.4f} vs theory {th:+.4f} ({(p.mean()-th)/se:+.2f} SE)")
    return Check("E3", "Simulated mean PnL matches the gamma formula (tol 4 SE)", bool(ok), "; ".join(parts))


def run_all() -> list[Check]:
    checks = [check_1_reference_price(), check_2_put_call_parity(), check_3_delta_bounds(),
              check_4_finite_difference(), check_5_large_n_error(), check_6_gbm_moments()]
    checks += check_7_real_data()
    checks += [check_extra_engine_vs_loop(), check_extra_unbiased_mean(), check_extra_gamma_theory()]
    return checks


def format_table(checks: list[Check]) -> str:
    lines = ["| # | Check | Result | Detail |", "|---|---|---|---|"]
    for c in checks:
        lines.append(f"| {c.id} | {c.name} | {'PASS' if c.passed else 'FAIL'} | {c.detail} |")
    return "\n".join(lines)


def main() -> int:
    checks = run_all()
    table = format_table(checks)
    print(table)
    cfg.RESULTS_DIR.mkdir(exist_ok=True)
    (cfg.RESULTS_DIR / "validation_report.md").write_text(
        "# Validation report (auto-generated by `python -m src.validation`)\n\n" + table + "\n")
    failed = [c.id for c in checks if not c.passed]
    print("\nFAILED:" if failed else "\nAll checks passed.", failed if failed else "")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
