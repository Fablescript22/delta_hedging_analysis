# EXPLAINER: the project in plain English

This file walks through what the code does, why the main results come out the way they do, and how to answer the standard questions about them. Every number quoted here is taken from the run results in `results/` (`docs/RESEARCH_REPORT.md` has the full tables and the README is the short front page). Nothing here is a sourced fact about markets; the one assumed input is the 6% risk-free rate in Part B, which is a placeholder.

---

## 1. The idea in one paragraph

An option seller sells a call option and then buys or sells the underlying (here, an index) so that small up and down moves in the index do not matter. The amount to hold is the option's **delta**. In theory you would adjust the holding at every instant. In real life you adjust at fixed times (say once a day), so a small error is left over each time. This project measures that leftover error (Part A, on simulated prices) and then checks whether the same story holds on real NIFTY 50 data, using India VIX as the market's "price of volatility" (Part B).

## 2. A tour of the code

Read the files in this order.

**`src/bs.py`** is the Black-Scholes formulas: call price, put price, delta and gamma. Everything is vectorised, meaning one call prices thousands of paths at once. It handles the edge case of zero time left (the price is then just the payoff) and refuses nonsense inputs such as negative volatility, NaN or infinite values. The hedging engine and the simulator do the same for cost rates, simulation sizes, time grids and strikes, so a typo fails loudly instead of producing a plausible-looking wrong number.

**`src/hedge.py`** is the hedging engine, the heart of the project. Given a set of price paths, a strike, a volatility, a rate and the rebalance times, it:

1. sells the option at its Black-Scholes price and puts the premium in a cash account;
2. at each rebalance, computes delta, buys or sells the underlying to hold exactly that amount, and funds the trade from cash;
3. lets cash grow at the risk-free rate between rebalances;
4. at maturity, adds up the cash plus the value of the held underlying, then subtracts what the option owes. That final number is the **hedging PnL**.

It can also charge proportional transaction costs, hold the last hedge to maturity when you rebalance less often than the data frequency, and use a different volatility for the hedge than for the sale (used in the stretch experiment). The same file contains `gamma_pnl_estimate`, which evaluates a discrete *approximation* of the theoretical gamma formula along each path (squared daily returns standing in for the true variance) so it can be compared with what the engine actually produced. It can use three different definitions of the squared return, which lets the project test how much the choice matters.

**`src/gbm_sim.py`** is Part A. It generates geometric Brownian motion paths (the textbook model of a stock price: random daily percentage moves with constant volatility) and runs the experiments A1 to A5. Every experiment draws from its own random stream built from seed 42, so results do not depend on the order things are run. This file also holds the exact closed-form expectation of the gamma formula, used to check the simulation.

**`src/real_data.py`** is Part B's data layer. It loads the 11 NIFTY files and 11 VIX files exactly as downloaded (they have different date formats, row orders and a hidden byte-order mark), treats two dead VIX rows as missing rather than guessing them, aligns the two series on common dates, and cuts rolling 21-trading-day windows. The raw files are never modified.

**`src/backtest.py`** is Part B's experiments. For each window it sells an at-the-money call priced at VIX/100, hedges daily, measures the realised volatility, and then runs the regression of PnL on the variance gap. It also does the overlap-aware statistics (Newey-West standard errors and a block bootstrap), the frequency study, the calm/turbulent split, the GBM benchmarks and the sensitivity checks.

**`src/research.py`** adds the extra empirical work on the same windows: confidence intervals for the headline effects, calm / drawdown / recovery market states, ten alternative ways to measure realised volatility (in three families, depending on whether they see the overnight gap), and a chronological calibration / validation / test split in which models are kept in two tracks: *ex-ante* (uses only what is known when the option is sold) and *ex-post* (uses the window's own realised path). **`src/calendar_time.py`** re-runs the hedge with the real calendar dates instead of 21 equal trading-day steps. **`src/frictions.py`** asks how much of the income survives trading costs, an option bid-ask spread and whole-lot trading; those are scenario assumptions on a round-number grid, because the data contain no quotes or fee schedules. **`src/integrity.py`** checks the 22 raw files against a SHA-256 manifest before anything runs, and **`src/data_dictionary.py`** writes `docs/DATA_DICTIONARY.md`. Run everything with `python -m src reproduce` (each stage runs in its own process and its time and memory are recorded by **`src/stages.py`**); **`src/verify_outputs.py`** is what CI uses to fail if the committed tables, documents or README are stale.

**`src/validation.py`** runs the brief's seven checks plus three extras. **`src/plots.py`** draws every figure. **`src/make_readme.py`** writes the README and the research report from the results files, so no number in it is typed by hand. **`src/config.py`** holds every setting, including the labelled placeholder rate. **`tests/`** is the pytest suite (over 250 tests, including ones that rebuild every Part B window from the raw CSV files with plain Python and compare against the saved results).

## 3. The gamma-based PnL result

**What it says.** If you hedge continuously using delta computed at the implied volatility `sigma_i`, but the market actually moves with realised volatility `sigma_r`, your profit as the seller is, to first order,

    PnL = integral of  (1/2) * Gamma * S^2 * (sigma_i^2 - sigma_r^2) dt

(with a small interest-rate growth factor). One important caveat: this is a continuous-time result. It is an exact identity only if you could watch the price continuously and hedge continuously. With daily closes, the code uses squared daily returns as a stand-in for the true variance, so what it computes is a close approximation, not an exact accounting. The experiments below measure how close. Three consequences:

- The sign is set by `sigma_i^2 - sigma_r^2`. Realised below implied: the seller makes money. Realised above: the seller loses.
- The size is set by `Gamma * S^2`, which is large only when the index is near the strike.
- This weight depends on the path. The same variance gap can be worth a lot or a little depending on *when* the volatile days happen.

**What the experiments found.**

- Part A, A3. At seven volatility settings the simulated mean PnL sits on the theoretical curve: every simulated mean is within 1.8 standard errors of it, and the sign always matches the gap. On a path-by-path basis, evaluating the discrete approximation along each path against the actual PnL gives R² = 0.999 at N = 250, and a new experiment (A6) shows its error shrinking steadily as the number of steps grows (from about 12.5% of the premium at N = 5 to 0.44% at N = 500). It is close but never exactly zero.
- Part B. The same path-by-path comparison on real NIFTY windows gives R² = 0.92 and slope 0.98. So the mechanism is right on real data. It is not an exact fit: on simulated windows with perfectly lognormal returns and the same 21 daily steps the R² is 0.97, and on the real windows it is a little lower (0.92 to 0.93, depending on whether squared log returns, squared simple returns or demeaned log returns are used), so the real data are somewhat harder to approximate. The README does not claim to know why.
- But the *simple* regression, PnL on (VIX² minus window realised variance), is much weaker on real data: slope 0.050 and R² 0.27, against a theoretical average slope of 0.40 and R² of 0.61 in an idealised simulation. Excluding the 66 windows that overlap the March 2020 crash lifts it to slope 0.236 and R² 0.59. The README (section 7) lays out what was measured: in the crash windows the volatility arrived while the index was far from the strike, where gamma is small, so a single "average realised variance" number is a poor summary of the window. The formula evaluated along the actual path captures this and fits well. Fat tails and volatility clustering are the likely further ingredient, but that was not separately tested and is not claimed.

**Why R² can be 0.84 in one place and 0.03 in another (A3).** Pooling all 140,000 simulated paths, PnL against the variance gap has R² = 0.84. Looking at one volatility setting at a time, the R² is only 0.03 to 0.64. Same data, same relationship. R² measures *the share of the wiggle that the regressor explains*, and that depends on how much the regressor itself wiggles. Across settings the gap varies widely (standard deviation 0.0250), so it dominates. Within a setting it varies only by sampling luck (about 0.0039, roughly six times narrower), and the other ingredient, when the variance arrived relative to the strike, becomes visible. It is like height against age: obvious when you compare children with adults, hard to see among a few people of the same age.

## 4. The 1/sqrt(N) scaling

Each rebalance interval leaves a small error. Over an interval `dt`, the leftover from not rebalancing is about `(1/2) Gamma S^2 sigma^2 (Z^2 - 1) dt`, where `Z` is the standard normal shock. Two things matter:

1. **Each error is tiny and has average zero.** `Z^2 - 1` is zero on average, so errors are as likely to help as to hurt.
2. **The errors from different intervals are (nearly) independent, so they add in variance, not in size.** Each has variance proportional to `dt^2`. With `N` intervals, `dt = T/N`, so the total variance is about `N * (T/N)^2 = T^2/N`, and the standard deviation is proportional to `1/sqrt(N)`.

Why not `1/N`? If the errors all pointed the same way and simply added up, the total would not shrink with N at all (each error is of order `dt`, and there are N of them, so the total is of order `N * dt = T`). If they were perfectly cancelling it would shrink faster. Independent random errors sit in between: they add in variance, so the total spread is the size of one error times the square root of the count, which gives `1/sqrt(N)`. A `1/N` law would need the sum to be effectively non-random, which it is not.

**Measured.** The log-log fit of the standard deviation of PnL against N (N = 5 to 500, 20,000 paths each) has slope **-0.4845** (regression SE 0.0021); for N of 20 or more it is -0.4891. Going from N = 5 to N = 500 cut the standard deviation by a factor of 9.26 (a `1/sqrt(N)` law predicts 10; `1/N` would predict 100). The fitted slope is within about 3% of -0.5 but several standard errors from it, because `std x sqrt(N)` is still creeping up (6.49 at N = 5 to 7.00 at N = 500). The honest summary is "close to -1/2", not "exactly -1/2".

On real data (only coarse frequencies are possible with daily closes), the difference between the PnL hedged every k days and the daily-hedged PnL follows a `1/sqrt(N)` pattern to within about 6% at all three coarser frequencies. That is a weak test (three points and one fitted constant) but consistent.

## 5. Six questions about the method

**1. What is gamma, and why is hedging error largest near the strike close to expiry?**
Delta says how much the option price moves when the index moves by one point. Gamma is how fast delta itself changes; it is the curvature of the option price. If gamma is small, delta barely changes between rebalances and a stale hedge is nearly right. If gamma is large, delta changes quickly and a stale hedge is wrong by a lot. Gamma is largest for an at-the-money option close to expiry, because then a small index move can flip the option from "almost surely worthless" to "almost surely exercised". With S0 = K = 100, sigma = 20%, r = 5%, the code gives a gamma of 0.019 with a year to go, 0.039 with three months, 0.099 with two weeks (tau = 0.04) and 0.199 with 2.5 trading days (tau = 0.01). At 20% away from the strike with three months left it is near zero. Since the hedging error per interval is proportional to gamma times the squared move, it is biggest where and when gamma is biggest.

**2. Why does the seller profit when realised volatility is lower than implied?**
The seller charged for volatility `sigma_i` when selling. The hedge costs money each time the index moves: the seller buys after rises and sells after falls, a systematic "buy high, sell low" effect that is paid in proportion to gamma times realised variance. The premium was set assuming that cost would be `sigma_i^2`. If the market delivers less movement (`sigma_r^2 < sigma_i^2`), the cost of hedging comes in below what was charged and the seller keeps the difference. That is exactly the gamma formula's `(sigma_i^2 - sigma_r^2)` term. In Part A at `sigma_r` = 10% (implied 20%) the seller makes about 3.84 per option on a premium of 10.45; at `sigma_r` = 30% the seller loses about 3.96.

**3. Why does hedging error shrink like 1/sqrt(N) and not 1/N?**
See section 4. The per-interval errors are random with zero mean, so they partly cancel. Independent random errors add in variance, so the standard deviation grows like the square root of their count times each one's size. Net result: standard deviation proportional to `1/sqrt(N)`. Measured slope -0.485.

**4. Does the drift mu of the stock matter for a perfectly hedged option? Why or why not?**
For a *perfectly* (continuously) hedged option, no. The hedge removes the exposure to the stock's direction, so the expected return of the stock cancels out of the replication argument; this is why `mu` does not appear in the Black-Scholes price. An important nuance shown by experiment A5: with *discrete* hedging the drift does leave a small footprint. At N = 250, a drift of 40% instead of 5% shifted the mean PnL by about 0.03 (under 0.3% of the premium), and cut the standard deviation of the PnL from 0.43 to 0.29, because a strongly drifting index spends less time near the strike, where gamma and therefore hedging error are largest. So: irrelevant for the price and for continuous replication, mildly relevant for the size of the discrete error.

**5. What is the volatility risk premium, and did your VIX-vs-realised plot show it?**
It is the tendency of implied volatility (what options are priced at) to exceed the volatility that is subsequently realised. Option buyers pay up for protection, like an insurance premium, and sellers collect it on average. In Part B the plot showed it: India VIX averaged 16.48 against 13.94 realised over the following 21 trading days, a gap of 2.53 volatility points (overlap-robust SE 0.50), with realised volatility below VIX in 81.2% of windows. The exceptions cluster in the March 2020 crash. In *variance* terms the average gap (0.0047) is not statistically distinguishable from zero (SE 0.0039), because the crash windows have huge negative gaps; I report both honestly. Also a caveat: VIX is built from a range of strikes and includes skew, so using it as an at-the-money volatility probably overstates the premium.

**6. What assumptions in Black-Scholes does your real-data backtest violate?**
Constant volatility (it changes day to day and clusters; VIX is itself a moving number and the hedge here uses the fixed initial VIX). Lognormal prices with no jumps (the worst day in the sample was -13.9% in log terms). Continuous trading (the hedge is rebalanced daily at closing prices). No transaction costs or bid-ask spreads (ignored in the baseline). A traded underlying with a known dividend and no basis (the index itself cannot be traded; futures and basis are ignored, and the price-return index excludes dividends). A constant, known risk-free rate (the 6% here is a placeholder, tested at 4% and 8%). One more issue is not a Black-Scholes assumption but affects the interpretation: no real option prices were used, so the option is a theoretical one priced from VIX.

## 6. Reading the main caveats

- **Overlap.** Consecutive 21-day windows share 20 days, so the 2,629 windows are not 2,629 independent data points; there are roughly 125 independent months. Standard errors are Newey-West (lag 20), about 10 times the naive ones for the slope, and the R² interval from a block bootstrap runs from 0.19 to 0.63. These numbers depend on how the overlap is modelled: I repeated them with other lags (the slope's standard error is 0.0114 at lag 5 and flat, about 0.016, from lag 20 to lag 63) and other bootstrap block lengths (the upper end of the R² interval rises from 0.63 to 0.68 as the block length goes from 21 to 63; the lower end stays near 0.19). The slope stays positive and far below the theory's 0.40 in every case, but the exact width of the intervals is not pinned down.
- **Placeholder rate.** The 6% risk-free rate is an assumption, not data. Moving it to 4% or 8% changes the mean PnL by at most 0.0033 percentage points and R² by at most 0.0034.
- **Where theory and data disagree.** The simple regression is weak on the full sample; the path-wise approximation fits well (but not exactly); the crash windows explain most of the gap. See section 7 of the research report.
- **Regimes and estimators.** The slope of PnL on the variance gap is about 0.04 in calm periods and drawdowns and about 0.20 in recoveries, and a slope fitted on calm 2016-2019 data explains calm 2023-onwards windows well but fails across the 2020 crash, so no single slope describes everything. The volatility risk premium is positive under all eleven volatility estimators, but its size runs from about 2.2 to 5.4 vol points depending on the estimator. The big values come from the range-based estimators that skip the overnight gap; add the gap back and they land at 2.2 to 2.5, next to the close-to-close family (2.5 to 2.8). So the estimators are not interchangeable, and the premium for a daily-close hedge is about 2.2 to 2.8 vol points.
- **Prediction versus explanation.** Models that use only what is known when the option is sold (VIX, VIX against the trailing 21-day volatility) have an out-of-sample R² between about -0.06 and +0.02 on the test periods: no predictive skill. The high R² values (about 0.9) belong to the path-wise gamma approximation, which uses the window's own realised path, so it explains the PnL after the fact and cannot be used to decide anything when selling the option.
- **Which clock.** Using real calendar dates (ACT/365) instead of 21 equal trading-day steps moves the mean PnL from +0.288% to +0.306% per window, the slope from 0.050 to 0.055 and R² from 0.27 to 0.29. The conclusions do not depend on the clock.
- **Exploratory, not confirmatory.** Many comparisons are reported without a multiple-testing correction, so the intervals are descriptive.
- **Not a trading strategy.** The +0.288% per window is the income of a theoretical Black-Scholes call priced from India VIX and hedged in the index. There are no option quotes, no futures hedge and no measured costs in the data, so it must not be described as a profitable strategy.
- **Costs.** The cost, spread and lot-size numbers are scenarios, not measurements. On the round-number grid the average income reaches zero at about 12 basis points of one-way hedging cost for daily hedging; what real costs are cannot be told from these files.
- **The chronological split** is a discipline check, not a blind test, because the headline results were known when the dates were chosen. Section 6 of the research report also shows how much of each bootstrap interval is just simulation noise (very little).
- **What was not done.** Delta-gamma hedging with a second option (a stretch goal), real option prices and futures hedging were not attempted; the last two need data that are not in the repository.

## 7. Project summary bullets (measured numbers only)

> **Discrete Delta-Hedging Error Analysis** | Python, NumPy, pandas, SciPy, matplotlib
> - Simulated GBM paths (fixed seed, 20,000 paths per setting) to delta-hedge short Black-Scholes calls at 7 rebalancing frequencies (N = 5 to 500); hedging-error dispersion scaled approximately as 1/sqrt(N) (fitted log-log slope -0.485, SE 0.002); simulated mean PnL matched a closed-form gamma-PnL expectation to within 1.8 standard errors across seven volatility settings.
> - Backtested daily delta-hedging of 21-day at-the-money calls on NIFTY 50 data (Jan 2016 to Oct 2026) using India VIX as implied volatility; hedging PnL regressed on the variance gap (implied² minus realised²) gave R² = 0.27 across 2,629 overlapping windows (overlap-robust inference; block-bootstrap 95% interval 0.19 to 0.63) and R² = 0.92 against a discrete path-wise gamma-PnL approximation (R² 0.92 to 0.93 across return conventions); documented and quantified where the simple regression breaks down (March 2020).

These bullets describe measurements on a *theoretical* hedge. They do not claim a tradable strategy, a predictive signal or a measured trading cost: the data contain no option quotes, no futures prices and no fee schedules, and the models that use only information known in advance have no out-of-sample skill.
