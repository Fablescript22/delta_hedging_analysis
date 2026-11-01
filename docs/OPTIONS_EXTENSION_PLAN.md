# Plan: from a theoretical option to an executable one

*Hand-written planning document (not generated). Nothing below has been done; each milestone needs data that is not in this repository.*

## Why this is a separate milestone

The current Part B hedges a **theoretical** Black-Scholes call priced from India VIX, using the index itself, with zero costs. That isolates the hedging mechanism, but it cannot say what a trader could have earned: there is no option price to sell at, no futures contract to hedge with, and no measured cost. The three steps below close those gaps one at a time, so that each answers a single question and none is hidden inside another.

## Milestone 1: actual option prices instead of VIX

*Question: how far is the volatility the option market charged for an at-the-money NIFTY option from India VIX, and does the gap change the result?*

- **Data needed.** Historical end-of-day prices for NIFTY index options (every strike and expiry), with contract specifications (expiry date, strike, lot size), for the same 2016-2026 span. NSE publishes these as daily derivatives bhavcopy files; the format changed over the period, so the loader must handle both.
- **Work.** Select the contract nearest at-the-money with about 21 trading days to expiry at each start date; invert Black-Scholes for its implied volatility using the sourced rate (below); compare with VIX; re-run the backtest selling the *actual* option at its settlement price and settling at the actual expiry payoff.
- **Acceptance.** Tests that the implied-volatility inversion round-trips Black-Scholes prices; that the contract chosen at each date is the documented rule; a table of (option implied vol - VIX) with HAC intervals; the Part B result recomputed with the real option, reported beside the theoretical one.
- **Known problems to plan for.** Settlement prices are not executable prices; illiquid strikes; European-style cash settlement against the index; expiry-day conventions (weekly versus monthly contracts changed over the period); dividends (the series used here is price return).

## Milestone 2: futures as the hedge instrument

*Question: how much of the result survives hedging with the instrument a desk would actually use?*

- **Data needed.** NIFTY futures daily prices for each expiry, with the expiry calendar.
- **Work.** Replace the index hedge with the near-month future (and its roll); the hedge ratio is applied to the future, so the cash account no longer pays the full notional; add the basis (future minus index) and financing as explicit terms; compute the delta from the forward, not the spot.
- **Rate.** Replace the 6% placeholder with a sourced short rate (for example the RBI/CCIL 91-day treasury-bill yield, daily) and report how much the result moves relative to the placeholder.
- **Acceptance.** An identity test: with a zero basis and the sourced rate equal to the placeholder, the futures hedge reproduces the index hedge; a decomposition of the PnL difference into basis, carry and roll.

## Milestone 3: measured frictions

*Question: replace the round-number cost grid with published schedules.*

- **Data needed.** The exchange's and broker's published fee schedules (brokerage, exchange transaction charges, securities transaction tax, GST, regulatory and stamp charges) for each year, lot sizes by date, and, if obtainable, quoted bid-ask spreads for the contracts used.
- **Work.** Apply the schedule per trade, rounded to whole lots, by date; replace the half-spread scenario with observed spreads where they exist, and say plainly where they do not.
- **Acceptance.** A cost-attribution table (fees, taxes, lot rounding, spread) and the net PnL with an interval; the earlier break-even grid compared with the measured schedule.

## What would change the conclusions, and what would not

- If the actual option implied volatility sits far below VIX at the money, the theoretical income shrinks accordingly; the README's statement that the theoretical number is optimistic would then be quantified.
- The 1/sqrt(N) scaling and the mechanism (PnL follows realised against implied variance along the path) do not depend on these inputs.
- The ex-ante finding (nothing known at the start predicts the PnL) should be re-checked with the real option's implied volatility as a predictor, since VIX is only a proxy.

## Order of work and what to report

1. Milestone 1 first (it changes what is being sold). 2. Milestone 2 (it changes the hedge). 3. Milestone 3 (it changes the cost). After each, rerun the full pipeline, keep the earlier theoretical result in the report for comparison, and state the new limitations. Until Milestone 1 is complete, no statement about tradable performance should be made.
