# Raw data sources

The 22 CSV files in this folder are **untouched** copies of the files supplied for this project.
No code in this repository writes to this folder.

| Series | Files | Coverage | Source | Downloaded |
|---|---|---|---|---|
| NIFTY 50 (price-return index, daily OHLC) | 11 yearly files `NIFTY 50_Historical_PR_*.csv` | 2016-01-01 to 2026-10-16 | https://www.niftyindices.com/reports/historical-data | 17 Oct 2026 |
| India VIX (daily OHLC, percent) | 11 yearly files `hist_india_vix_*.csv` | 2016-01-01 to 2026-10-16 (see gap below) | https://www.nseindia.com/reports-indices-historical-vix | 17 Oct 2026 |

Source URLs and download dates are as stated by the person who downloaded the files.

| Input | Value | Source |
|---|---|---|
| Risk-free rate for Part B | 6% per year, constant | **Unsourced assumption (placeholder).** No sourced rate was supplied. A sensitivity check at 4% and 8% is run and reported in the README. |

## Known problems in the raw files (all handled in `src/real_data.py`, none edited in place)

1. **VIX stops on 2022-12-01.** The 2022 VIX file is named `...-to-01-12-2022.csv`; December 2022
   (21 NIFTY trading days) has no VIX. A window can only start on a date that has a VIX close.
2. **Two stale VIX rows** (2021-02-12 and 2021-03-30) have Open = High = Low = Prev.Close = 0 and a Close
   that merely repeats the previous close. They are treated as missing, not forward-filled.
3. **Weekend-dated rows** (10 in NIFTY, 9 in VIX) are special trading sessions (Diwali Muhurat, Budget-day
   Saturdays, etc.). They are kept as ordinary trading days by default (`DROP_WEEKEND_SESSIONS` in
   `src/config.py`); a sensitivity run with them dropped is part of the backtest.
4. **One NIFTY-only weekend date** (2024-03-02, a Saturday special session) has no VIX row; it stays in
   the price paths but cannot be a window start.
5. File formats differ between the two sources (date format, row order, BOM, trailing spaces in VIX
   column names). The loaders normalise this.
6. A full report is in `results/data_inspection.txt` (regenerate with `python src/inspect_data.py`).

## Methodology references (official documents)

These describe what the two series are. They were read on the dates shown; this project does not reproduce them.

| Topic | Document | What it says that matters here |
|---|---|---|
| India VIX: what it measures | NSE, *India VIX* product page, https://www.nseindia.com/products-services/indices-indiavix-index (read 21 Oct 2026) | Built from NIFTY option order-book prices using the CBOE VIX methodology "with suitable amendments"; indicates expected volatility over the next 30 calendar days; quoted as an annualised percentage. |
| India VIX: computation | NSE, *Computation methodology of India VIX*, https://nsearchives.nseindia.com/web/sites/default/files/inline-files/India_VIX_comp_meth.pdf (read 21 Oct 2026) | Time to expiry is computed in minutes; near- and mid-month variances are interpolated to a constant 30-day maturity; the square root of the variance times 100 is the VIX value. It points to the white paper for the details and does not state the day-count basis. |
| India VIX: white paper | NSE, *White paper on India VIX*, https://nsearchives.nseindia.com/web/sites/default/files/inline-files/white_paper_IndiaVIX.pdf (linked from the product page; not read for this project) | Full formula; to be consulted before any claim about the exact annualisation basis. |
| NIFTY 50 index | NSE Indices Limited, *Methodology Document for Equity Indices* (November 2022), https://nsearchives.nseindia.com/content/indices/Method_Nifty_50.pdf (read 21 Oct 2026) | NIFTY 50 is a 50-stock, float-adjusted market-capitalisation-weighted index. The excerpt read does not itself define the price-return versus total-return variants; the files used here are labelled "PR" (price return) by the source. |

Consequences for the analysis: VIX is a 30-calendar-day, model-free, annualised figure built from option prices across strikes, not an at-the-money Black-Scholes volatility; using it as the Black-Scholes sigma is an approximation (README, Limitations).

## Terms of use of the data

The niftyindices.com *Terms of Use* (https://niftyindices.com/terms-of-use, no date shown; read 21 Oct 2026) say, among other things, that material from the site may not be "copied, modified, reproduced, republished, uploaded" without the company's prior written permission, that the site may be used "only for lawful purposes by individuals for personal, non-commercial use only", and that systematic or automated data collection needs express written consent. The NSE site (nseindia.com) has its own terms, which were not checked. These terms do not mention historical data specifically, so whether a public copy of the CSV files is allowed is **not settled**, and nothing here is legal advice.

Practical consequence: the public repository does **not** redistribute the CSV files; they are kept only in the author's private working copy for personal, non-commercial use. To reproduce the results, download the same files from the two source pages (selecting the same date ranges), place them unchanged in `data/raw/`, and run `python -m src check-data`; `MANIFEST.json` (SHA-256, row counts, date ranges) confirms that your download is byte-identical to the data used here.
