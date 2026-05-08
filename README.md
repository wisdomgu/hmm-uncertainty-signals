# HMM Uncertainty Signals in Optimal Trade Execution
[![DOI](https://zenodo.org/badge/1228534501.svg)](https://doi.org/10.5281/zenodo.20079826)

This study asks whether HMM-derived confidence scores can predict when
regime-aware execution outperforms TWAP, and at what temporal resolution
that prediction holds.

Prior work (see companion paper below) showed that RL agents cannot exploit
regime signals even when the true label is in the state space. This study
asks a simpler question: can hand-crafted HMM uncertainty signals at least
*predict* when the regime-aware rule works? The answer turns out to depend
critically on the evaluation horizon.

---

## Research Questions

1. Do HMM confidence scores predict regime-aware execution quality on real assets?
2. At what temporal resolution does predictive power emerge?
3. Which uncertainty signal, composite confidence, raw entropy, or stay
   probability, is most informative, and does this vary across asset classes?

---

## Key Findings

| Asset | Signal | Horizon | Spearman ρ | p-value |
|-------|--------|---------|------------|---------|
| IWM | Raw entropy | 10-day | −0.411 | < 0.001 ✓ |
| BTC-USD | Raw entropy | 10-day | −0.190 | < 0.001 |
| BTC-USD | Stay probability | 10-day | −0.204 | < 0.001 |
| GLD | Raw entropy | 10-day | −0.167 | 0.013 |
| ETH-USD | Stay probability | 10-day | −0.110 | 0.044 |
| TLT | Stay probability | 10-day | −0.318 | < 0.001 |

Negative ρ = higher uncertainty predicts worse regime-aware execution (correct direction).  
IWM entropy is the primary finding: survives Bonferroni correction and 100% of bootstrap resamples are negative.

**Core result:** Daily signals are largely uninformative. Predictive power
emerges at 3-day aggregation and stabilises by 10 days. Raw entropy
outperforms composite confidence for equity indices; stay probability
is more informative for long-duration cryptocurrency assets. QQQ shows
no consistent signal and serves as a case study in data-period sensitivity
for sector-concentrated indices.

---

## Setup

```bash
git clone https://github.com/satishgarg/hmm-uncertainty-signals
cd hmm-uncertainty-signals
pip install -r requirements.txt
```

---

## Reproducing Results

```bash
python run_analysis.py          # full pipeline: downloads data, fits HMMs, runs all tests
```

All data is downloaded automatically via yfinance (internet required).  
Analysis uses a rolling one-year window ending at the time of the run; results
will reflect current market conditions rather than a fixed historical snapshot.  
Outputs are written to `results/` and `figures/` (created automatically on first run).
Expected runtime: ~5–10 minutes on a standard CPU.

To reproduce individual tables:

```bash
python rolling_window_test.py   # Table 6: signal emergence across horizons
python alt_execution_test.py    # Table 5: execution model sensitivity
python seed_robustness_test.py  # Table 4: bootstrap stability (1000 resamples)
```

---

## Structure
```
├── alt_execution_test.py     # Table 5: execution model sensitivity (±10% noise)
├── analysis_core.py          # Spearman tests, filter comparison, threshold sweep
├── data_collector.py         # Downloads OHLCV, computes signals and execution costs
├── plotting.py               # Publication figures
├── regime.py                 # 4-state HMM fitting, smoothing, walk-forward labelling
├── report.py                 # Summary printout and CSV export
├── requirements.txt
├── rolling_window_test.py    # Table 6: signal emergence across W ∈ {1,3,5,10,21}
├── run_analysis.py           # Entry point: runs full pipeline end-to-end
├── seed_robustness_test.py   # Table 4: bootstrap stability for primary findings
```
---

## What Each Script Does

**`regime.py`**: Core HMM module. Fits a 4-state Gaussian HMM (crash,
bearish, transitional, bullish) with BIC-based covariance selection (diagonal
vs full), state identification by return + trend score, minimum-duration
smoothing to suppress single-day regime switches, and walk-forward
out-of-sample labelling where sample size permits.

**`data_collector.py`**: Downloads daily OHLCV for all eight assets,
fits the HMM, and computes per-day metrics: composite confidence, raw
posterior entropy, stay probability, regime duration, and execution cost
difference (regime-aware minus TWAP). Outputs `data/raw_backtest_data.json` (generated on each run).

**`analysis_core.py`**: Four analyses: (1) confidence distribution
characterisation, (2) Spearman correlation between each signal and execution
cost difference, (3) alternative filter comparison across entropy / regime
duration / stay probability, (4) binary threshold sweep over θ ∈ [0.30, 0.90].

**`rolling_window_test.py`**: Replicates the temporal aggregation analysis,
testing all three signals at W ∈ {1, 3, 5, 10, 21} days and saving
`figures/signal_emergence.png`.

**`alt_execution_test.py`**: Adds ±10% Gaussian noise to cost differences
and re-runs Spearman tests for key asset–signal pairs to test sensitivity
to fill model misspecification.

**`seed_robustness_test.py`**: Bootstrap stability via 1000 resamples for
IWM entropy and BTC-USD stay probability at W = 10.

---

## Assets and Data

Eight assets via yfinance, rolling one-year window of daily OHLCV:

| Asset | Class | n (days) |
|-------|-------|----------|
| SPY, QQQ, IWM | Equity ETF | ~230 |
| BTC-USD, ETH-USD | Cryptocurrency | 345 |
| GLD | Commodity ETF | ~230 |
| TLT | Bond ETF | ~230 |
| AAPL | Large-cap equity | ~230 |

Execution costs are simulated from OHLCV prices. No live order or tick data is used.

---

## Notes on Limitations

All results use in-sample HMM fitting. Walk-forward out-of-sample validation
requires ≥ 20 windows (minimum n ≈ 672 days); no asset in this sample meets
that threshold. Reported effect sizes should be treated as upper bounds on
out-of-sample predictive power. Because the analysis uses a rolling data window,
specific point estimates will shift as the trailing year advances.

---

## Companion Paper

This paper extends:

> Garg, S. (2025). Regime Awareness in RL for Optimal Trade Execution:
> A Simulation Study. SSRN preprint 6559598.  
> https://papers.ssrn.com/abstract=6559598

That study showed flat PPO cannot exploit regime signals even with the true
label in the state space. This study asks the prior question: are the signals
empirically predictive at all, and at what timescale?

---

## Built On

- [Amrouni et al. (2022)](https://arxiv.org/abs/2202.00941): CTMSTOU simulation environment (JP Morgan AI Research)
- [hmmlearn](https://hmmlearn.readthedocs.io/): HMM fitting
- [yfinance](https://github.com/ranaroussi/yfinance): Market data
