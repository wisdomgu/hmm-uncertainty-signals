"""
run_analysis.py
===============
Entry point. Run this file to reproduce all results from the paper:

  "Temporal Aggregation Reveals HMM Regime Uncertainty Signals in
   Optimal Trade Execution Across Asset Classes"

Usage:
    python run_analysis.py

Outputs (saved to results/ and figures/):
    results/spearman_results.csv
    results/alternative_filters.csv
    results/threshold_sweep.csv
    results/summary_table.csv
    figures/confidence_distributions.png
    figures/threshold_sweep.png
    figures/pooled_scatter.png
    figures/filter_comparison.png
    figures/signal_emergence.png
    data/raw_backtest_data.json

Author: Satish Garg
Date: 2025
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
import json
import time

from data_collector import collect_all_assets
from analysis_core  import (
    run_spearman_analysis,
    run_alternative_filters,
    run_threshold_sweep,
    run_rolling_spearman,   
)
from plotting import (
    plot_confidence_distributions,
    plot_threshold_sweep,
    plot_pooled_scatter,
    plot_filter_comparison,
    plot_signal_emergence,
)
from report import print_summary, save_summary_table

ASSETS  = ["SPY", "QQQ", "IWM", "BTC-USD", "ETH-USD", "GLD", "TLT", "AAPL"]
PERIOD  = "1y"
RESULTS = Path("results")
FIGURES = Path("figures")
DATA    = Path("data")

RESULTS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)


def main():
    print("=" * 65)
    print("  HMM Uncertainty Signals — Reproduction Script")
    print("=" * 65)
    print("\n[1/5] Collecting backtest data for all assets...")
    t0 = time.time()
    all_data = collect_all_assets(ASSETS, PERIOD)
    total_days = sum(
        len(v["daily"]) for v in all_data.values() if "daily" in v
    )
    print(f"      Done in {time.time() - t0:.1f}s  |  {total_days} total days")

    with open(DATA / "raw_backtest_data.json", "w") as f:
        json.dump(all_data, f, indent=2, default=str)

    print("\n[2/5] Plotting confidence distributions...")
    plot_confidence_distributions(
        all_data, FIGURES / "confidence_distributions.png"
    )

    print("\n[3/5] Running Spearman correlation analysis...")
    spearman_df = run_spearman_analysis(all_data)
    spearman_df.to_csv(RESULTS / "spearman_results.csv", index=False)
    print(spearman_df.to_string(index=False))

    print("\n[4/5] Testing alternative filters (entropy / duration / stay_prob)...")
    filters_df = run_alternative_filters(all_data)
    filters_df.to_csv(RESULTS / "alternative_filters.csv", index=False)
    print(filters_df.to_string(index=False))
    plot_filter_comparison(filters_df, FIGURES / "filter_comparison.png")
    rolling_df = run_rolling_spearman(all_data)

    print("\n[5/5] Running threshold sweep (0.30 → 0.90)...")
    sweep_df = run_threshold_sweep(all_data)
    sweep_df.to_csv(RESULTS / "threshold_sweep.csv", index=False)
    plot_threshold_sweep(sweep_df, FIGURES / "threshold_sweep.png")
    plot_pooled_scatter(all_data, FIGURES / "pooled_scatter.png")
    plot_signal_emergence(FIGURES / "signal_emergence.png")

    print("\n" + "=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    summary = print_summary(spearman_df, filters_df, sweep_df, all_data, rolling_df=rolling_df)
    save_summary_table(summary, RESULTS / "summary_table.csv")

    print(f"\nAll outputs saved to:")
    print(f"  figures/  → {[p.name for p in FIGURES.glob('*.png')]}")
    print(f"  results/  → {[p.name for p in RESULTS.glob('*.csv')]}")
    print(f"  data/     → raw_backtest_data.json")
    print("\nDone.")


if __name__ == "__main__":
    main()