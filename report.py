"""
report.py
=========
Prints a human-readable summary of all findings to stdout
and saves a structured summary CSV for the paper's results section.

The summary is designed to answer the paper's core questions:

  Q1: Does HMM confidence saturate at 100% transition zone across assets?
  Q2: Does continuous confidence predict execution edge (Spearman)?
  Q3: Which alternative filter is strongest?
  Q4: Does any binary threshold recover a meaningful split?
"""

import numpy as np
import pandas as pd
from pathlib import Path


def _line(char="─", width=65):
    return char * width


def print_summary(spearman_df, filters_df, sweep_df, all_data, rolling_df=None):
    summary = {}

    print("\n" + _line("═"))
    print("  Q1: CONFIDENCE SATURATION")
    print(_line())

    pct_transitions = {}
    for ticker, asset_data in all_data.items():
        if asset_data.get("error") or not asset_data.get("daily"):
            continue
        pct = asset_data.get("pct_transition", 100.0)
        pct_transitions[ticker] = pct
        flag = "✓ OK" if pct <= 5 else ("✓ partial" if pct <= 20 else "⚠ NOT SATURATED")
        print(f"  {ticker:10s}  transition zone: {pct:5.1f}%  {flag}")

    overall_saturation = np.mean(list(pct_transitions.values()))
    pct_above_thresh = 100.0 - overall_saturation
    print(f"\n  Mean transition zone: {overall_saturation:.1f}%  (mean above threshold: {pct_above_thresh:.1f}%)")
    print(f"  Finding: {'CONFIRMED — threshold saturates' if overall_saturation < 10 else 'PARTIAL — some assets escape saturation'}")

    summary["q1_mean_saturation_pct"] = round(overall_saturation, 1)
    summary["q1_n_saturated"]         = sum(1 for v in pct_transitions.values() if v >= 95)
    summary["q1_n_total"]             = len(pct_transitions)

    print("\n" + _line("═"))
    print("  Q2: SPEARMAN CORRELATION (confidence → cost_diff)")
    print(_line())

    pooled_row = spearman_df[spearman_df["asset"] == "POOLED"]
    if not pooled_row.empty:
        rho = pooled_row["spearman_rho"].values[0]
        p   = pooled_row["spearman_p"].values[0]
        n   = pooled_row["n_days"].values[0]
        print(f"  POOLED: ρ={rho:.4f}, p={p:.4f}, n={n}")
        sig = "SIGNIFICANT ✓" if p < 0.05 else "NOT significant ✗"
        print(f"  → {sig}")
        summary["q2_pooled_rho"] = rho
        summary["q2_pooled_p"]   = p
        summary["q2_pooled_n"]   = n
        summary["q2_significant"] = bool(p < 0.05)

    print("\n  Per asset:")
    asset_rows = spearman_df[spearman_df["asset"] != "POOLED"]
    for _, row in asset_rows.iterrows():
        sig = "✓" if row["significant_0.05"] else "✗"
        print(f"  {row['asset']:10s}  ρ={row['spearman_rho']:+.4f}  "
              f"p={row['spearman_p']:.4f}  "
              f"pct_above_thresh={row['pct_above_0.60']:.1f}%  {sig}")

    n_sig = asset_rows["significant_0.05"].sum()
    print(f"\n  Significant per-asset: {n_sig}/{len(asset_rows)}")
    summary["q2_n_significant_assets"] = int(n_sig)

    print("\n" + _line("═"))
    print("  Q3: ALTERNATIVE FILTER COMPARISON (pooled)")
    print(_line())

    pooled_filters = filters_df[filters_df["asset"] == "POOLED"].copy()
    if not pooled_filters.empty:
        pooled_filters = pooled_filters.sort_values("spearman_rho")
        for _, row in pooled_filters.iterrows():
            sig  = "✓" if row["significant"] else "✗"
            print(f"  {row['filter']:20s}  ρ={row['spearman_rho']:+.4f}  "
                  f"p={row['spearman_p']:.4f}  {sig}")

        best_filter = pooled_filters.iloc[
            pooled_filters["spearman_rho"].abs().argmax()
        ]
        print(f"\n  Strongest filter: {best_filter['filter']}  "
              f"(ρ={best_filter['spearman_rho']:.4f})")
        summary["q3_best_filter"]     = best_filter["filter"]
        summary["q3_best_filter_rho"] = best_filter["spearman_rho"]

    print("\n" + _line("═"))
    print("  Q4: THRESHOLD SWEEP — does any threshold recover a split?")
    print(_line())

    pooled_sweep = sweep_df[sweep_df["asset"] == "POOLED"].dropna(subset=["gap"]).copy()
    if not pooled_sweep.empty:
        best_row = pooled_sweep.loc[pooled_sweep["gap"].abs().idxmin()]
        worst_row = pooled_sweep.loc[pooled_sweep["gap"].abs().idxmax()]

        print(f"  Threshold with smallest gap (least useful): "
              f"{best_row['threshold']:.2f}  gap={best_row['gap']:+.6f}")
        print(f"  Threshold with largest gap (most useful):  "
              f"{worst_row['threshold']:.2f}  gap={worst_row['gap']:+.6f}  "
              f"coverage={worst_row['pct_above']:.1f}%")

        mw_p = worst_row.get("mw_p", np.nan)
        sig   = f"p={mw_p:.4f} ✓" if not np.isnan(mw_p) and mw_p < 0.05 else "not significant ✗"
        print(f"  Significance at optimal threshold: {sig}")

        summary["q4_optimal_threshold"]    = float(worst_row["threshold"])
        summary["q4_optimal_gap"]          = float(worst_row["gap"])
        summary["q4_optimal_coverage_pct"] = float(worst_row["pct_above"])
        summary["q4_optimal_mw_p"]         = float(mw_p) if not np.isnan(mw_p) else None

    rolling_w10_sig = 0
    rolling_w10_best_rho = 0.0
    if rolling_df is not None and not rolling_df.empty:
        print("\n" + _line("═"))
        print("  Q5: W=10 ROLLING WINDOW — SIGNAL EMERGENCE")
        print(_line())
        w10 = rolling_df[
            (rolling_df["window"] == 10) & (rolling_df["asset"] != "POOLED")
        ].copy()
        for _, row in w10.sort_values(["signal", "asset"]).iterrows():
            sig_mark = "✓" if row["significant_0.05"] and row["correctly_signed"] else "✗"
            print(f"  {row['asset']:10s}  {row['signal']:12s}  "
                  f"ρ={row['spearman_rho']:+.4f}  p={row['spearman_p']:.4f}  {sig_mark}")
        sig_correct = w10[w10["significant_0.05"] & w10["correctly_signed"]]
        rolling_w10_sig = len(sig_correct)
        if rolling_w10_sig > 0:
            rolling_w10_best_rho = sig_correct["spearman_rho"].min()
        print(f"\n  Significant & correctly-signed at W=10: {rolling_w10_sig}/{len(w10)}")
        summary["q5_w10_n_significant"] = rolling_w10_sig
        summary["q5_w10_best_rho"]      = round(rolling_w10_best_rho, 4)

        print(f"\n  Signal emergence (key asset-signal pairs):")
        key_pairs = [("IWM", "entropy"), ("SPY", "entropy"), ("BTC-USD", "stay_prob")]
        for ticker, signal in key_pairs:
            rhos = []
            for w in [1, 3, 5, 10, 21]:
                sub = rolling_df[
                    (rolling_df["asset"] == ticker) &
                    (rolling_df["signal"] == signal) &
                    (rolling_df["window"] == w)
                ]
                rhos.append(f"{sub['spearman_rho'].values[0]:+.3f}" if len(sub) > 0 else "   n/a")
            print(f"  {ticker:8s} {signal:12s}  W1={rhos[0]}  W3={rhos[1]}  "
                  f"W5={rhos[2]}  W10={rhos[3]}  W21={rhos[4]}")

    print("\n" + _line("═"))
    print("VERDICT")
    print(_line())

    sat_transition = summary.get("q1_mean_saturation_pct", 100)
    is_saturated   = sat_transition < 10  

    if rolling_df is not None and not rolling_df.empty:
        w10_pooled = rolling_df[
            (rolling_df["window"] == 10) & (rolling_df["asset"] == "POOLED") &
            (rolling_df["signal"] == "entropy")
        ]
        rho_test  = w10_pooled["spearman_rho"].values[0] if len(w10_pooled) > 0 else summary.get("q2_pooled_rho", 0)
        sig_test  = rolling_w10_sig >= 2   
    else:
        rho_test  = summary.get("q2_pooled_rho", 0)
        sig_test  = summary.get("q2_pooled_p", 1) < 0.05

    if is_saturated and sig_test and rho_test < 0:
        verdict = (
            "STRONG: Threshold saturates (>90% days above 0.60) AND W=10 rolling\n"
            "  signals are significant. Temporal aggregation (3-10 days) is required\n"
            "  before regime uncertainty predicts execution quality."
        )
    elif is_saturated and not sig_test:
        verdict = (
            "SATURATION: Binary threshold saturates across all assets. Daily signals\n"
            "  are uninformative as expected; run rolling_window_test.py to check\n"
            "  whether W=10 aggregation reveals significance."
        )
    elif not is_saturated and sig_test:
        verdict = (
            "CALIBRATION: Saturation is asset-class dependent AND rolling signals\n"
            "  predict edge. Study is about cross-asset calibration of the threshold."
        )
    else:
        verdict = (
            "EXPLORATORY: Saturation and rolling signals both weak. Consider extending\n"
            "  the data period (period='2y') or checking HMM fit quality (test.py)."
        )

    print(f"\n  → {verdict}")
    summary["verdict"] = verdict.replace("\n", " ").replace("  ", " ")

    print("\n" + _line("═"))
    return summary


def save_summary_table(summary: dict, save_path: Path):
    rows = [{"metric": k, "value": v} for k, v in summary.items()]
    pd.DataFrame(rows).to_csv(save_path, index=False)
    print(f"\nSummary saved: {save_path}")