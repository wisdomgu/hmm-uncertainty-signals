"""
analysis_core.py
================
The four statistical analyses that form the paper's results section.

Analysis 1: Confidence distribution characterization (→ plotting.py)
Analysis 2: Spearman correlation — confidence vs execution cost diff
Analysis 3: Alternative filters — entropy / duration / stay_prob vs confidence
Analysis 4: Threshold sweep — binary threshold performance across 0.30→0.90

All functions return pandas DataFrames for clean CSV export.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr, mannwhitneyu
from typing import Dict, List

def _safe(val, default=np.nan):
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


def _build_pooled_df(all_data: dict) -> pd.DataFrame:
    """
    Flatten all asset daily records into one pooled DataFrame.
    Excludes assets with errors or fewer than 10 valid days.
    """
    rows = []
    for ticker, asset_data in all_data.items():
        if asset_data.get("error") or not asset_data.get("daily"):
            continue
        for day in asset_data["daily"]:
            # Skip days with missing key metrics
            if any(np.isnan(_safe(day.get(k, np.nan)))
                   for k in ["confidence", "cost_diff"]):
                continue
            rows.append({
                "asset":           ticker,
                "date":            day["date"],
                "regime":          int(day.get("regime", -1)),
                "confidence":      _safe(day.get("confidence")),
                "entropy":         _safe(day.get("entropy")),
                "regime_duration": _safe(day.get("regime_duration", 1)),
                "stay_prob":       _safe(day.get("stay_prob")),
                "twap_cost":       _safe(day.get("twap_cost")),
                "regime_cost":     _safe(day.get("regime_cost")),
                "cost_diff":       _safe(day.get("cost_diff")),
            })

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["confidence", "cost_diff"])
    return df

def run_spearman_analysis(all_data: dict) -> pd.DataFrame:
    """
    Core hypothesis test: does confidence predict execution cost difference?

    Tests both:
      H1: higher confidence → more negative cost_diff (regime-aware wins more)
      Null: confidence and cost_diff are independent

    Returns DataFrame with one row per asset + one pooled row.
    """
    pooled = _build_pooled_df(all_data)
    rows   = []

    def _spearman_row(label, sub):
        if len(sub) < 10:
            return None
        conf = sub["confidence"].values
        diff = sub["cost_diff"].values

        rho, p = spearmanr(conf, diff)

        r_p, p_p = pearsonr(conf, diff)

        median_conf = np.median(conf)
        high_mask   = conf >= median_conf
        low_mask    = ~high_mask

        mean_diff_high = np.nanmean(diff[high_mask])
        mean_diff_low  = np.nanmean(diff[low_mask])

        if high_mask.sum() > 5 and low_mask.sum() > 5:
            mw_stat, mw_p = mannwhitneyu(
                diff[high_mask], diff[low_mask], alternative="less"
            )
        else:
            mw_stat, mw_p = np.nan, np.nan

        return {
            "asset":                label,
            "n_days":               len(sub),
            "spearman_rho":         round(rho,   4),
            "spearman_p":           round(p,     4),
            "pearson_r":            round(r_p,   4),
            "pearson_p":            round(p_p,   4),
            "mean_conf":            round(np.mean(conf), 4),
            "std_conf":             round(np.std(conf),  4),
            "pct_above_0.60":       round((conf >= 0.60).mean() * 100, 1),
            "mean_cost_diff_high":  round(mean_diff_high, 6),
            "mean_cost_diff_low":   round(mean_diff_low,  6),
            "mw_p":                 round(mw_p,  4) if not np.isnan(mw_p) else np.nan,
            "significant_0.05":     bool(p < 0.05),
        }

    for ticker in pooled["asset"].unique():
        sub = pooled[pooled["asset"] == ticker]
        row = _spearman_row(ticker, sub)
        if row:
            rows.append(row)

    row = _spearman_row("POOLED", pooled)
    if row:
        rows.append(row)

    return pd.DataFrame(rows)


def run_alternative_filters(all_data: dict, window: int = 10) -> pd.DataFrame:
    """
    Compare four candidate filters for predicting execution edge at W-day rolling horizon:
      1. confidence     (composite: 0.7*entropy_norm + 0.3*stability)
      2. entropy        (raw HMM posterior entropy — lower = more certain)
      3. regime_duration (days in current regime — longer = more stable)
      4. stay_prob      (transition matrix diagonal — model's own stay estimate)

    Spearman rho is computed against the W-day rolling mean of cost_diff, not the
    raw daily value.  At daily resolution (W=1) all signals are near-zero (as shown
    in the paper); the paper's Figure 5 uses W=10 to reveal the real signal.
    """
    pooled = _build_pooled_df(all_data)

    pooled = pooled.sort_values(["asset", "date"])
    pooled["rolling_cost_diff"] = (
        pooled.groupby("asset")["cost_diff"]
        .transform(lambda x: x.rolling(window, min_periods=window).mean())
    )
    pooled_w = pooled.dropna(subset=["rolling_cost_diff"]).copy()

    rows = []

    filters = {
        "confidence":      "confidence",
        "entropy":         "entropy",
        "regime_duration": "regime_duration",
        "stay_prob":       "stay_prob",
    }

    def _filter_row(filter_name, col, asset_label, sub):
        vals = sub[col].dropna()
        diff = sub.loc[vals.index, "rolling_cost_diff"]
        if len(vals) < 10:
            return None
        rho, p = spearmanr(vals.values, diff.values)
        return {
            "filter":         filter_name,
            "asset":          asset_label,
            "window":         window,
            "n_days":         len(vals),
            "spearman_rho":   round(rho, 4),
            "spearman_p":     round(p,   4),
            "significant":    bool(p < 0.05),
            "mean_filter":    round(vals.mean(), 4),
            "std_filter":     round(vals.std(),  4),
            "pct_missing":    round(sub[col].isna().mean() * 100, 1),
        }

    for filter_name, col in filters.items():
        if col not in pooled_w.columns:
            print(f"  [WARN] Column '{col}' not in data — skipping filter {filter_name}")
            continue

        row = _filter_row(filter_name, col, "POOLED", pooled_w)
        if row:
            rows.append(row)

        for ticker in pooled_w["asset"].unique():
            sub = pooled_w[pooled_w["asset"] == ticker]
            row = _filter_row(filter_name, col, ticker, sub)
            if row:
                rows.append(row)

    return pd.DataFrame(rows)


def run_rolling_spearman(all_data: dict, windows: List[int] = None) -> pd.DataFrame:
    """
    Compute Spearman rho between each uncertainty signal and rolling W-day
    mean cost_diff for every asset × window combination.

    This is the core analysis behind the paper's Table 3/6 and Figure 4.
    Signal strength grows with window because regime-driven execution savings
    accumulate over multi-day periods and daily noise partially cancels.

    Returns a DataFrame with one row per (asset, signal, window).
    """
    if windows is None:
        windows = [1, 3, 5, 10, 21]

    signals = ["confidence", "entropy", "stay_prob"]
    rows: List[dict] = []

    for window in windows:
        pooled_rows = []

        for ticker, asset_data in all_data.items():
            if not asset_data.get("daily"):
                continue
            df = pd.DataFrame(asset_data["daily"])
            df = df.dropna(subset=["cost_diff"])
            df["rolling_cost_diff"] = df["cost_diff"].rolling(window, min_periods=window).mean()
            df = df.dropna(subset=["rolling_cost_diff"])
            if len(df) < window + 5:
                continue

            for signal in signals:
                sub = df.dropna(subset=[signal])
                if len(sub) < 10:
                    continue
                rho, p = spearmanr(sub[signal].values, sub["rolling_cost_diff"].values)
                rows.append({
                    "asset":            ticker,
                    "signal":           signal,
                    "window":           window,
                    "n_days":           len(sub),
                    "spearman_rho":     round(rho, 4),
                    "spearman_p":       round(p,   4),
                    "significant_0.05": bool(p < 0.05),
                    "correctly_signed": bool(rho < 0),
                })
                pooled_rows.append({"asset": ticker, "signal": signal,
                                     "rho": rho, "p": p})

        for signal in signals:
            sub = [r for r in pooled_rows if r["signal"] == signal]
            if len(sub) < 2:
                continue
            z_vals = np.arctanh(np.clip([r["rho"] for r in sub], -0.999, 0.999))
            pooled_rho = float(np.tanh(z_vals.mean()))
            n_sig = sum(1 for r in sub if r["p"] < 0.05 and r["rho"] < 0)
            rows.append({
                "asset":            "POOLED",
                "signal":           signal,
                "window":           window,
                "n_days":           sum(r["rho"] != np.nan for r in sub),
                "spearman_rho":     round(pooled_rho, 4),
                "spearman_p":       np.nan,   # pooled z-test omitted
                "significant_0.05": bool(n_sig >= 2),
                "correctly_signed": bool(pooled_rho < 0),
            })

    return pd.DataFrame(rows)

def run_threshold_sweep(all_data: dict) -> pd.DataFrame:
    """
    For each threshold t in [0.30, 0.90]:
      - Split days into high-confidence (conf >= t) and low-confidence (conf < t)
      - Compute mean cost_diff for each group
      - Record pct_days_above (coverage) and statistical separation

    This reveals:
      a) Whether any threshold produces a meaningful split
      b) The optimal threshold (if one exists)
      c) Coverage-accuracy tradeoff

    Run pooled and per-asset.
    """
    pooled = _build_pooled_df(all_data)
    thresholds = np.arange(0.30, 0.91, 0.02)
    rows = []

    def _sweep_row(threshold, asset_label, sub):
        conf = sub["confidence"].values
        diff = sub["cost_diff"].values
        n    = len(sub)

        high_mask = conf >= threshold
        low_mask  = ~high_mask
        n_high = high_mask.sum()
        n_low  = low_mask.sum()

        mean_high = np.nanmean(diff[high_mask]) if n_high > 0 else np.nan
        mean_low  = np.nanmean(diff[low_mask])  if n_low  > 0 else np.nan

        mw_p = np.nan
        if n_high >= 5 and n_low >= 5:
            try:
                _, mw_p = mannwhitneyu(diff[high_mask], diff[low_mask],
                                       alternative="less")
            except Exception:
                pass

        return {
            "threshold":          round(threshold, 2),
            "asset":              asset_label,
            "n_total":            n,
            "n_above":            int(n_high),
            "n_below":            int(n_low),
            "pct_above":          round(n_high / n * 100, 1),
            "mean_cost_diff_above": round(mean_high, 6) if not np.isnan(mean_high) else np.nan,
            "mean_cost_diff_below": round(mean_low,  6) if not np.isnan(mean_low)  else np.nan,
            "gap":                round(mean_high - mean_low, 6)
                                  if not np.isnan(mean_high) and not np.isnan(mean_low)
                                  else np.nan,
            "mw_p":               round(mw_p, 4) if not np.isnan(mw_p) else np.nan,
        }

    for t in thresholds:
        row = _sweep_row(t, "POOLED", pooled)
        rows.append(row)

        for ticker in pooled["asset"].unique():
            sub = pooled[pooled["asset"] == ticker]
            row = _sweep_row(t, ticker, sub)
            rows.append(row)

    return pd.DataFrame(rows)

def regime_breakdown(all_data: dict) -> pd.DataFrame:

    pooled = _build_pooled_df(all_data)
    rows = []
    labels = {0: "crash", 1: "bearish", 2: "transitional", 3: "bullish"}

    for asset in list(pooled["asset"].unique()) + ["POOLED"]:
        sub = pooled if asset == "POOLED" else pooled[pooled["asset"] == asset]
        for r, label in labels.items():
            rm = sub[sub["regime"] == r]
            if len(rm) < 3:
                continue
            rows.append({
                "asset":          asset,
                "regime":         label,
                "n_days":         len(rm),
                "pct_days":       round(len(rm)/len(sub)*100, 1),
                "mean_cost_diff": round(rm["cost_diff"].mean(), 6),
                "std_cost_diff":  round(rm["cost_diff"].std(),  6),
                "mean_confidence":round(rm["confidence"].mean(), 4),
                "mean_duration":  round(rm["regime_duration"].mean(), 1),
            })

    return pd.DataFrame(rows)