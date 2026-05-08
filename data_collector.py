"""
data_collector.py
=================
Collects per-day confidence scores and execution costs for each asset.
Runs the HMM pipeline and execution simulation to produce the data used
by analysis_core.py and plotting.py.

Key output per asset:
    daily: list of dicts with keys:
        date, confidence, regime, twap_cost, regime_cost,
        cost_diff, entropy, regime_duration, stay_prob
"""

import numpy as np
import pandas as pd
import yfinance as yf
import warnings

warnings.filterwarnings("ignore")

from regime import (
    fit_hmm_4state,
    smooth_regimes,
    walk_forward_regimes,
)

def _safe_float(val, default=0.0):
    try:
        f = float(val)
        return default if (np.isnan(f) or np.isinf(f)) else f
    except Exception:
        return default


def _safe_float_or_nan(val):
    try:
        f = float(val)
        return np.nan if np.isinf(f) else f
    except Exception:
        return np.nan


def _fetch(ticker, period):
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "volume" not in df.columns and "Volume" in df.columns:
        df["volume"] = df["Volume"]
    return df


def _compute_regime_duration(states):
    durations = np.ones(len(states), dtype=int)
    for i in range(1, len(states)):
        if states[i] == states[i - 1]:
            durations[i] = durations[i - 1] + 1
        else:
            durations[i] = 1
    return durations


def _compute_stay_probs(states, model):
    state_map_inv = {v: k for k, v in model._state_map.items()}
    stay_probs = []
    for s in states:
        raw = state_map_inv.get(int(s), int(s))
        stay_probs.append(float(model.transmat_[raw, raw]))
    return np.array(stay_probs)


def _simulate_execution(df, states):
    """
    Simulate two execution strategies per trading day.

    TWAP:         open price (uniform intraday execution baseline).
    Regime-aware: applies strategy based on prior day's inferred regime:
        Crash (0)       → open price (halt / no improvement over TWAP)
        Bearish (1) or
        Transitional(2) → midpoint of open and prior close (patient limit)
        Bullish (3)     → open price (aggressive, same as TWAP)

    Returns df with twap_cost and regime_cost columns added.
    """
    df = df.copy()
    df["regime"]        = states
    df["regime_signal"] = pd.Series(states, index=df.index).shift(1)
    df["prior_close"]   = df["Close"].shift(1)
    df["twap_cost"]     = df["Open"]
    df["regime_cost"]   = df["Open"]

    crash_mask = df["regime_signal"] == 0
    df.loc[crash_mask, "regime_cost"] = df.loc[crash_mask, "Open"]

    patient_mask = df["regime_signal"].isin([1, 2])
    df.loc[patient_mask, "regime_cost"] = (
        df.loc[patient_mask, "Open"] + df.loc[patient_mask, "prior_close"]
    ) / 2

    df.dropna(subset=["regime_signal", "prior_close"], inplace=True)
    return df


def collect_asset(ticker, period="1y"):
    print(f"  {ticker}...", end=" ", flush=True)

    try:
        raw = _fetch(ticker, period)
        if raw.empty or len(raw) < 60:
            return {"ticker": ticker, "error": "Insufficient data", "daily": []}

        n = len(raw)

        use_walkforward = n >= 300
        method = "in-sample"

        wf_entropy   = None
        wf_stay_prob = None
        model        = None  
        scaler       = None

        if use_walkforward:
            states, confidence, wf_entropy, wf_stay_prob = walk_forward_regimes(
                raw, train_window=252, step=21
            )
            valid = states != -1
            if valid.sum() < 20:
                print(
                    f"(walk-forward only {valid.sum()} OOS days — "
                    f"falling back to in-sample) ",
                    end="", flush=True,
                )
                use_walkforward = False
                wf_entropy   = None
                wf_stay_prob = None
            else:
                df            = raw[valid].copy()
                states        = states[valid]
                conf_s        = confidence[valid]
                wf_entropy    = wf_entropy[valid]
                wf_stay_prob  = wf_stay_prob[valid]
                method        = "walk-forward OOS"

        if not use_walkforward:
            states, conf_s, model, scaler, df = fit_hmm_4state(raw)
            if states is None:
                return {"ticker": ticker, "error": "HMM fit failed", "daily": []}

        states = smooth_regimes(states, min_duration=3)

        df_exec   = _simulate_execution(df, states)
        durations = _compute_regime_duration(states)

        if wf_entropy is not None and wf_stay_prob is not None:
            entropy_vals = wf_entropy
            stay_probs   = wf_stay_prob
            n_valid = int((~np.isnan(entropy_vals)).sum())
            print(f"  [entropy] {n_valid}/{len(entropy_vals)} valid", end=" ", flush=True)
        else:
            feat_cols = ["returns", "volatility", "momentum", "trend",
                         "drawdown", "volume_lead"]
            stay_probs = _compute_stay_probs(states, model)
            available  = [c for c in feat_cols if c in df.columns]
            if len(available) == len(feat_cols):
                try:
                    feat_matrix = df[feat_cols].dropna()
                    scaled      = scaler.transform(feat_matrix.values)
                    probs_mat   = model.predict_proba(scaled)
                    ordered_cols = [model._state_map[r]
                                    for r in range(model.n_components)]
                    probs_mat   = probs_mat[:, np.argsort(ordered_cols)]
                    ent_vals    = -np.sum(
                        probs_mat * np.log(probs_mat + 1e-8), axis=1
                    )
                    entropy_series = pd.Series(ent_vals, index=feat_matrix.index)
                    entropy_vals   = entropy_series.reindex(df.index).values
                    print(
                        f"  [entropy] "
                        f"{(~np.isnan(entropy_vals)).sum()}/{len(entropy_vals)} valid",
                        end=" ", flush=True,
                    )
                except Exception as e:
                    print(f"  [entropy] failed: {e}", end=" ", flush=True)
                    entropy_vals = np.full(len(states), np.nan)
            else:
                entropy_vals = np.full(len(states), np.nan)

        exec_idx     = df_exec.index
        conf_aligned  = pd.Series(conf_s,     index=df.index).reindex(exec_idx).values
        dur_aligned   = pd.Series(durations,  index=df.index).reindex(exec_idx).values
        stay_aligned  = pd.Series(stay_probs, index=df.index).reindex(exec_idx).values
        ent_aligned   = pd.Series(entropy_vals, index=df.index).reindex(exec_idx).values
        state_aligned = pd.Series(states,     index=df.index).reindex(exec_idx).values

        daily = []
        for i, (date, row) in enumerate(df_exec.iterrows()):
            twap   = _safe_float(row["twap_cost"])
            regime = _safe_float(row["regime_cost"])
            conf   = _safe_float(
                conf_aligned[i] if i < len(conf_aligned) else np.nan
            )
            dur  = (
                int(dur_aligned[i])
                if i < len(dur_aligned) and not np.isnan(dur_aligned[i])
                else 1
            )
            stay = _safe_float_or_nan(
                stay_aligned[i] if i < len(stay_aligned) else np.nan
            )
            ent  = _safe_float_or_nan(
                ent_aligned[i]  if i < len(ent_aligned)  else np.nan
            )
            st   = (
                int(state_aligned[i])
                if i < len(state_aligned) and not np.isnan(state_aligned[i])
                else -1
            )

            daily.append({
                "date":            str(date.date()),
                "regime":          st,
                "confidence":      round(conf, 4),
                "entropy":         round(ent,  4) if not np.isnan(ent)  else None,
                "regime_duration": dur,
                "stay_prob":       round(stay, 4) if not np.isnan(stay) else None,
                "twap_cost":       round(twap,   4),
                "regime_cost":     round(regime, 4),
                "cost_diff":       round(regime - twap, 6),
            })

        LOW_CONF_THRESH = 0.60
        pct_transition  = round(
            sum(1 for d in daily if d["confidence"] < LOW_CONF_THRESH)
            / max(len(daily), 1) * 100,
            1,
        )

        print(f"OK ({len(daily)} days, {pct_transition}% transition zone, {method})")

        return {
            "ticker":         ticker,
            "n_days":         len(daily),
            "period":         period,
            "method":         method,
            "pct_transition": pct_transition,
            "daily":          daily,
            "error":          None,
        }

    except Exception as e:
        import traceback
        print(f"FAILED ({e})")
        traceback.print_exc()
        return {"ticker": ticker, "error": str(e), "daily": []}


def collect_all_assets(assets, period="1y"):
    results = {}
    for ticker in assets:
        results[ticker] = collect_asset(ticker, period)
    return results