import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import json

with open("data/raw_backtest_data.json") as f:
    all_data = json.load(f)

print("ALTERNATIVE EXECUTION MODEL TEST")
print("="*60)
print("Tests whether key findings hold under different fill assumptions")
print()

KEY_PAIRS = [
    ("IWM",    "entropy",   10),
    ("SPY",    "entropy",   10),
    ("BTC-USD","stay_prob", 10),
    ("GLD",    "stay_prob", 10),
]

for ticker, signal_col, window in KEY_PAIRS:
    asset_data = all_data.get(ticker, {})
    if not asset_data.get("daily"):
        continue

    df = pd.DataFrame(asset_data["daily"])

    df = df.dropna(subset=[signal_col, "cost_diff"])
    df["rolling_cost_diff"] = df["cost_diff"].rolling(window).mean()
    df = df.dropna(subset=["rolling_cost_diff"])

    rho_base, p_base = spearmanr(df[signal_col], df["rolling_cost_diff"])

    rng = np.random.default_rng(42)
    noise = rng.normal(0, df["cost_diff"].std() * 0.1, size=len(df))
    df["cost_diff_perturbed"] = df["cost_diff"] + noise
    df["rolling_perturbed"] = df["cost_diff_perturbed"].rolling(window).mean()
    df = df.dropna(subset=["rolling_perturbed"])
    rho_pert, p_pert = spearmanr(df[signal_col], df["rolling_perturbed"])

    print(f"{ticker:8s} {signal_col:10s} W={window}")
    print(f"  Baseline:   rho={rho_base:+.4f}  p={p_base:.4f}")
    print(f"  +10% noise: rho={rho_pert:+.4f}  p={p_pert:.4f}")
    stable = "STABLE" if (rho_base < 0) == (rho_pert < 0) and p_pert < 0.05 else "UNSTABLE"
    print(f"  → {stable}")
    print()