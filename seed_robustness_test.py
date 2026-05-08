import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import json
import sys

seed_start = int(sys.argv[1]) if len(sys.argv) > 2 else 0
seed_end   = int(sys.argv[2]) if len(sys.argv) > 2 else 20

print(f"Seed range: {seed_start}–{seed_end}")

with open("data/raw_backtest_data.json") as f:
    all_data = json.load(f)

asset_data = all_data.get("IWM", {})
if not asset_data.get("daily"):
    print("No IWM data")
    exit()

df = pd.DataFrame(asset_data["daily"])
df = df.dropna(subset=["entropy", "cost_diff"])
df["rolling_cost_diff"] = df["cost_diff"].rolling(10).mean()
df = df.dropna(subset=["rolling_cost_diff"])

rng = np.random.default_rng(42)
boot_rhos = []
for _ in range(1000):
    idx = rng.integers(0, len(df), size=len(df))
    sample = df.iloc[idx]
    rho, _ = spearmanr(sample["entropy"], sample["rolling_cost_diff"])
    boot_rhos.append(rho)

boot_rhos = np.array(boot_rhos)
ci_low  = np.percentile(boot_rhos, 2.5)
ci_high = np.percentile(boot_rhos, 97.5)
pct_negative = (boot_rhos < 0).mean() * 100

print(f"\nIWM Entropy -- 10-day window Bootstrap Stability (n=1000 resamples)")
print(f"  Point estimate:  rho = {spearmanr(df['entropy'], df['rolling_cost_diff'])[0]:.4f}")
print(f"  95% CI:          [{ci_low:.4f}, {ci_high:.4f}]")
print(f"  % negative rho:  {pct_negative:.1f}%")
print(f"  Stable negative: {'YES' if ci_high < 0 else 'NO -- CI crosses zero'}")

asset_data2 = all_data.get("SPY", {})
df2 = pd.DataFrame(asset_data2["daily"])
df2 = df2.dropna(subset=["entropy", "cost_diff"])
df2["rolling_cost_diff"] = df2["cost_diff"].rolling(10).mean()
df2 = df2.dropna(subset=["rolling_cost_diff"])

boot_rhos2 = []
for _ in range(1000):
    idx = rng.integers(0, len(df2), size=len(df2))
    sample = df2.iloc[idx]
    rho, _ = spearmanr(sample["entropy"], sample["rolling_cost_diff"])
    boot_rhos2.append(rho)

boot_rhos2 = np.array(boot_rhos2)
ci_low2  = np.percentile(boot_rhos2, 2.5)
ci_high2 = np.percentile(boot_rhos2, 97.5)

print(f"\nSPY Entropy -- 10-day window Bootstrap Stability (n=1000 resamples)")
print(f"  Point estimate:  rho = {spearmanr(df2['entropy'], df2['rolling_cost_diff'])[0]:.4f}")
print(f"  95% CI:          [{ci_low2:.4f}, {ci_high2:.4f}]")
print(f"  % negative rho:  {(np.array(boot_rhos2) < 0).mean()*100:.1f}%")
print(f"  Stable negative: {'YES' if ci_high2 < 0 else 'NO -- CI crosses zero'}")

asset_data3 = all_data.get("BTC-USD", {})
df3 = pd.DataFrame(asset_data3["daily"])
df3 = df3.dropna(subset=["stay_prob", "cost_diff"])
df3["rolling_cost_diff"] = df3["cost_diff"].rolling(10).mean()
df3 = df3.dropna(subset=["rolling_cost_diff"])

boot_rhos3 = []
for _ in range(1000):
    idx = rng.integers(0, len(df3), size=len(df3))
    sample = df3.iloc[idx]
    rho, _ = spearmanr(sample["stay_prob"], sample["rolling_cost_diff"])
    boot_rhos3.append(rho)

boot_rhos3 = np.array(boot_rhos3)
ci_low3  = np.percentile(boot_rhos3, 2.5)
ci_high3 = np.percentile(boot_rhos3, 97.5)

print(f"\nBTC Stay Prob -- 10-day window Bootstrap Stability (n=1000 resamples)")
print(f"  Point estimate:  rho = {spearmanr(df3['stay_prob'], df3['rolling_cost_diff'])[0]:.4f}")
print(f"  95% CI:          [{ci_low3:.4f}, {ci_high3:.4f}]")
print(f"  % negative rho:  {(np.array(boot_rhos3) < 0).mean()*100:.1f}%")
print(f"  Stable negative: {'YES' if ci_high3 < 0 else 'NO -- CI crosses zero'}")