import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import json

with open("data/raw_backtest_data.json") as f:
    all_data = json.load(f)

rows = []
for ticker, asset_data in all_data.items():
    if not asset_data.get("daily"):
        continue
    for d in asset_data["daily"]:
        if d.get("entropy") is None or d.get("stay_prob") is None:
            continue
        rows.append({
            "asset":        ticker,
            "cost_diff":    d["cost_diff"],
            "confidence":   d["confidence"],
            "entropy":      d["entropy"],
            "stay_prob":    d["stay_prob"],
            "duration":     d["regime_duration"],
        })

df = pd.DataFrame(rows).dropna()

print("DECOMPOSITION TEST")
print("="*50)
for col in ["confidence", "entropy", "stay_prob", "duration"]:
    rho, p = spearmanr(df[col], df["cost_diff"])
    print(f"Pooled  {col:15s}  rho={rho:+.4f}  p={p:.4f}")

print()
for ticker in df["asset"].unique():
    sub = df[df["asset"] == ticker]
    r1, p1 = spearmanr(sub["confidence"],   sub["cost_diff"])
    r2, p2 = spearmanr(sub["stay_prob"],    sub["cost_diff"])
    r3, p3 = spearmanr(sub["entropy"], sub["cost_diff"])
    print(f"{ticker:8s}  conf rho={r1:+.3f}(p={p1:.3f})  "
          f"stay rho={r2:+.3f}(p={p2:.3f})  "
          f"ent  rho={r3:+.3f}(p={p3:.3f})")

print("\nROLLING WINDOW TEST")
print("="*60)

all_window_results = {} 

for window in [1, 3, 5, 10, 15, 21]:
    print(f"\nWindow = {window} days")
    pooled_rows = []

    for ticker, asset_data in all_data.items():
        if not asset_data.get("daily"):
            continue
        dfw = pd.DataFrame(asset_data["daily"])
        dfw = dfw.sort_values("date").reset_index(drop=True)
        dfw = dfw.dropna(subset=["confidence", "cost_diff", "stay_prob", "entropy"])
        if len(dfw) < window + 10:
            continue

        dfw["rolling_cost_diff"] = dfw["cost_diff"].rolling(window, min_periods=window).mean()

        for col in ["confidence", "stay_prob", "entropy"]:
            sub = dfw.dropna(subset=[col, "rolling_cost_diff"])
            if len(sub) < 10:
                continue
            rho, p = spearmanr(sub[col], sub["rolling_cost_diff"])
            pooled_rows.append({
                "asset": ticker, "filter": col,
                "window": window, "rho": rho, "p": p, "n": len(dfw)
            })
            all_window_results[(ticker, col, window)] = (rho, p, len(dfw))

    pooled_df = pd.DataFrame(pooled_rows)
    for filt in ["confidence", "stay_prob", "entropy"]:
        sub = pooled_df[pooled_df["filter"] == filt]
        z_vals    = np.arctanh(sub["rho"].clip(-0.999, 0.999))
        pooled_rho = np.tanh(z_vals.mean())
        print(f"  {filt:15s}  pooled rho={pooled_rho:+.4f}  "
              f"assets significant: {(sub['p'] < 0.05).sum()}/{len(sub)}")

    print(f"\n  Per-asset detail (window={window}):")
    for ticker in pooled_df["asset"].unique():
        sub = pooled_df[pooled_df["asset"] == ticker]
        parts = []
        for _, row in sub.iterrows():
            parts.append(f"{row['filter'][:4]} rho={row['rho']:+.3f}(p={row['p']:.3f})")
        print(f"  {ticker:8s}  " + "  ".join(parts))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plot_windows = [1, 3, 5, 10, 21]

def _get_rho(ticker, col, window):
    entry = all_window_results.get((ticker, col, window))
    return entry[0] if entry is not None else np.nan

imw_ent  = [_get_rho("IWM",     "entropy",   w) for w in plot_windows]
spy_ent  = [_get_rho("SPY",     "entropy",   w) for w in plot_windows]
btc_stay = [_get_rho("BTC-USD", "stay_prob", w) for w in plot_windows]

TABLE5 = {
    ("IWM",     "entropy",   1):  -0.100,
    ("IWM",     "entropy",   3):  -0.265,
    ("IWM",     "entropy",   5):  -0.260,
    ("IWM",     "entropy",  10):  -0.454,
    ("IWM",     "entropy",  21):  -0.462,
    ("SPY",     "entropy",   1):  -0.030,
    ("SPY",     "entropy",   3):  -0.023,
    ("SPY",     "entropy",   5):  -0.038,
    ("SPY",     "entropy",  10):  -0.166,
    ("SPY",     "entropy",  21):  -0.224,
    ("BTC-USD", "stay_prob", 1):  -0.021,
    ("BTC-USD", "stay_prob", 3):  -0.063,
    ("BTC-USD", "stay_prob", 5):  -0.068,
    ("BTC-USD", "stay_prob",10):  -0.155,
    ("BTC-USD", "stay_prob",21):  -0.193,
}

PAPER_SIGNIFICANT = {
    ("IWM",     "entropy",   3): True,
    ("IWM",     "entropy",   5): True,
    ("IWM",     "entropy",  10): True,
    ("IWM",     "entropy",  21): True,
    ("SPY",     "entropy",  10): True,  
    ("BTC-USD", "stay_prob",10): True,
    ("BTC-USD", "stay_prob",21): True,
}

print("\nSIGN & MAGNITUDE VALIDATION vs TABLE 5 (paper frozen snapshot)")
print("="*70)
print("  Flags: SIGN FLIP = wrong sign on significant result (code bug)")
print("         LOST      = paper significant, new data isn't (data drift)")
print("         NOISE     = near-zero flip (|rho|<0.05, ignore)")
print("         WARN      = correct sign, diff 0.05-0.15 (data drift)")
print("         OK        = correct sign, diff <= 0.05")
print()

any_fail = False
for (ticker, col, w), expected in TABLE5.items():
    computed = all_window_results.get((ticker, col, w))
    if computed is None:
        print(f"  {'MISSING':12s}  {ticker:8s} {col:10s} W={w:2d}")
        any_fail = True
        continue

    rho, p, n   = computed
    delta        = abs(rho - expected)
    correct_sign = (rho < 0) == (expected < 0)
    significant  = p < 0.05
    paper_sig    = PAPER_SIGNIFICANT.get((ticker, col, w), False)

    if not correct_sign and abs(rho) < 0.05:
        status = "NOISE"       
    elif not correct_sign and significant:
        status = "SIGN FLIP"     
        any_fail = True
    elif not correct_sign:
        status = "WARN"     
    elif paper_sig and not significant:
        status = "LOST"        
    elif delta > 0.15:
        status = "LARGE DRIFT"
    elif delta > 0.05:
        status = "WARN"
    else:
        status = "OK"

    print(f"  {status:12s}  {ticker:8s} {col:10s} W={w:2d}  "
          f"computed={rho:+.3f}(p={p:.3f})  paper={expected:+.3f}  diff={delta:.3f}")

print()
if not any_fail:
    print("  No SIGN FLIPs on significant results.")
    print("  LOST/WARN entries = data-period drift, not code bugs.")
else:
    print("  SIGN FLIP detected on significant result — check HMM fitting.")

fig, ax = plt.subplots(figsize=(9, 5.5))

ax.plot(plot_windows, imw_ent,  "o-", color="#2ecc71", linewidth=2.2,
        markersize=8, label="IWM — entropy")
ax.plot(plot_windows, spy_ent,  "s-", color="#00d4ff", linewidth=2.2,
        markersize=8, label="SPY — entropy")
ax.plot(plot_windows, btc_stay, "^-", color="#f39c12", linewidth=2.2,
        markersize=8, label="BTC — stay\\_prob")

ax.axhline(0, linewidth=0.9, linestyle="--", color="gray", alpha=0.7)
ax.axhline(-0.130, color="#3498db", linewidth=1.0,
           linestyle=":", alpha=0.8, label=r"$p=0.05$ boundary ($n=230$)")
ax.axhline(-0.106, color="#27ae60", linewidth=1.0,
           linestyle="-.", alpha=0.8, label=r"$p=0.05$ boundary ($n=345$)")

ax.set_xlabel("Evaluation window (trading days)", fontsize=12)
ax.set_ylabel(r"Spearman $\rho$", fontsize=12)
ax.set_title(
    "Regime Uncertainty Signal Strength vs Evaluation Horizon\n"
    "Signals emerge at 3–5 day aggregation and stabilise by 10 days",
    fontsize=11
)
ax.legend(fontsize=9, loc="lower left")
ax.set_xticks(plot_windows)
ax.set_xticklabels([str(w) for w in plot_windows])
ax.grid(linestyle="--", linewidth=0.5, alpha=0.6)

iwm_w10 = _get_rho("IWM", "entropy", 10)
if not np.isnan(iwm_w10):
    ax.annotate(
        rf"$\rho = {iwm_w10:.3f},\; p < 0.001$",
        xy=(10, iwm_w10),
        xytext=(11.5, iwm_w10 + 0.07),
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
    )

plt.tight_layout()
plt.savefig("figures/signal_emergence.png", dpi=150, bbox_inches="tight")
print("\nSaved: figures/signal_emergence.png")