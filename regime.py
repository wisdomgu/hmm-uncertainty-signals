import numpy as np
from hmmlearn import hmm
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import shap

def fit_hmm_4state(df, n_states=4):
    """
    4-state HMM: crash(0), bearish(1), transitional(2), bullish(3).
    Ordered by combined return+trend score ascending.
    Crash state is identified as the one with highest volatility among
    the two lowest-return states.
    """
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "volume" not in df.columns and "Volume" in df.columns:
        df["volume"] = df["Volume"]
    df = df.loc[:, ~df.columns.duplicated()]

    df["returns"]    = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()
    df["momentum"]   = df["Close"].pct_change(5)
    df["trend"]      = df["Close"] / df["Close"].rolling(20).mean() - 1
    df["drawdown"]   = df["Close"] / df["Close"].cummax() - 1
    df["log_volume"] = np.log(df["volume"])
    df["volume_z"]   = (
        df["log_volume"] - df["log_volume"].rolling(20).mean()
    ) / df["log_volume"].rolling(20).std()
    df["volume_lead"] = df["volume_z"].shift(1)

    features_df = df[["returns", "volatility", "momentum", "trend", "drawdown", "volume_lead"]].dropna()
    df = df.loc[features_df.index].copy()

    scaler = StandardScaler()
    features = scaler.fit_transform(features_df.values)

    n_states = 4
    best_model, cov_type, bic_results = fit_best_hmm(features, n_states)

    if best_model is None:
        return None, None, None, None, df

    raw_states = best_model.predict(features)
    probs      = best_model.predict_proba(features)
    means      = best_model.means_

    scores = np.array([float(means[i][0]) + float(means[i][3]) for i in range(n_states)])
    order  = np.argsort(scores)

    low_two = order[:2]
    vols    = [float(means[s][1]) for s in low_two]
    crash_raw = int(low_two[np.argmax(vols)])
    bear_raw  = int(low_two[np.argmin(vols)])

    high_two  = order[2:]
    trans_raw = int(high_two[0])
    bull_raw  = int(high_two[1])

    state_map = {
        crash_raw: 0,
        bear_raw:  1,  
        trans_raw: 2,  
        bull_raw:  3,  
    }

    states = np.array([state_map[int(s)] for s in raw_states])
    probs  = probs[:, [crash_raw, bear_raw, trans_raw, bull_raw]]

    entropy    = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    stability  = np.zeros(len(states))
    stability[1:] = (states[1:] == states[:-1]).astype(float)
    confidence = 0.7 * (1.0 - entropy / np.log(n_states)) + 0.3 * stability

    best_model._cov_type_selected = cov_type
    best_model._bic_results = {k: round(v["bic"], 2) for k, v in bic_results.items()}
    best_model._state_map   = state_map

    ordered_states = [crash_raw, bear_raw, trans_raw, bull_raw]
    best_model._state_order = ordered_states

    return states, confidence, best_model, scaler, df

def fit_hmm(df, n_states=3):
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if "volume" not in df.columns and "Volume" in df.columns:
        df["volume"] = df["Volume"]
    df = df.loc[:, ~df.columns.duplicated()]
    df["returns"] = df["Close"].pct_change()
    df["volatility"] = df["returns"].rolling(10).std()
    df["momentum"] = df["Close"].pct_change(5)
    df["trend"] = df["Close"] / df["Close"].rolling(20).mean() - 1
    df["drawdown"] = df["Close"] / df["Close"].cummax() - 1
    df["log_volume"] = np.log(df["volume"])
    df["volume_z"] = (
    df["log_volume"] - df["log_volume"].rolling(20).mean()
) / df["log_volume"].rolling(20).std()
    df["volume_momentum"] = df["log_volume"].diff(3)
    df["volume_lead"] = df["volume_z"].shift(1)

    features_df = df[["returns", "volatility", "momentum", "trend", "drawdown", "volume_lead"]].dropna()
    df = df.loc[features_df.index].copy()

    scaler = StandardScaler()
    features = scaler.fit_transform(features_df.values)

    if np.any(np.isnan(features)) or np.any(np.isinf(features)):
        raise ValueError(f"Features contain NaN/inf after scaling.")

    best_model, cov_type, bic_results = fit_best_hmm(features, n_states)

    if best_model is None:
        states = _fallback_regime(df, n_states)
        confidence = np.full(len(states), 0.5)
        return states, confidence, None, None, df

    full_bic = bic_results.get("full", {}).get("bic")
    diag_bic = bic_results.get("diag", {}).get("bic")

    print(f"HMM covariance selection: {cov_type} won "
        f"(full BIC={round(full_bic,1) if full_bic else 'N/A'}, "
        f"diag BIC={round(diag_bic,1) if diag_bic else 'N/A'})")

    if best_model is None:
        states = _fallback_regime(df, n_states)
        confidence = np.full(len(states), 0.5)
        return states, confidence, None, None, df

    raw_states = best_model.predict(features)
    probs = best_model.predict_proba(features)
    means = best_model.means_

    scores = np.array([float(means[i][0]) + float(means[i][3]) for i in range(n_states)])
    order = np.argsort(scores)
    state_map = {int(old): int(new) for new, old in enumerate(order)}
    states = np.array([state_map[int(s)] for s in raw_states])
    probs = probs[:, order]

    entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
    confidence = 1.0 - entropy / np.log(n_states)

    best_model._cov_type_selected = cov_type
    best_model._bic_results = {
        k: round(v["bic"], 2) for k, v in bic_results.items()
    }

    return states, confidence, best_model, scaler, df

def fit_best_hmm(features: np.ndarray, n_states: int = 3) -> tuple:
    results = {}
    
    for cov_type in ["full", "diag"]:
        best_score = -np.inf
        best_model = None
        
        for seed in range(40):  
            try:
                model = hmm.GaussianHMM(
                    n_components=n_states,
                    covariance_type=cov_type,
                    n_iter=200,
                    random_state=42 + seed  
                )
                model.fit(features)
                score = model.score(features)
                
                state_probs = model.predict_proba(features).mean(axis=0)
                if np.min(state_probs) < 0.03:
                    continue
                    
                if score > best_score and _is_valid(model, features, n_states):
                    best_score = score
                    best_model = model
            except Exception as e: 
                print(f"[HMM ERROR] {cov_type}: {e}")
                continue

        if best_model:
            n_params = {
                "full": n_states * (n_states - 1) + n_states * features.shape[1] + n_states * features.shape[1] ** 2,
                "diag": n_states * (n_states - 1) + n_states * features.shape[1] * 2,
            }
            bic = -2 * best_score + n_params[cov_type] * np.log(len(features))
            results[cov_type] = {"model": best_model, "bic": bic, "score": best_score}

    if len(results) == 0:
        return None, None, {}
    
    best_cov = min(results, key=lambda k: results[k]["bic"])
    winner = results[best_cov]["model"]
    
    return winner, best_cov, results


def _is_valid(model, features, n_states):
    """Reject degenerate solutions where one state has <3% of data."""
    states = model.predict(features)
    n = len(states)
    min_pct = 0.03 
    for i in range(n_states):
        if (states == i).sum() < min_pct * n:
            return False
    return True

def _fallback_regime(df, n_states=4):
    rolling_return = df["Close"].pct_change(20)
    rolling_vol    = df["Close"].pct_change().rolling(10).std()
    vol_threshold_high = rolling_vol.quantile(0.90)
    vol_threshold_mid  = rolling_vol.quantile(0.67)

    if n_states == 4:
        states = np.where(
            rolling_vol > vol_threshold_high, 0,     
            np.where(rolling_return < 0, 1,              
            np.where(rolling_vol > vol_threshold_mid, 2, 
            3))                                          
        )
    elif n_states == 3:
        states = np.where(
            rolling_return < 0, 0,
            np.where(rolling_vol > vol_threshold_mid, 1, 2)
        )
    else:
        states = (rolling_return >= 0).astype(int).values

    states = np.array(states, dtype=int)
    states[:20] = 1 if n_states <= 3 else 2
    print("Using fallback regime detection")
    return states

def get_recommendation_4state(state: int) -> dict:
    recs = {
        0: {
            "regime": "crash",
            "action": "Halt or minimum size only",
            "detail": "Extreme volatility spike - spreads are 2-3x normal. Any execution is costly. Wait for stabilisation.",
            "color":  "purple"
        },
        1: {
            "regime": "bearish",
            "action": "Use patient limit orders",
            "detail": "Price is falling - let it come to you. Split into smaller tranches, avoid urgency.",
            "color":  "red"
        },
        2: {
            "regime": "transitional",
            "action": "Reduce size, use limits",
            "detail": "High-volatility transitional regime - direction unclear. Minimise market impact, wait for confirmation.",
            "color":  "yellow"
        },
        3: {
            "regime": "bullish",
            "action": "Execute urgently",
            "detail": "Price is rising - delay costs you. Use market orders or aggressive limits.",
            "color":  "green"
        }
    }
    return recs.get(state, recs[2])


def get_recommendation(state: int) -> dict:
    recs = {
        0: {
            "regime": "bearish",
            "action": "Use patient limit orders",
            "detail": "Price is falling - let it come to you. Split into smaller tranches, avoid urgency.",
            "color": "red"
        },
        1: {
            "regime": "transitional",
            "action": "Reduce size, use limits",
            "detail": "High-volatility transitional regime - direction unclear. Minimise market impact, wait for confirmation.",
            "color": "yellow"
        },
        2: {
            "regime": "bullish",
            "action": "Execute urgently",
            "detail": "Price is rising - delay costs you. Use market orders or aggressive limits.",
            "color": "green"
        }
    }
    return recs.get(state, recs[1])

def smooth_regimes(states: np.ndarray, min_duration: int = 3) -> np.ndarray:
    """
    Suppress regime switches that last fewer than min_duration days.
    Prevents whipsawing on noisy days.
    """
    smoothed = states.copy()
    i = 0
    while i < len(states):
        j = i
        while j < len(states) and states[j] == states[i]:
            j += 1
        duration = j - i
        if duration < min_duration and i > 0:
            smoothed[i:j] = smoothed[i - 1]  # revert to previous regime
        i = j
    return smoothed

def walk_forward_regimes(df, n_states=4, train_window=252, step=21):
    """
    Rolling walk-forward: fit on 252 trading days, predict next 21.
    Returns OOS regime labels, confidence, entropy, and stay_prob — all from
    the same walk-forward model so they are internally consistent.
    Previously entropy/stay_prob were recomputed from a separate in-sample
    refit in data_collector.py, causing systematic mismatch with the OOS states.
    """
    all_states     = np.full(len(df), -1,      dtype=int)
    all_confidence = np.full(len(df), np.nan)
    all_entropy    = np.full(len(df), np.nan)
    all_stay_prob  = np.full(len(df), np.nan)

    for start in range(0, len(df) - train_window - step, step):
        train_end = start + train_window
        test_end  = min(train_end + step, len(df))

        train_df = df.iloc[start:train_end].copy()
        test_df  = df.iloc[train_end:test_end].copy()

        if "volume" not in test_df.columns and "Volume" in test_df.columns:
            test_df["volume"] = test_df["Volume"]

        try:
            _, _, model, scaler, _ = fit_hmm_4state(train_df, n_states)
            if model is None:
                continue

            test_df["returns"]    = test_df["Close"].pct_change()
            test_df["volatility"] = test_df["returns"].rolling(10).std()
            test_df["momentum"]   = test_df["Close"].pct_change(5)
            test_df["trend"]      = test_df["Close"] / test_df["Close"].rolling(20).mean() - 1
            test_df["drawdown"]   = test_df["Close"] / test_df["Close"].cummax() - 1
            test_df["log_volume"] = np.log(test_df["volume"])
            test_df["volume_z"]   = (
                test_df["log_volume"] - test_df["log_volume"].rolling(20).mean()
            ) / test_df["log_volume"].rolling(20).std()
            test_df["volume_lead"] = test_df["volume_z"].shift(1)

            test_features = test_df[
                ["returns", "volatility", "momentum", "trend", "drawdown", "volume_lead"]
            ].dropna()

            if len(test_features) == 0:
                continue

            scaled    = scaler.transform(test_features.values)
            raw       = model.predict(scaled)
            probs     = model.predict_proba(scaled)
            state_map = model._state_map
            states    = np.array([state_map[int(s)] for s in raw])

            idx = df.index.get_indexer(test_features.index)
            all_states[idx] = states

            entropy = -np.sum(probs * np.log(probs + 1e-8), axis=1)
            all_entropy[idx] = entropy
            stability = np.zeros(len(states))
            stability[1:] = (states[1:] == states[:-1]).astype(float)
            all_confidence[idx] = 0.7 * (1.0 - entropy / np.log(n_states)) + 0.3 * stability

            stay = np.array([float(model.transmat_[int(s), int(s)]) for s in raw])
            all_stay_prob[idx] = stay

        except Exception as e:
            print(f"Walk-forward failed at window {start}: {e}")
            continue

    return all_states, all_confidence, all_entropy, all_stay_prob

FEATURE_NAMES = ["returns", "volatility", "momentum", "trend", "drawdown", "volume_lead"]

def _emission_proba(model: GaussianHMM, x: np.ndarray) -> np.ndarray:
    """
    Compute P(state | x_t) ∝ P(x_t | state) * P(state)
    using emission probabilities scaled by stationary prior.
    This is the per-day, sequence-independent approximation -
    appropriate for SHAP since we want feature → regime attribution.
    """
    from scipy.stats import multivariate_normal
    n_states = model.n_components
    log_emission = np.zeros(n_states)

    for s in range(n_states):
        mean = model.means_[s]
        if model.covariance_type == "full":
            cov = model.covars_[s]
        else:
            cov = np.diag(model.covars_[s])
        log_emission[s] = multivariate_normal.logpdf(x, mean=mean, cov=cov)

    stationary = model.get_stationary_distribution()
    log_prior = np.log(stationary + 1e-12)
    log_posterior = log_emission + log_prior
    log_posterior -= log_posterior.max()
    posterior = np.exp(log_posterior)
    posterior /= posterior.sum()
    return posterior


def explain_regime_shap(
    model: GaussianHMM,
    scaler,
    features_df: pd.DataFrame,
    state_map: dict,
    n_background: int = 50,
    n_explain: int = 1,
) -> dict:
    """
    Run KernelSHAP on the HMM emission-based classifier.

    Args:
        model:        Fitted GaussianHMM from fit_hmm_4state
        scaler:       StandardScaler used during fit
        features_df:  Raw (unscaled) feature DataFrame - last n_explain rows are explained
        state_map:    model._state_map  {raw_state → ordered_state}
        n_background: Number of background samples for SHAP kernel
        n_explain:    How many recent days to explain (default 1 = today)

    Returns:
        dict with keys:
          - shap_values: shape [n_explain, n_features, n_states]
          - base_values: expected probs per state [n_states]
          - features_today: raw feature values for the explained row(s)
          - feature_names: list of 6 feature names
          - predicted_state: ordered state index (0-3)
          - state_probs: posterior probs for each ordered state
    """

    scaled = scaler.transform(features_df.values)

    ordered_states = [crash_raw, bear_raw, trans_raw, bull_raw] = [
        k for k, v in sorted(state_map.items(), key=lambda kv: kv[1])
    ]

    def predict_fn(X: np.ndarray) -> np.ndarray:
        out = np.zeros((len(X), 4))
        for i, x in enumerate(X):
            raw_probs = _emission_proba(model, x)
            out[i] = raw_probs[ordered_states]
        return out

    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(scaled), size=min(n_background, len(scaled)), replace=False)
    background = scaled[bg_idx]

    explainer = shap.KernelExplainer(predict_fn, background)

    explain_rows = scaled[-n_explain:]
    shap_vals = explainer.shap_values(explain_rows, nsamples=200, silent=True)
    shap_matrix = np.stack(shap_vals, axis=-1)

    base_values = explainer.expected_value

    state_probs = predict_fn(explain_rows[-1:] if explain_rows.ndim == 1 else explain_rows[-1:])[0]
    predicted_state = int(np.argmax(state_probs))

    raw_features_today = features_df.iloc[-n_explain:].values

    return {
        "shap_values": shap_matrix.tolist(),
        "base_values": np.array(base_values).tolist(), 
        "features_today": raw_features_today.tolist(), 
        "feature_names": FEATURE_NAMES,
        "predicted_state": predicted_state,
        "state_probs": state_probs.tolist(),   
        "state_labels": ["crash", "bearish", "transitional", "bullish"],
    }