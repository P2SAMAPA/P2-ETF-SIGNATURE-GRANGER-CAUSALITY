import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def compute_composite_macro_factor(macro_df):
    """Compute composite macro factor from all macro variables."""
    if len(macro_df) < 2:
        return np.ones(len(macro_df)) * 0.5
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    factor = pca.fit_transform(macro_scaled).flatten()
    factor = (factor - factor.min()) / (factor.max() - factor.min() + 1e-8)
    return factor

def signature(series, depth=4):
    """
    Compute the truncated signature of a 1D series up to depth.
    Returns a vector of signature features.
    """
    if len(series) < 2:
        return np.zeros(depth)
    increments = np.diff(series)
    sig = []
    # Level 1: sum of increments
    sig.append(np.sum(increments))
    if depth >= 2:
        sum_inc = np.sum(increments)
        sum_sq = np.sum(increments**2)
        sig.append(0.5 * (sum_inc**2 - sum_sq))
    if depth >= 3:
        sig.append((np.sum(increments)**3) / 6.0)
    if depth >= 4:
        sig.append((np.sum(increments)**4) / 24.0)
    if depth >= 5:
        sig.append((np.sum(increments)**5) / 120.0)
    return np.array(sig)

def joint_signature(series1, series2, depth=4):
    """
    Compute the joint signature of two series (interleaved).
    This captures the interaction between the two paths.
    """
    if len(series1) < 2 or len(series2) < 2:
        return np.zeros(depth * 2)
    # Interleave the two series
    min_len = min(len(series1), len(series2))
    interleaved = np.zeros(min_len * 2)
    interleaved[0::2] = series1[:min_len]
    interleaved[1::2] = series2[:min_len]
    return signature(interleaved, depth * 2)

def signature_granger_test(X, Y, depth=4, max_lag=5):
    """
    Test whether X Granger-causes Y using path signatures.
    Returns the Granger causality score (F-statistic-like).
    """
    if len(X) < max_lag + 2 or len(Y) < max_lag + 2:
        return 0.0
    # Prepare data: for each time t, use lagged signatures
    X_sigs = []
    Y_sigs = []
    joint_sigs = []
    Y_future = []
    for t in range(max_lag, len(Y) - 1):
        # Signature of X up to lag
        x_seg = X[t-max_lag:t+1]
        x_sig = signature(x_seg, depth)
        # Signature of Y up to lag
        y_seg = Y[t-max_lag:t+1]
        y_sig = signature(y_seg, depth)
        # Joint signature (interaction)
        joint_sig = joint_signature(x_seg, y_seg, depth)
        X_sigs.append(x_sig)
        Y_sigs.append(y_sig)
        joint_sigs.append(joint_sig)
        Y_future.append(Y[t+1])
    X_sigs = np.array(X_sigs)
    Y_sigs = np.array(Y_sigs)
    joint_sigs = np.array(joint_sigs)
    Y_future = np.array(Y_future)
    if len(Y_future) < 10:
        return 0.0
    # Standardise
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sigs)
    Y_scaled = scaler.fit_transform(Y_sigs)
    joint_scaled = scaler.fit_transform(joint_sigs)
    # Model 1: Y_future ~ signature(Y) (restricted model)
    ridge_y = Ridge(alpha=1.0)
    ridge_y.fit(Y_scaled, Y_future)
    pred_y = ridge_y.predict(Y_scaled)
    # Model 2: Y_future ~ signature(Y) + signature(X) + joint signature (full model)
    full_features = np.concatenate([Y_scaled, X_scaled, joint_scaled], axis=1)
    ridge_full = Ridge(alpha=1.0)
    ridge_full.fit(full_features, Y_future)
    pred_full = ridge_full.predict(full_features)
    # Granger causality score: improvement in R²
    ss_res_y = np.sum((Y_future - pred_y)**2)
    ss_res_full = np.sum((Y_future - pred_full)**2)
    ss_tot = np.sum((Y_future - np.mean(Y_future))**2)
    if ss_tot == 0:
        return 0.0
    r2_y = 1 - ss_res_y / ss_tot
    r2_full = 1 - ss_res_full / ss_tot
    improvement = max(0.0, r2_full - r2_y)
    return improvement

def signature_granger_score(returns, macro_df, depth=4, max_lag=5):
    """
    Compute per-ETF signature Granger causality score.
    Higher score = the ETF Granger-causes other ETFs.
    """
    if len(returns) < max_lag + 10 or macro_df is None or len(macro_df) < max_lag + 10:
        return 0.0
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < max_lag + 10:
        return 0.0
    # Compute macro factor
    macro_factor = compute_composite_macro_factor(macro_df)[-1]
    # Compute Granger causality from macro factor to returns
    # This tells us how much macro causes the ETF
    gc_score = signature_granger_test(macro_factor, returns, depth, max_lag)
    return float(gc_score)
