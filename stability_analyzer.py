"""
Ecosystem stability using community matrix (VAR(1)) and eigenvalue analysis.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.linalg import eig
import warnings
import config

class EcosystemStability:
    def __init__(self, window=252, var_lag=1):
        self.window = window
        self.var_lag = var_lag
        self.J_ = None
        self.eigenvalues_ = None
        self.lambda_max_ = None
        self.destabilizing_ = None

    def fit(self, returns):
        """
        returns: (T, n) numpy array of log returns (T >= window)
        Estimate community matrix J via OLS: r_{t+1} = J * r_t + c
        """
        if returns.shape[0] < self.window:
            return False
        Y = returns[-self.window:]  # (window, n)
        n = Y.shape[1]
        # Build lagged matrix X = r_t
        X = Y[:-1]   # (window-1, n)
        y = Y[1:]    # (window-1, n)
        # Estimate J column by column (each asset's next return as function of all current returns)
        J_est = np.zeros((n, n))
        for j in range(n):
            reg = LinearRegression(fit_intercept=True)
            reg.fit(X, y[:, j])
            J_est[:, j] = reg.coef_   # shape (n,)
        self.J_ = J_est
        # Compute eigenvalues and left eigenvectors
        w, vl = eig(self.J_, left=True, right=False)   # w = eigenvalues, vl = left eigenvectors
        self.eigenvalues_ = w
        self.lambda_max_ = np.max(w.real)
        # Get left eigenvector for the largest eigenvalue
        idx = np.argmax(w.real)
        left_eig = vl[:, idx]
        left_eig = np.abs(left_eig) / (np.sum(np.abs(left_eig)) + 1e-12)
        self.destabilizing_ = left_eig
        return True

    def get_stability_status(self):
        if self.lambda_max_ is None:
            return "unknown"
        return "unstable" if self.lambda_max_ > config.EIGENVALUE_THRESHOLD else "stable"

    def get_destabilizing_etfs(self, tickers):
        if self.destabilizing_ is None:
            return []
        scores = list(zip(tickers, self.destabilizing_))
        scores.sort(key=lambda x: -x[1])
        return [{"ticker": t, "impact": float(s)} for t, s in scores[:config.TOP_N_DESTAB]]

    def get_interaction_strength(self):
        if self.J_ is None:
            return np.nan
        off_diag = self.J_.copy()
        np.fill_diagonal(off_diag, 0)
        return np.mean(np.abs(off_diag))

    def get_effective_diversity(self):
        if self.J_ is None:
            return np.nan
        off_diag = np.abs(self.J_.flatten())
        off_diag = off_diag / (off_diag.sum() + 1e-12)
        entropy = -np.sum(off_diag * np.log(off_diag + 1e-12))
        return np.exp(entropy)
