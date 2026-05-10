"""
Ecosystem stability using community matrix (VAR(1) via OLS per row) and eigenvalue analysis.
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.linalg import eig
import config
import warnings

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
        """
        if returns.shape[0] < self.window:
            return False
        Y = returns[-self.window:]   # use last window
        n = Y.shape[1]
        # Build X = lagged returns (t) and y = returns at t+1
        X = Y[:-1]
        y = Y[1:]
        # If n is too large, ensure enough samples
        if X.shape[0] < n + 1:
            return False
        J = np.zeros((n, n))
        # For each asset i, regress y[:,i] on X (all assets' lagged returns)
        for i in range(n):
            reg = LinearRegression(fit_intercept=False)  # we don't include intercept? Could add but then J includes constant? We'll use no intercept to keep pure interaction.
            reg.fit(X, y[:, i])
            J[i, :] = reg.coef_
        self.J_ = J
        # Compute eigenvalues
        self.eigenvalues_ = eig(self.J_)[0]
        self.lambda_max_ = np.max(self.eigenvalues_.real)
        # Left eigenvector of largest eigenvalue
        _, v, _ = eig(self.J_, left=True, right=False)
        idx = np.argmax(self.eigenvalues_.real)
        left_eig = v[:, idx]
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
