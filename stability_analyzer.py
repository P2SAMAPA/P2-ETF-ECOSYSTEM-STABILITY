"""
Ecosystem stability using community matrix (VAR(1)) and eigenvalue analysis.
"""

import numpy as np
from statsmodels.tsa.api import VAR
from scipy.linalg import eig
from sklearn.linear_model import LinearRegression
import warnings

class EcosystemStability:
    def __init__(self, window=252, var_lag=1):
        self.window = window
        self.var_lag = var_lag
        self.J_ = None            # community matrix (n_assets x n_assets)
        self.eigenvalues_ = None
        self.lambda_max_ = None
        self.destabilizing_ = None

    def fit(self, returns):
        """
        returns: (T, n) numpy array of log returns (T >= window)
        """
        if returns.shape[0] < self.window:
            return False
        # Use last `window` observations
        Y = returns[-self.window:]
        # Fit VAR(1) model
        try:
            model = VAR(Y)
            results = model.fit(maxlags=self.var_lag, trend='c')
            # Extract coefficient matrix for lag 1 (shape: n x n)
            # results.coefs is a list of length lag, each (n x n)
            self.J_ = results.coefs[0].T   # transpose to have J such that r_{t+1} = J * r_t + c
            # Compute eigenvalues
            self.eigenvalues_ = eig(self.J_)[0]
            self.lambda_max_ = np.max(self.eigenvalues_.real)
            # Destabilizing contribution: eigenvector of the largest eigenvalue
            # The eigenvector v (right) gives sensitivity of the eigenvalue to J_ij changes
            # We compute a score per asset: sum_{j} |v_j| * |J_{.,j}| (influence on eigenvalue)
            # Simpler: use the left eigenvector of the largest eigenvalue as "impact"
            _, v, _ = eig(self.J_, left=True, right=False)
            idx = np.argmax(self.eigenvalues_.real)
            left_eig = v[:, idx]   # left eigenvector
            # Normalise
            left_eig = np.abs(left_eig) / np.sum(np.abs(left_eig))
            self.destabilizing_ = left_eig
            return True
        except Exception as e:
            warnings.warn(f"VAR fitting failed: {e}")
            return False

    def get_stability_status(self):
        if self.lambda_max_ is None:
            return "unknown"
        return "unstable" if self.lambda_max_ > config.EIGENVALUE_THRESHOLD else "stable"

    def get_destabilizing_etfs(self, tickers):
        if self.destabilizing_ is None:
            return []
        # Return top assets by left eigenvector magnitude
        scores = list(zip(tickers, self.destabilizing_))
        scores.sort(key=lambda x: -x[1])
        return [{"ticker": t, "impact": float(s)} for t, s in scores[:config.TOP_N_DESTAB]]

    def get_interaction_strength(self):
        """Mean absolute off‑diagonal of J (connectivity strength)."""
        if self.J_ is None:
            return np.nan
        off_diag = self.J_.copy()
        np.fill_diagonal(off_diag, 0)
        return np.mean(np.abs(off_diag))

    def get_effective_diversity(self, weights=None):
        """Shannon entropy of interaction strengths (or equal if no weights)."""
        if self.J_ is None:
            return np.nan
        off_diag = np.abs(self.J_.flatten())
        off_diag = off_diag / (off_diag.sum() + 1e-12)
        # Shannon entropy
        entropy = -np.sum(off_diag * np.log(off_diag + 1e-12))
        return np.exp(entropy)   # effective number of "interactions"
