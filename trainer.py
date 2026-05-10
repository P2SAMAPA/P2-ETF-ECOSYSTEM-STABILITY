"""
Main trainer: for each universe, test three windows, select best (max correlation with future drawdown),
compute current stability metrics, and save results.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import config
import data_manager
from stability_analyzer import EcosystemStability
import push_results

def compute_future_drawdown(returns, lookahead):
    """Drawdown over next `lookahead` days: (min price / current price) - 1."""
    # returns is a 1D array of log returns
    cum_ret = np.exp(np.cumsum(returns))
    future_min = np.minimum.accumulate(cum_ret)
    drawdown = (future_min - cum_ret) / cum_ret
    return drawdown

def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set")
        return

    df = data_manager.load_master_data()
    all_results = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} ===")
        returns_wide = data_manager.prepare_returns_matrix(df, tickers)
        if returns_wide.empty:
            continue

        # Prepare full returns matrix as numpy (dates x tickers)
        all_returns = returns_wide.values
        # For each window, compute rolling stability and predictive correlation
        best_window = None
        best_corr = -np.inf
        best_metrics = None

        for win in config.WINDOWS:
            print(f"  Testing window = {win} days")
            # We'll compute λ_max for each end date available
            # Then correlate with future drawdown computed from the index (equal‑weighted portfolio)
            # For simplicity, use the average of all ETFs as the "market" for drawdown
            # But we need a future series for each time point.
            # Instead, compute λ_max time series and drawdown time series over the same dates.
            lambda_series = []
            drawdown_series = []
            for i in range(win, len(all_returns) - config.LOOKAHEAD):
                Y = all_returns[i-win:i]
                es = EcosystemStability(window=win, var_lag=config.VAR_LAG)
                if es.fit(Y):
                    lambda_series.append(es.lambda_max_)
                    # Compute future drawdown over next LOOKAHEAD days for the equal‑weighted portfolio
                    port_returns = np.mean(all_returns[i:i+config.LOOKAHEAD], axis=1)
                    cum = np.exp(np.cumsum(port_returns))
                    dd = (np.minimum.accumulate(cum) - cum) / cum
                    future_max_dd = np.min(dd)   # most negative drawdown
                    drawdown_series.append(future_max_dd)
            if len(lambda_series) < 10:
                continue
            # Correlation between λ_max and subsequent drawdown (should be negative: higher λ => deeper drawdown)
            corr = np.corrcoef(lambda_series, drawdown_series)[0,1]
            print(f"    Correlation with {config.LOOKAHEAD}-day drawdown: {corr:.3f}")
            if abs(corr) > best_corr:
                best_corr = abs(corr)
                best_window = win
                # Store the last available stability object for current output
                # Re‑fit on last full window for final metrics
                final_es = EcosystemStability(window=win, var_lag=config.VAR_LAG)
                final_es.fit(all_returns[-win:])
                best_metrics = final_es

        if best_window is None:
            print(f"  No valid window for {universe_name}")
            continue

        print(f"  Selected window: {best_window} days (|corr|={best_corr:.3f})")
        es = best_metrics

        # Prepare output
        status = es.get_stability_status()
        destabilizing = es.get_destabilizing_etfs(tickers)
        interaction_strength = es.get_interaction_strength()
        effective_diversity = es.get_effective_diversity()

        universe_results = {
            "selected_window": best_window,
            "stability_status": status,
            "lambda_max": float(es.lambda_max_),
            "interaction_strength": float(interaction_strength),
            "effective_diversity": float(effective_diversity),
            "destabilizing_etfs": destabilizing,
            "all_tickers": {t: {"destabilizing_impact": float(imp)} for t, imp in zip(tickers, es.destabilizing_)} if es.destabilizing_ is not None else {}
        }
        all_results[universe_name] = universe_results

    Path("results").mkdir(exist_ok=True)
    local_path = Path(f"results/stability_{config.TODAY}.json")
    with open(local_path, "w") as f:
        json.dump({"run_date": config.TODAY, "universes": all_results}, f, indent=2)

    push_results.push_daily_result(local_path)
    print("\n=== Ecosystem stability analysis complete ===")

if __name__ == "__main__":
    main()
