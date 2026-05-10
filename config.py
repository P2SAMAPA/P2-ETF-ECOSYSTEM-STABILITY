"""
Configuration for P2-ETF-ECOSYSTEM-STABILITY engine.
"""

import os
from datetime import datetime

# --- Hugging Face ---
DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
DATA_FILE = "master_data.parquet"
OUTPUT_REPO = "P2SAMAPA/p2-etf-ecosystem-stability-results"

# --- Universe definitions ---
FI_COMMODITIES = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU",
    "GDX", "XME", "IWF", "XSD", "XBI", "IWM", "IWD", "IWO"
]
COMBINED = list(set(FI_COMMODITIES + EQUITY_SECTORS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES,
    "EQUITY_SECTORS": EQUITY_SECTORS,
    "COMBINED": COMBINED
}

# --- Stability parameters ---
WINDOWS = [60, 120, 252]                     # rolling estimation windows (days)
VAR_LAG = 1                                  # VAR order (use daily lag)
LOOKAHEAD = 5                                # days ahead to compute drawdown for window selection
EIGENVALUE_THRESHOLD = 0.0                   # positive real part = unstable
TOP_N_DESTAB = 3                             # number of most destabilising ETFs
N_BINS = 20                                  # for interaction strength histogram (optional)

# --- Output ---
TODAY = datetime.now().strftime("%Y-%m-%d")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
