# P2-ETF-ECOSYSTEM-STABILITY

**Community matrix stability analysis** using May's Diversity‑Stability Theorem from ecology.  
Models ETF interactions as a VAR(1) and computes the largest real eigenvalue (λ_max) – positive values indicate structural instability, a possible early warning of market crashes.

## Features

- Estimates community matrix **J** via vector autoregression (VAR(1)).
- Computes eigenvalues of **J**; λ_max measures stability.
- Tests three rolling windows (60, 120, 252 days) and selects the one maximising correlation with future drawdown.
- Identifies **destabilising ETFs** via left eigenvector of the largest eigenvalue.
- Outputs stability status, effective diversity (Shannon entropy of interactions), and per‑universe warnings.

## Data

Uses `P2SAMAPA/fi-etf-macro-signal-master-data`.  
Results stored in `P2SAMAPA/p2-etf-ecosystem-stability-results`.

## Installation

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-ECOSYSTEM-STABILITY.git
cd P2-ETF-ECOSYSTEM-STABILITY
pip install -r requirements.txt
References
May, R. M. (1972). Will a large complex system be stable?

Ives, A. R. (1995). Measuring resilience in stochastic systems.

Fernholz, R. (2002). Stochastic portfolio theory (diversity).
