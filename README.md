# Signature Granger Causality for ETFs

Goes beyond linear Granger and transfer entropy. The signature of a path encodes all iterated integrals. Tests whether the signature of series X up to time t contains information about Y's future that signature of Y doesn't – a fully nonlinear, path-dependent causality test. The per‑ETF score measures macro → ETF causality strength.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- Truncated signature (up to depth 4)
- Joint signature for interaction terms
- Ridge regression for restricted/full models
- Score = R² improvement (Granger causality strength)
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-signature-granger-causality-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt`
3. Run training: `python train.py` (fast)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High signature GC score → macro strongly causes ETF returns.
- Low score → macro does not cause ETF returns.

## Requirements

See `requirements.txt`.
