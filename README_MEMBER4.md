# NeuroKernel — Member 4 integrated patch

This patch adds Member 4's resource investigation layer and includes the updated `src/dashboard.py` with the integration already applied.

## Included
- `src/anomaly_investigation.py` — normal-baseline comparisons and resource attribution.
- `src/member4_integration.py` — adapter using the repository's existing `data/X_train.csv` and `data/y_train.csv`.
- `src/dashboard.py` — updated dashboard showing likely resource category, explanation, and resource evidence.
- `tests/test_anomaly_investigation.py` — unit tests.
- `docs/MEMBER4_INTEGRATION.md` — integration and verification notes.

Copy these files into the matching locations in the existing Neuro-Kernel repository. The dashboard integration is already applied in this version of `dashboard.py`.

Run from the repository root:

```bash
python -m py_compile src/anomaly_investigation.py src/member4_integration.py src/dashboard.py
python tests/test_anomaly_investigation.py
streamlit run src/dashboard.py
```

The investigation is a heuristic explanation layer, not causal proof or a malicious-process detector. The Random Forest model and its predictions are unchanged.
