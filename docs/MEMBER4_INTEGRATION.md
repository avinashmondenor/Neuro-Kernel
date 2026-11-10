# Member 4 — integration with the current NeuroKernel repository

This patch targets the repository's actual layout and existing live-monitor schema:

- `src/live_monitor.py` returns dictionaries containing `pid`, `process_name`,
  the ten trained model features, `prediction`, `status`, and
  `anomaly_probability`.
- `src/dashboard.py` calls `monitor.scan()` and displays the resulting records.
- `src/train_model.py` reads `data/X_train.csv` and `data/y_train.csv`;
  the target column is `label`, with `0 = Normal` and `1 = Anomaly`.

## Files

- `src/anomaly_investigation.py`: robust normal-baseline comparisons and
  CPU / Memory / Disk / Process activity attribution.
- `src/member4_integration.py`: adapter that builds the baseline from the
  existing training CSVs and enriches live records.
- `tests/test_anomaly_investigation.py`: synthetic unit tests.

## Integrate into `src/dashboard.py`

1. Add this import after the existing `from live_monitor import LiveProcessMonitor`:

```python
from member4_integration import Member4Investigator
```

2. Immediately after `monitor = get_monitor()` has been initialized successfully,
   create the investigator once:

```python
@st.cache_resource
def get_investigator():
    return Member4Investigator(PROJECT_DIR)

try:
    investigator = get_investigator()
except Exception as exc:
    st.error("Could not initialize Member 4 anomaly investigation.")
    st.exception(exc)
    st.stop()
```

3. In `live_dashboard()`, immediately after:

```python
records = monitor.scan()
```

add:

```python
records = investigator.investigate_many(records)
```

4. Add these columns to the `display_columns` list:

```python
"anomaly_type",
"investigation_explanation",
```

5. Under the selected process details, show the investigation:

```python
st.write("**Likely resource category:**", process.get("anomaly_type", "Unknown"))
st.info(process.get("investigation_explanation", "No investigation explanation available."))
```

For a process marked Normal, the explanation will say it was classified Normal.
For a model anomaly without a matching high-side baseline outlier, the category
will be `Unclassified`; the code deliberately does not invent a cause.

## Run and verify on Ubuntu

From the repository root, with the project's virtual environment activated:

```bash
pip install -r requirements.txt
python -m py_compile src/anomaly_investigation.py src/member4_integration.py
python tests/test_anomaly_investigation.py
streamlit run src/dashboard.py
```

Then verify:
- The dashboard still scans and lists processes.
- The first sample is still `Warming up`; investigation must not turn it into
  a false prediction.
- After the next sample, normal/anomaly statuses still come from the existing
  Random Forest.
- An anomalous process shows CPU, Memory, Disk, or Process activity only when
  the relevant feature exceeds its normal-only reference.
- If no resource family is isolated, it displays `Unclassified`.
- Refresh, filtering, and process detail selection still work.

## Important interpretation

The baseline is fitted from label-0 rows in `X_train.csv` only. The model is
unchanged. The investigator is a transparent heuristic explanation layer,
not a causal model or a malicious-process detector. High resource use is not
proof that a process is malicious. Disk fields must remain bytes-per-second,
matching the live monitor's rate calculation.

## Publishing limitation

The connected GitHub identity currently has read-only (`push: false`) access
to `avinashmondenor/Neuro-Kernel`. These files can be tested locally, but a
branch, commit, or pull request cannot be published until an account with
write access pushes the patch or grants the connected identity write access.

## Dashboard integration status in this patch

The supplied `src/dashboard.py` already contains the import, cached investigator initialization, record enrichment, table columns, selected-process explanation, and interpretation notes described above. Replace the repository's existing `src/dashboard.py` with this integrated version after backing up any unrelated local dashboard edits.
