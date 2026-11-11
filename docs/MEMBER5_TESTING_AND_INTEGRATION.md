# Member 5 — Testing & Final Integration

**Contributor:** Harshita Meena  
**Role:** Member 5 — Testing & Final Integration

## Scope completed

This contribution adds reproducible validation around the existing NeuroKernel pipeline without changing or retraining the Random Forest model.

- Evaluates the checked-in model against the held-out `X_test.csv` / `y_test.csv`.
- Reports accuracy, class precision/recall/F1, confusion matrix, false positives and false negatives.
- Runs a two-scan live monitor smoke test (the monitor needs two samples to calculate CPU and disk rates).
- Passes live records through Member 4's investigator and checks that the expected explanation fields are present.
- Adds bounded CPU, memory and disk workload generators for manual dashboard demonstrations.
- Adds regression tests for warm-up records, missing telemetry and separation of ML predictions from heuristic resource explanations.

## Reproducible setup

Use Linux/Ubuntu for the `/proc`-based process monitor. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run automated tests and validation

```bash
python -m pytest -q
python src/member5_validation.py
```

The validation command writes a timestamped machine-readable report to:

```text
logs/member5_validation.json
```

The held-out confusion matrix uses this label order:

```text
                 Predicted Normal   Predicted Anomaly
Actual Normal          TN                  FP
Actual Anomaly         FN                  TP
```

- **False positive (FP):** an actually normal held-out row predicted as anomalous.
- **False negative (FN):** an actually anomalous held-out row predicted as normal.

## Controlled workload demonstration

Start the dashboard in a second terminal:

```bash
streamlit run src/dashboard.py
```

After the dashboard has started, use another terminal to run one workload at a time:

```bash
python scripts/run_test_workload.py cpu --seconds 15
python scripts/run_test_workload.py memory --seconds 15 --memory-mb 96
python scripts/run_test_workload.py disk --seconds 15 --disk-mb 32
```

The workload script is bounded (maximum 60 seconds, 256 MiB memory allocation and 128 MiB temporary disk write). It removes its temporary disk file automatically. Run each scenario separately and allow the monitor to collect at least two samples.

**Important:** These workloads are controlled stress stimuli, not ground-truth labelled samples from the model's training distribution. Record the telemetry and model response as an observed demonstration. Do not claim that the model must classify every generated workload as an anomaly. A CPU/memory/disk explanation is produced by Member 4's separate normal-baseline heuristic, not by the binary Random Forest.

## End-to-end checks

| Check | Expected behaviour |
|---|---|
| Normal everyday process | Model result remains `Normal` when prediction is 0 |
| First scan / new process | `Warming up`; no prediction until a second sample |
| CPU workload | CPU telemetry rises if the OS exposes it; model result is recorded, not assumed |
| Memory workload | RSS / memory percentage rises for the workload process |
| Disk workload | Temporary file I/O is generated; per-process I/O may be restricted by OS permissions |
| Missing telemetry | Unavailable fields do not crash the investigation layer |
| Process exits during scan | Monitor skips inaccessible/terminated process safely |
| Dashboard + investigator | Process row retains model status and includes Member 4 explanation fields |
| FP/FN | Counted only on held-out data where labels are available |

## Measured results

Validation was executed against the model and held-out split included in this ZIP. The saved JSON report (`logs/member5_validation.json`) is the detailed run record.

| Metric | Measured result |
|---|---:|
| Held-out samples | 68,551 |
| Accuracy | 74.72% |
| Normal precision | 65.42% |
| Normal recall | 73.75% |
| Anomaly precision | 81.94% |
| Anomaly recall | 75.34% |
| Anomaly F1 | 78.50% |
| True negatives | 19,590 |
| False positives | 6,973 |
| False negatives | 10,355 |
| True positives | 31,633 |

Confusion matrix (`[[TN, FP], [FN, TP]]`):

```text
[[19590,  6973],
 [10355, 31633]]
```

Live integration smoke test: 18 processes were returned in each of two scans; all 18 had predictions on the second scan, and all required Member 4 integration fields were present. This validates data-flow integration only, not live prediction correctness.

**Model compatibility warning:** the saved Random Forest was serialized with scikit-learn 1.9.1, while the validation environment used 1.8.0. Scikit-learn emitted `InconsistentVersionWarning`. The evaluation completed, but for a reproducible supported run install the model's training version (`scikit-learn==1.9.1`) before evaluating or running the dashboard. Do not silently treat cross-version execution as warning-free.

The live smoke test has no independent labels for currently running applications. Therefore it reports process counts, ready predictions and field integration, but it does **not** label live predictions as true/false positives or negatives.

## Interpretation and limitations

1. The ML model predicts only **Normal (0)** or **Anomaly (1)**.
2. CPU / Memory / Disk attribution is a separate heuristic investigation layer.
3. High resource use is not proof of malware or malicious intent.
4. A high anomaly probability is a model score, not a calibrated security risk unless separately calibrated.
5. Process telemetry can be unavailable due to permissions, process termination, platform differences or kernel restrictions.
6. The live monitor depends on two samples to calculate rate-based CPU and disk features.
7. Controlled test workloads are not equivalent to labelled production incidents.

## Final demo checklist

- [ ] Install dependencies in a clean virtual environment.
- [ ] Run `python -m pytest -q`.
- [ ] Run `python src/member5_validation.py` and retain `logs/member5_validation.json`.
- [ ] Launch `streamlit run src/dashboard.py`.
- [ ] Show normal process rows and warming-up behaviour.
- [ ] Run CPU, memory and disk workloads separately.
- [ ] Select a process and show Member 4's explanation and resource evidence.
- [ ] Explain FP/FN from the held-out confusion matrix.
- [ ] State limitations and distinguish ML predictions from heuristic explanations.
