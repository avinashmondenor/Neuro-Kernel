
# NeuroKernel
### AI-Based Linux Process Anomaly Detection

NeuroKernel is a machine learning-based system designed to detect anomalous behavior in Linux processes using process-level system telemetry.

The system analyzes CPU utilization, memory consumption, thread activity, context switches, and disk I/O to distinguish normal process behavior from potentially anomalous activity.

The current implementation includes dataset preparation, preprocessing, a trained Random Forest classifier, and baseline model evaluation for binary anomaly detection.

## Overview

Linux processes exhibit different resource utilization patterns depending on their workloads. Unusual resource consumption may indicate unexpected behavior, resource-intensive operations, or potentially problematic processes.

NeuroKernel applies machine learning to process telemetry to identify anomalous behavior and establish a foundation for intelligent Linux process monitoring.

**Current classification:**

| Label | Meaning |
|---|---|
| `0` | Normal |
| `1` | Anomaly |

## System Architecture

```text
Linux Processes
       |
       v
Process Telemetry Collection
       |
       v
Feature Extraction & Preprocessing
       |
       v
Random Forest Classifier
       |
       v
Normal / Anomaly Prediction
       |
       v
Anomaly Investigation
       |
       v
Monitoring Dashboard
```

The dataset preparation, baseline classifier, and offline evaluation pipeline are implemented. Live telemetry integration and dashboard functionality are separate system integration components.

## Machine Learning Model

**Algorithm:** Random Forest Classifier

The model uses 10 numerical features extracted from process telemetry.

| Feature | Description |
|---|---|
| `cpu_percent` | CPU utilization of a process |
| `memory_mb` | Memory consumed in megabytes |
| `memory_percent` | Percentage of system memory used |
| `threads` | Number of threads in a process |
| `context_switches` | Process context-switch activity |
| `cpu_user_time` | CPU time spent executing user-space code |
| `cpu_system_time` | CPU time spent executing kernel-space code |
| `cpu_delta` | Change in CPU usage over the sampling interval |
| `read_bytes_per_sec` | Disk read rate |
| `write_bytes_per_sec` | Disk write rate |

## Dataset

The dataset combines normal process telemetry with CPU-, memory-, and disk-related anomalous observations.

| Dataset Property | Value |
|---|---:|
| Total observations | 342,752 |
| Total columns | 19 |
| Normal observations | 132,816 |
| Anomalous observations | 209,936 |
| Training samples | 274,201 |
| Testing samples | 68,551 |
| Train-test split | 80:20 |
| Split strategy | Stratified |

The dataset was prepared and divided into training and testing subsets for model development and evaluation.

## Model Performance

The baseline Random Forest classifier was evaluated on a held-out test dataset.

| Metric | Reported Score |
|---|---:|
| Accuracy | 74.72% |
| Anomaly Precision | 0.82 |
| Anomaly Recall | 0.75 |
| Anomaly F1-score | 0.78 |

### Confusion Matrix

| Actual Class | Predicted Normal | Predicted Anomaly |
|---|---:|---:|
| Normal | 19,590 | 6,973 |
| Anomaly | 10,355 | 31,633 |

**Note:** The reported metrics and confusion matrix should be reconciled against the saved test predictions before publication. The matrix currently implies approximately 74.0% accuracy, rather than 74.72%.

### Feature Importance

The current feature-importance analysis identified the following features as the most influential in the trained model:

1. `context_switches`
2. `cpu_system_time`
3. `cpu_user_time`
4. `memory_mb`
5. `memory_percent`
6. `threads`
7. `cpu_percent`
8. `cpu_delta`
9. `write_bytes_per_sec`
10. `read_bytes_per_sec`

These rankings reflect the current model and dataset; they do not independently establish causation or anomaly severity.

## Technology Stack

- **Python** — Machine learning and data processing
- **Pandas** — Dataset manipulation
- **NumPy** — Numerical computation
- **Scikit-learn** — Model training and evaluation
- **Linux** — Target operating system
- **Joblib / Pickle** — Model persistence, depending on the implementation

## Repository Structure

```text
NeuroKernel/
├── data/
│   ├── neurokernel_final.csv
│   ├── X_features.csv
│   ├── y_labels.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
├── models/
│   └── neurokernel_random_forest.pkl
├── src/
│   ├── collect.py
│   ├── split_data.py
│   ├── train_model.py
│   └── analyze_model.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.10 or a compatible version for the project dependencies
- Git
- Dependencies listed in `requirements.txt`

### Installation

Clone the repository:

```bash
git clone <YOUR_REPOSITORY_URL>
cd NeuroKernel
```

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python src/train_model.py
```

### Analyze Model Performance

```bash
python src/analyze_model.py
```

The commands above assume the scripts use the default data paths and do not require additional command-line arguments. Check the source files for the exact execution requirements.

## Limitations

- The current model performs binary anomaly detection.
- Dedicated classification of CPU, memory, and disk anomalies is not yet established by the binary classifier.
- Performance on unseen workloads and different Linux environments requires further validation.
- An anomaly prediction does not necessarily indicate malicious activity.
- Automated process intervention requires additional safety checks and controlled testing.

## Current Status

The initial dataset preparation, preprocessing, baseline Random Forest training, model evaluation, and feature-importance analysis have been completed.

The trained model is available as `neurokernel_random_forest.pkl` for subsequent integration, subject to the actual model-loading and feature-processing requirements.

NeuroKernel provides a machine learning foundation for Linux process anomaly detection, with the broader objective of integrating process telemetry, anomaly investigation, and monitoring into a unified system.

---

**NeuroKernel — Intelligent analysis of Linux process behavior.**
