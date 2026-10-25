# NeuroKernel

## AI-Based Linux Process Anomaly Detection

NeuroKernel is an AI-based system designed to monitor Linux processes and identify abnormal process behavior using system telemetry and machine learning.

The current implementation focuses on binary anomaly detection:

- `0` → Normal
- `1` → Anomaly

## Current Progress

The initial data collection, preprocessing, validation and baseline machine learning model have been completed.

### Dataset

Final dataset:

- 342,752 observations
- 19 columns
- 132,816 normal observations
- 209,936 anomalous observations

Anomaly categories collected:

- CPU
- Memory
- Disk

The dataset contains process-level telemetry such as:

- CPU usage
- Memory usage
- Number of threads
- Context switches
- CPU user/system time
- CPU delta
- Disk read rate
- Disk write rate

## Machine Learning Features

The current Random Forest model uses 10 numerical features:

1. cpu_percent
2. memory_mb
3. memory_percent
4. threads
5. context_switches
6. cpu_user_time
7. cpu_system_time
8. cpu_delta
9. read_bytes_per_sec
10. write_bytes_per_sec

## Dataset Split

The dataset was divided using an 80/20 stratified train-test split.

Training:

274,201 observations

Testing:

68,551 observations

## Current Machine Learning Model

Algorithm:

**Random Forest Classifier**

The current model performs binary classification:

```text
Normal → 0
Anomaly → 1

### Baseline Model Performance

The current Random Forest model achieves approximately **75% accuracy** on the held-out test dataset.

- Accuracy: **74.72%**
- Anomaly precision: **0.82**
- Anomaly recall: **0.75**
- Anomaly F1-score: **0.78**

Confusion Matrix:

```text
                 Predicted
                 Normal  Anomaly

Actual Normal     19590    6973
Actual Anomaly    10355   31633
```

This model is considered the **current baseline** for NeuroKernel. Further model improvement and validation will be performed as the project progresses.

## Feature Importance

The most important features used by the current Random Forest model are:

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

## Completed Work

- Process telemetry data collection
- Normal telemetry collection
- CPU anomaly data collection
- Memory anomaly data collection
- Disk anomaly data collection
- Dataset combination
- Dataset validation
- Feature preparation
- Missing-value and duplicate checks
- 80/20 stratified train-test split
- Random Forest model training
- Model evaluation
- Feature importance analysis
- Baseline model saved as `neurokernel_random_forest.pkl`

## Current Project Status

The initial machine learning pipeline has been completed.

The current system can take process-level telemetry features and classify an observation as:

```text
Normal
   or
Anomaly
```

The current baseline performance is approximately **75% accuracy**.

The dataset and model are now ready for further development by the team.

## Remaining Work

### 1. Improve Model Performance

Investigate ways to improve the current baseline model through:

- Better feature engineering
- Hyperparameter tuning
- Testing different machine learning algorithms
- Cross-validation
- Improving the quality and diversity of telemetry data

### 2. Anomaly Type Classification

The current model performs binary classification.

The next stage is to investigate classification of different anomaly types:

```text
Normal
CPU Anomaly
Memory Anomaly
Disk Anomaly
```

### 3. Real-Time Process Monitoring

Connect the trained model to live Linux process telemetry.

The intended pipeline is:

```text
Running Processes
       ↓
Telemetry Collection
       ↓
Feature Extraction
       ↓
Machine Learning Model
       ↓
Normal / Anomaly
       ↓
Anomaly Type
```

### 4. NeuroKernel Dashboard

Develop a dashboard that can display:

- Running processes
- CPU usage
- Memory usage
- Disk activity
- Number of threads
- Anomaly status
- Anomaly probability
- Anomaly type

### 5. Automated Response

After reliable anomaly detection is established, investigate safe responses such as:

- Warning the user
- Reducing process priority
- Pausing a suspicious process
- Terminating a confirmed abnormal process

These actions should be carefully tested so that legitimate system processes are not affected.

### 6. Final System Integration

The final NeuroKernel system should combine:

```text
Linux Process Monitoring
          ↓
Telemetry Collection
          ↓
Feature Processing
          ↓
ML Anomaly Detection
          ↓
Anomaly Classification
          ↓
Dashboard
          ↓
Safe Response
```

## Project Structure

```text
NeuroKernel/
│
├── data/
│   ├── neurokernel_final.csv
│   ├── X_features.csv
│   ├── y_labels.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   └── y_test.csv
│
├── models/
│   └── neurokernel_random_forest.pkl
│
├── src/
│   ├── collect.py
│   ├── split_data.py
│   ├── train_model.py
│   └── analyze_model.py
│
├── data/
├── models/
├── src/
├── README.md
├── requirements.txt
└── .gitignore
```

## Team Handoff

The initial dataset, preprocessing pipeline and baseline Random Forest model have already been prepared.

The team can continue development from the existing model instead of recreating the initial dataset.

### Immediate priorities

1. Improve and validate the ML model.
2. Test additional machine learning algorithms.
3. Improve the quality and diversity of telemetry data.
4. Develop CPU/Memory/Disk anomaly classification.
5. Implement real-time process monitoring.
6. Connect the ML model to live telemetry.
7. Develop the NeuroKernel dashboard.
8. Implement and safely test anomaly-response mechanisms.
9. Integrate and evaluate the complete NeuroKernel system.
