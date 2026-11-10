import sys
from pathlib import Path

import pandas as pd
import streamlit as st


# Make the src directory available for imports.
PROJECT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_monitor import LiveProcessMonitor
from member4_integration import Member4Investigator


st.set_page_config(
    page_title="NeuroKernel | Live Monitor",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 NeuroKernel")
st.subheader("Live Process Anomaly Detection")

st.markdown(
    """
    Monitor running processes using live system telemetry and the
    existing trained machine-learning model. Member 4's investigation layer
    adds a likely resource category and a human-readable explanation.
    """
)


@st.cache_resource
def get_monitor():
    """Keep one monitor instance so consecutive scans can be compared."""
    return LiveProcessMonitor()


@st.cache_resource
def get_investigator():
    """Build the resource baseline once from normal training samples."""
    return Member4Investigator(PROJECT_DIR)


try:
    monitor = get_monitor()
except Exception as exc:
    st.error("Could not initialize the live process monitor.")
    st.exception(exc)
    st.stop()

try:
    investigator = get_investigator()
except Exception as exc:
    st.error("Could not initialize Member 4 anomaly investigation.")
    st.exception(exc)
    st.stop()


def live_dashboard():
    st.caption("The dashboard refreshes approximately every 10 seconds.")

    try:
        records = monitor.scan()
        records = investigator.investigate_many(records)
    except Exception as exc:
        st.error("The live process scan or anomaly investigation failed.")
        st.exception(exc)
        return

    if not records:
        st.warning("No process records were returned.")
        return

    df = pd.DataFrame(records)

    # Keep counts separate from processes still waiting for predictions.
    ready = df[df["prediction"].notna()].copy()
    warming = df[df["prediction"].isna()].copy()

    anomaly_count = int((ready["prediction"] == 1).sum())
    normal_count = int((ready["prediction"] == 0).sum())
    warming_count = len(warming)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Processes Scanned", len(df))
    col2.metric("Normal", normal_count)
    col3.metric("Anomalies Predicted", anomaly_count)
    col4.metric("Warming Up", warming_count)

    st.divider()

    st.subheader("Live Process Table")

    filter_choice = st.selectbox(
        "Filter processes",
        [
            "All processes",
            "Anomalies only",
            "Normal only",
            "Warming up",
        ],
        key="process_filter",
    )

    if filter_choice == "Anomalies only":
        shown = df[df["prediction"] == 1].copy()
    elif filter_choice == "Normal only":
        shown = df[df["prediction"] == 0].copy()
    elif filter_choice == "Warming up":
        shown = df[df["prediction"].isna()].copy()
    else:
        shown = df.copy()

    # Show likely anomalies first, followed by higher CPU usage.
    if not shown.empty:
        shown["_anomaly_sort"] = (
            shown["prediction"].fillna(-1) == 1
        ).astype(int)

        shown = shown.sort_values(
            by=["_anomaly_sort", "cpu_percent"],
            ascending=[False, False],
        )

    display_columns = [
        "pid",
        "process_name",
        "cpu_percent",
        "memory_mb",
        "memory_percent",
        "threads",
        "status",
        "anomaly_probability",
        "anomaly_type",
        "investigation_explanation",
    ]

    available_columns = [
        column for column in display_columns if column in shown.columns
    ]

    display_df = shown[available_columns].copy()

    if "anomaly_probability" in display_df.columns:
        display_df["anomaly_probability"] = display_df[
            "anomaly_probability"
        ].map(
            lambda value: (
                f"{float(value) * 100:.1f}%"
                if pd.notna(value)
                else "Unavailable"
            )
        )

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Anomaly is the model's prediction, not proof of malicious activity. "
        "The resource category is a heuristic explanation based on deviations "
        "from normal training data. Warming-up processes need another sample."
    )

    st.divider()

    st.subheader("Process Details")

    if not df.empty:
        selected_pid = st.selectbox(
            "Select a process ID",
            options=df["pid"].tolist(),
            format_func=lambda pid: (
                f"{pid} — "
                f"{df.loc[df['pid'] == pid, 'process_name'].iloc[0]}"
            ),
            key="selected_pid",
        )

        selected_rows = df[df["pid"] == selected_pid]

        if not selected_rows.empty:
            process = selected_rows.iloc[0]

            detail1, detail2, detail3 = st.columns(3)

            detail1.write(f"**Process:** {process['process_name']}")
            detail2.write(f"**PID:** {process['pid']}")
            detail3.write(f"**Status:** {process['status']}")

            probability = process.get("anomaly_probability")

            if pd.notna(probability):
                st.metric(
                    "Model-reported anomaly probability",
                    f"{float(probability) * 100:.1f}%",
                )
            else:
                st.info(
                    "Anomaly probability is not available for this process."
                )

            st.markdown("#### Member 4 — Anomaly Investigation")
            st.write(
                "**Likely resource category:**",
                process.get("anomaly_type", "Unknown"),
            )
            explanation = process.get(
                "investigation_explanation",
                "No investigation explanation available.",
            )
            st.info(str(explanation))

            resource_details = process.get("resource_details")
            if resource_details:
                st.write("**Resource evidence:**")
                st.write(resource_details)

            note = process.get("telemetry_note")

            if isinstance(note, str) and note:
                st.warning(note)

            with st.expander("View complete telemetry and investigation data"):
                st.json(process.to_dict())

    with st.expander("How to interpret the results"):
        st.markdown(
            """
            - **Normal:** The model predicted class 0.
            - **Anomaly:** The model predicted class 1.
            - **Warming up:** The monitor needs another sample.
            - **Anomaly probability:** The probability reported by the model,
              when supported.
            - **Likely resource category:** Member 4's heuristic grouping of
              unusual CPU, memory, disk, or process-activity measurements.
            - **Unclassified:** The model detected an anomaly, but the
              investigation layer could not isolate a resource category.

            The investigation layer compares measurements with a baseline
            built only from label-0 normal training rows. It is not causal
            proof and does not determine whether a process is malicious.
            Investigate unusual results before drawing conclusions.
            """
        )


st.divider()

if st.button("🔄 Refresh Dashboard", type="primary"):
    st.rerun()

live_dashboard()
