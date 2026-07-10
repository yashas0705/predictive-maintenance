import streamlit as st
import pandas as pd
import plotly.express as px

from utils import ensure_setup, get_conn, days_to_failure, risk_bucket, FAILURE_TYPES

st.set_page_config(page_title="Breakdown Predictions", page_icon="⚠️", layout="wide")
ensure_setup()
st.title("⚠️ Breakdown Predictions")
st.caption("Fleet-wide ranking by urgency — highest risk machines need attention first.")

conn = get_conn()
latest_logs = pd.read_sql(
    """
    SELECT m.name, m.machine_type, m.location, sl.* FROM sensor_logs sl
    INNER JOIN (
        SELECT machine_id, MAX(id) AS max_id FROM sensor_logs GROUP BY machine_id
    ) latest ON sl.machine_id = latest.machine_id AND sl.id = latest.max_id
    INNER JOIN machines m ON m.id = sl.machine_id
    """,
    conn,
)

if latest_logs.empty:
    st.info("No sensor readings logged yet. Add machines and log readings first.")
    st.stop()
numeric_cols = [
    "risk_score",
    "tool_wear",
    "air_temp",
    "process_temp",
    "rpm",
    "torque"
]

for col in numeric_cols:
    if col in latest_logs.columns:
        latest_logs[col] = pd.to_numeric(
            latest_logs[col],
            errors="coerce"
        )

latest_logs["days_to_failure"] = latest_logs.apply(
    lambda r: days_to_failure(r["tool_wear"], r["risk_score"]), axis=1
)
latest_logs["status"] = latest_logs["risk_score"].apply(lambda r: risk_bucket(r)[0])
latest_logs["failure_mode"] = latest_logs["predicted_failure_type"].map(FAILURE_TYPES).fillna("—")
latest_logs = latest_logs.sort_values("risk_score", ascending=False)

st.subheader("Urgency Table")
display_df = latest_logs[[
    "name", "machine_type", "location", "risk_score", "status", "failure_mode", "days_to_failure", "timestamp"
]].rename(columns={
    "name": "Machine", "machine_type": "Type", "location": "Location",
    "risk_score": "Risk %", "status": "Status", "failure_mode": "Predicted Failure Mode",
    "days_to_failure": "Days to Failure (est.)", "timestamp": "Last Reading"
})

def highlight_risk(row):
    if row["Status"] == "High":
        return ["background-color: #f7b2b2"] * len(row)
    elif row["Status"] == "Medium":
        return ["background-color: #fff3b0"] * len(row)
    return ["background-color: #c8f0c8"] * len(row)

st.dataframe(display_df.style.apply(highlight_risk, axis=1), use_container_width=True, hide_index=True)

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Risk Distribution by Machine")
    fig = px.bar(latest_logs, x="name", y="risk_score", color="status",
                 color_discrete_map={"Low": "#4caf50", "Medium": "#ffc107", "High": "#f44336"},
                 labels={"name": "Machine", "risk_score": "Risk %"})
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Failure Mode Breakdown")
    mode_counts = latest_logs[latest_logs["predicted_failure_type"] != "None"]["failure_mode"].value_counts()
    if not mode_counts.empty:
        fig2 = px.pie(values=mode_counts.values, names=mode_counts.index, hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("No machines currently showing failure indicators.")
