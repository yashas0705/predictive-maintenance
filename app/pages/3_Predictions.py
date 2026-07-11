import streamlit as st
import pandas as pd
import plotly.express as px

from utils import ensure_setup, get_conn, days_to_failure, risk_bucket, FAILURE_TYPES, safe_num

st.set_page_config(page_title="Breakdown Predictions", page_icon="⚠️", layout="wide")
ensure_setup()
st.title("⚠️ Breakdown Predictions")
st.caption("Fleet-wide ranking by urgency — highest risk machines need attention first.")

THEME_COLORS = ["#5B47FB", "#FF6F91", "#00C2A8", "#FFB347"]
BUCKET_COLORS = {"Low": "#00C2A8", "Medium": "#FFB347", "High": "#FF6F91", "No data": "#C9CDD6"}

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

latest_logs["risk_score"] = latest_logs["risk_score"].apply(lambda v: safe_num(v, default=0.0))
latest_logs["tool_wear"] = latest_logs["tool_wear"].apply(lambda v: safe_num(v, default=0.0))

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
        return ["background-color: #ffd8e0"] * len(row)
    elif row["Status"] == "Medium":
        return ["background-color: #fff0d6"] * len(row)
    elif row["Status"] == "Low":
        return ["background-color: #d3f5ec"] * len(row)
    return [""] * len(row)

st.dataframe(display_df.style.apply(highlight_risk, axis=1), use_container_width=True, hide_index=True)

st.divider()
c1, c2 = st.columns(2)
with c1:
    st.subheader("Risk Distribution by Machine")
    fig = px.bar(latest_logs, x="name", y="risk_score", color="status",
                 color_discrete_map=BUCKET_COLORS,
                 labels={"name": "Machine", "risk_score": "Risk %"})
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.subheader("Failure Mode Breakdown")
    mode_counts = latest_logs[latest_logs["predicted_failure_type"] != "None"]["failure_mode"].value_counts()
    if not mode_counts.empty:
        fig2 = px.pie(values=mode_counts.values, names=mode_counts.index, hole=0.45,
                       color_discrete_sequence=THEME_COLORS)
        fig2.update_traces(textinfo="percent+label", pull=[0.03] * len(mode_counts))
        fig2.update_layout(margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.caption("No machines currently showing failure indicators.")
