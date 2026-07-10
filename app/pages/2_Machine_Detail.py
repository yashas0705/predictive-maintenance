import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime

from utils import ensure_setup, get_conn, load_model, predict_risk, days_to_failure, risk_bucket, now_str, FAILURE_TYPES

st.set_page_config(page_title="Machine Detail", page_icon="📊", layout="wide")
ensure_setup()
st.title("📊 Machine Detail")

conn = get_conn()
machines = pd.read_sql("SELECT * FROM machines ORDER BY name", conn)

if machines.empty:
    st.info("No machines yet — add one first.")
    st.stop()

machine_name = st.selectbox("Select Machine", machines["name"])
machine_row = machines[machines["name"] == machine_name].iloc[0]
machine_id = int(machine_row["id"])

logs = pd.read_sql(
    "SELECT * FROM sensor_logs WHERE machine_id = ? ORDER BY timestamp", conn, params=(machine_id,)
)

st.markdown(f"**Type:** {machine_row['machine_type']}  |  **Location:** {machine_row['location'] or '—'}  |  **Installed:** {machine_row['install_date']}")

col_log, col_gauge = st.columns([1, 1])

with col_log:
    st.subheader("Log a New Reading")
    with st.form("log_reading"):
        r1, r2 = st.columns(2)
        air_temp = r1.number_input("Air Temp (K)", value=298.0, step=0.1)
        process_temp = r2.number_input("Process Temp (K)", value=308.0, step=0.1)
        r3, r4 = st.columns(2)
        rpm = r3.number_input("Rotational Speed (rpm)", value=1500, step=10)
        torque = r4.number_input("Torque (Nm)", value=40.0, step=0.5)
        tool_wear = st.number_input(
            "Tool Wear (min)",
            value=float(logs["tool_wear"].iloc[-1]) if not logs.empty else 0.0,
            step=1.0,
        )
        submit = st.form_submit_button("Run Prediction & Log", type="primary")

        if submit:
            model, encoder = load_model()
            risk_score, failure_type = predict_risk(
                model, encoder, machine_row["machine_type"], air_temp, process_temp, rpm, torque, tool_wear
            )
            cur = conn.cursor()
            cur.execute(
                """INSERT INTO sensor_logs
                   (machine_id, timestamp, air_temp, process_temp, rpm, torque, tool_wear, risk_score, predicted_failure_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (machine_id, now_str(), air_temp, process_temp, rpm, torque, tool_wear, risk_score, failure_type),
            )
            conn.commit()
            st.success(f"Logged. Risk score: {risk_score}%")
            st.rerun()

with col_gauge:
    st.subheader("Current Risk")
    if logs.empty:
        st.caption("No readings logged yet.")
    else:
        latest = logs.iloc[-1]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=latest["risk_score"],
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#333"},
                "steps": [
                    {"range": [0, 15], "color": "#c8f0c8"},
                    {"range": [15, 50], "color": "#fff3b0"},
                    {"range": [50, 100], "color": "#f7b2b2"},
                ],
            },
            title={"text": "Failure Risk"},
        ))
        fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig, use_container_width=True)

        label, emoji = risk_bucket(latest["risk_score"])
        ft = latest["predicted_failure_type"]
        ft_label = FAILURE_TYPES.get(ft, "No failure mode indicated")
        st.markdown(f"**Status:** {emoji} {label}")
        st.markdown(f"**Likely failure mode:** {ft_label}")
        eta = days_to_failure(latest["tool_wear"], latest["risk_score"])
        if eta:
            st.markdown(f"**Estimated days to failure:** {eta}")

st.divider()

if len(logs) >= 2:
    st.subheader("Sensor Trends")
    logs["timestamp"] = pd.to_datetime(logs["timestamp"])

    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(logs, x="timestamp", y=["air_temp", "process_temp"], markers=True,
                       title="Temperature Over Time", labels={"value": "Kelvin", "timestamp": "Time"})
        st.plotly_chart(fig, use_container_width=True)

        fig3 = px.line(logs, x="timestamp", y="tool_wear", markers=True,
                        title="Tool Wear Over Time", labels={"tool_wear": "Minutes"})
        fig3.add_hline(y=200, line_dash="dash", line_color="red", annotation_text="Wear failure threshold")
        st.plotly_chart(fig3, use_container_width=True)

    with c2:
        fig2 = px.line(logs, x="timestamp", y=["rpm", "torque"], markers=True,
                        title="Speed & Torque Over Time")
        st.plotly_chart(fig2, use_container_width=True)

        fig4 = px.area(logs, x="timestamp", y="risk_score", title="Risk Score Trend",
                        labels={"risk_score": "Risk %"})
        st.plotly_chart(fig4, use_container_width=True)
else:
    st.info("Log at least 2 readings to see trend charts.")
