import streamlit as st
import pandas as pd

from utils import ensure_setup, get_conn, load_model, predict_risk, days_to_failure, risk_bucket

st.set_page_config(page_title="PredictAI | Fleet Overview", page_icon="🛠️", layout="wide")

with st.spinner("Setting up (first run only — training model & loading demo data)..."):
    ensure_setup()

st.title("🛠️ PredictAI — Predictive Maintenance for Small Business")
st.caption("Fleet health at a glance. Add machines, log readings, and catch failures before they happen.")

conn = get_conn()
machines = pd.read_sql("SELECT * FROM machines ORDER BY id DESC", conn)

if machines.empty:
    st.info("No machines added yet. Go to **Add Machine** in the sidebar to get started.")
    st.stop()

model, encoder = load_model()

# latest reading per machine
latest_logs = pd.read_sql(
    """
    SELECT sl.* FROM sensor_logs sl
    INNER JOIN (
        SELECT machine_id, MAX(id) AS max_id FROM sensor_logs GROUP BY machine_id
    ) latest ON sl.machine_id = latest.machine_id AND sl.id = latest.max_id
    """,
    conn,
)

col_hi, col_med, col_lo, col_total = st.columns(4)
merged = machines.merge(latest_logs, left_on="id", right_on="machine_id", how="left")

def bucket_of(row):
    if pd.isna(row.get("risk_score")):
        return "No data"
    return risk_bucket(row["risk_score"])[0]

merged["bucket"] = merged.apply(bucket_of, axis=1)

col_total.metric("Total Machines", len(machines))
col_hi.metric("🔴 High Risk", (merged["bucket"] == "High").sum())
col_med.metric("🟡 Medium Risk", (merged["bucket"] == "Medium").sum())
col_lo.metric("🟢 Low Risk", (merged["bucket"] == "Low").sum())

st.divider()
st.subheader("Machine Fleet")

cols = st.columns(3)
for i, row in merged.iterrows():
    with cols[i % 3]:
        with st.container(border=True):
            st.markdown(f"**{row['name']}**  \n`{row['machine_type']}-type` · {row['location'] or 'No location'}")
            if pd.isna(row.get("risk_score")):
                st.caption("No sensor readings logged yet.")
            else:
                label, emoji = risk_bucket(row["risk_score"])
                st.markdown(f"### {emoji} {row['risk_score']}% risk")
                st.caption(f"Status: **{label}**")
                eta = days_to_failure(row["tool_wear"], row["risk_score"])
                if eta:
                    st.caption(f"⏱️ Est. {eta} days to failure if untreated")
            st.caption(f"Installed: {row['install_date']}")

st.divider()
st.caption("Built with AI4I 2020 dataset · XGBoost · Streamlit · SQLite")
