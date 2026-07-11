import streamlit as st
import pandas as pd
import plotly.express as px

from utils import ensure_setup, get_conn, load_model, predict_risk, days_to_failure, risk_bucket, safe_num

st.set_page_config(page_title="PredictAI | Fleet Overview", page_icon="🛠️", layout="wide")

with st.spinner("Setting up (first run only — training model & loading demo data)..."):
    ensure_setup()

THEME_COLORS = ["#5B47FB", "#FF6F91", "#00C2A8", "#FFB347"]
BUCKET_COLORS = {"Low": "#00C2A8", "Medium": "#FFB347", "High": "#FF6F91", "No data": "#C9CDD6"}

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

merged = machines.merge(latest_logs, left_on="id", right_on="machine_id", how="left")
merged["risk_score_clean"] = merged["risk_score"].apply(lambda v: safe_num(v, default=None))


def bucket_of(row):
    return risk_bucket(row["risk_score_clean"])[0]


merged["bucket"] = merged.apply(bucket_of, axis=1)

col_hi, col_med, col_lo, col_total = st.columns(4)
col_total.metric("Total Machines", len(machines))
col_hi.metric("🔴 High Risk", (merged["bucket"] == "High").sum())
col_med.metric("🟡 Medium Risk", (merged["bucket"] == "Medium").sum())
col_lo.metric("🟢 Low Risk", (merged["bucket"] == "Low").sum())

st.divider()

col_cards, col_chart = st.columns([2, 1])

with col_cards:
    st.subheader("Machine Fleet")
    cols = st.columns(2)
    for i, row in merged.iterrows():
        with cols[i % 2]:
            with st.container(border=True):
                st.markdown(f"**{row['name']}**  \n`{row['machine_type']}-type` · {row['location'] or 'No location'}")
                risk = row["risk_score_clean"]
                if risk is None:
                    st.caption("No sensor readings logged yet.")
                else:
                    label, emoji = risk_bucket(risk)
                    st.markdown(f"### {emoji} {risk}% risk")
                    st.caption(f"Status: **{label}**")
                    eta = days_to_failure(row.get("tool_wear"), risk)
                    if eta:
                        st.caption(f"⏱️ Est. {eta} days to failure if untreated")
                st.caption(f"Installed: {row['install_date']}")

with col_chart:
    st.subheader("Risk Distribution")
    bucket_counts = merged["bucket"].value_counts().reindex(
        ["Low", "Medium", "High", "No data"]
    ).dropna()
    if bucket_counts.sum() > 0:
        fig = px.pie(
            values=bucket_counts.values,
            names=bucket_counts.index,
            hole=0.45,
            color=bucket_counts.index,
            color_discrete_map=BUCKET_COLORS,
        )
        fig.update_traces(textinfo="percent+label", pull=[0.03] * len(bucket_counts))
        fig.update_layout(showlegend=False, margin=dict(l=10, r=10, t=10, b=10), height=320)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption("No data to chart yet.")

st.divider()
st.caption("Built with AI4I 2020 dataset · XGBoost · Streamlit · SQLite")
