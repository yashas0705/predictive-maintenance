import streamlit as st
import pandas as pd
from datetime import date

from utils import ensure_setup, get_conn, load_model, predict_risk, now_str

st.set_page_config(page_title="Maintenance Log", page_icon="🔧", layout="wide")
ensure_setup()
st.title("🔧 Maintenance Log")

conn = get_conn()
machines = pd.read_sql("SELECT * FROM machines ORDER BY name", conn)

if machines.empty:
    st.info("No machines yet — add one first.")
    st.stop()

st.subheader("Log a Service Event")
with st.form("maintenance_form"):
    c1, c2 = st.columns(2)
    machine_name = c1.selectbox("Machine", machines["name"])
    serviced_at = c2.date_input("Service Date", value=date.today())
    technician = st.text_input("Technician (optional)")
    notes = st.text_area("Notes", placeholder="e.g. Replaced worn tool bit, recalibrated torque sensor")
    reset_wear = st.checkbox("Reset tool wear to 0 (part replaced)", value=True)
    submit = st.form_submit_button("Log Maintenance", type="primary")

    if submit:
        machine_id = int(machines[machines["name"] == machine_name]["id"].iloc[0])
        machine_type = machines[machines["name"] == machine_name]["machine_type"].iloc[0]
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO maintenance_log (machine_id, serviced_at, notes, technician) VALUES (?, ?, ?, ?)",
            (machine_id, str(serviced_at), notes, technician),
        )

        if reset_wear:
            last_log = pd.read_sql(
                "SELECT * FROM sensor_logs WHERE machine_id = ? ORDER BY id DESC LIMIT 1",
                conn, params=(machine_id,)
            )
            if not last_log.empty:
                model, encoder = load_model()
                l = last_log.iloc[0]
                risk_score, failure_type = predict_risk(
                    model, encoder, machine_type, l["air_temp"], l["process_temp"], l["rpm"], l["torque"], 0.0
                )
                cur.execute(
                    """INSERT INTO sensor_logs
                       (machine_id, timestamp, air_temp, process_temp, rpm, torque, tool_wear, risk_score, predicted_failure_type)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (machine_id, now_str(), l["air_temp"], l["process_temp"], l["rpm"], l["torque"], 0.0, risk_score, failure_type),
                )
        conn.commit()
        st.success(f"✅ Maintenance logged for {machine_name}" + (" — tool wear reset." if reset_wear else "."))
        st.rerun()

st.divider()
st.subheader("Maintenance History")
history = pd.read_sql(
    """
    SELECT m.name AS Machine, ml.serviced_at AS "Serviced On", ml.technician AS Technician, ml.notes AS Notes
    FROM maintenance_log ml JOIN machines m ON m.id = ml.machine_id
    ORDER BY ml.serviced_at DESC
    """,
    conn,
)
if history.empty:
    st.caption("No maintenance events logged yet.")
else:
    st.dataframe(history, use_container_width=True, hide_index=True)
