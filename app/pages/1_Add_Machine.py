import streamlit as st
import pandas as pd
from datetime import date

from utils import ensure_setup, get_conn, load_model, predict_risk, now_str

st.set_page_config(page_title="Add Machine", page_icon="➕", layout="wide")
ensure_setup()
st.title("➕ Add a Machine")

tab1, tab2 = st.tabs(["Register New Machine", "Upload Sensor CSV (bulk log)"])

with tab1:
    with st.form("add_machine_form"):
        c1, c2 = st.columns(2)
        name = c1.text_input("Machine Name / ID", placeholder="e.g. CNC Lathe #3")
        machine_type = c2.selectbox("Quality/Type Variant", ["L", "M", "H"], help="Low / Medium / High spec variant")
        location = c1.text_input("Location (optional)", placeholder="e.g. Shop Floor A")
        install_date = c2.date_input("Install Date", value=date.today())

        st.markdown("**Initial sensor reading** (optional — you can log later)")
        r1, r2, r3 = st.columns(3)
        air_temp = r1.number_input("Air Temp (K)", value=298.0, step=0.1)
        process_temp = r2.number_input("Process Temp (K)", value=308.0, step=0.1)
        rpm = r3.number_input("Rotational Speed (rpm)", value=1500, step=10)
        r4, r5 = st.columns(2)
        torque = r4.number_input("Torque (Nm)", value=40.0, step=0.5)
        tool_wear = r5.number_input("Tool Wear (min)", value=0.0, step=1.0)

        submitted = st.form_submit_button("Add Machine", type="primary")

        if submitted:
            if not name:
                st.error("Machine name is required.")
            else:
                conn = get_conn()
                cur = conn.cursor()
                cur.execute(
                    "INSERT INTO machines (name, machine_type, install_date, location) VALUES (?, ?, ?, ?)",
                    (name, machine_type, str(install_date), location),
                )
                machine_id = cur.lastrowid

                model, encoder = load_model()
                risk_score, failure_type = predict_risk(
                    model, encoder, machine_type, air_temp, process_temp, rpm, torque, tool_wear
                )
                cur.execute(
                    """INSERT INTO sensor_logs
                       (machine_id, timestamp, air_temp, process_temp, rpm, torque, tool_wear, risk_score, predicted_failure_type)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (machine_id, now_str(), air_temp, process_temp, rpm, torque, tool_wear, risk_score, failure_type),
                )
                conn.commit()
                conn.close()
                st.success(f"✅ Machine '{name}' added — initial risk score: {risk_score}%")
                st.balloons()

with tab2:
    st.markdown("Upload a CSV with columns: `machine_id, timestamp, air_temp, process_temp, rpm, torque, tool_wear`")
    conn = get_conn()
    machines = pd.read_sql("SELECT id, name FROM machines", conn)
    if machines.empty:
        st.warning("Register a machine first before bulk-uploading logs.")
    else:
        st.dataframe(machines, use_container_width=True, hide_index=True)
        uploaded = st.file_uploader("Sensor log CSV", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            required = {"machine_id", "timestamp", "air_temp", "process_temp", "rpm", "torque", "tool_wear"}
            if not required.issubset(df.columns):
                st.error(f"CSV must contain columns: {required}")
            else:
                model, encoder = load_model()
                cur = conn.cursor()
                type_map = dict(pd.read_sql("SELECT id, machine_type FROM machines", conn).values)
                for _, row in df.iterrows():
                    mtype = type_map.get(row["machine_id"], "M")
                    risk_score, failure_type = predict_risk(
                        model, encoder, mtype, row["air_temp"], row["process_temp"],
                        row["rpm"], row["torque"], row["tool_wear"]
                    )
                    cur.execute(
                        """INSERT INTO sensor_logs
                           (machine_id, timestamp, air_temp, process_temp, rpm, torque, tool_wear, risk_score, predicted_failure_type)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (row["machine_id"], row["timestamp"], row["air_temp"], row["process_temp"],
                         row["rpm"], row["torque"], row["tool_wear"], risk_score, failure_type),
                    )
                conn.commit()
                st.success(f"✅ Uploaded and scored {len(df)} sensor readings.")
