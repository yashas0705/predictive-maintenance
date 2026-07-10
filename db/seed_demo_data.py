"""
Populate the DB with a few demo machines + realistic sensor history,
so the app looks alive the moment a recruiter opens it.
Run from the app/ directory: python ../db/seed_demo_data.py
"""
import os
import sys
import random
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app"))
from utils import init_db, get_conn, load_model, predict_risk, now_str  # noqa: E402

random.seed(7)

MACHINES = [
    ("CNC Lathe 1", "M", "2023-11-02", "Shop Floor A"),
    ("Drill Press 2", "L", "2024-02-14", "Shop Floor A"),
    ("Hydraulic Press 3", "H", "2022-08-20", "Shop Floor B"),
    ("Conveyor Motor 4", "L", "2024-05-10", "Warehouse"),
]


def simulate_history(machine_type, days=20, degrade=False):
    """Generate a plausible sensor trend, optionally degrading toward failure."""
    base_air = 298.0 + random.uniform(-1, 1)
    base_process = base_air + 10 + random.uniform(-0.5, 0.5)
    base_rpm = random.uniform(1350, 1650)
    base_torque = random.uniform(35, 55)
    tool_wear = 0.0

    rows = []
    start = datetime.now() - timedelta(days=days)
    for d in range(days):
        ts = start + timedelta(days=d)
        drift = d / days if degrade else 0
        air_temp = base_air + random.uniform(-0.3, 0.3) + drift * 1.5
        process_temp = base_process + random.uniform(-0.3, 0.3) + drift * 2
        rpm = base_rpm - drift * 150 + random.uniform(-20, 20)
        torque = base_torque + drift * 15 + random.uniform(-2, 2)
        tool_wear += random.uniform(6, 10) + (drift * 4 if degrade else 0)
        rows.append((ts.strftime("%Y-%m-%d %H:%M:%S"), air_temp, process_temp, rpm, torque, round(tool_wear, 1)))
    return rows


def main():
    init_db()
    conn = get_conn()
    cur = conn.cursor()
    model, encoder = load_model()

    degrade_flags = [False, False, True, True]  # last two machines trend toward failure

    for (name, mtype, install, loc), degrade in zip(MACHINES, degrade_flags):
        cur.execute(
            "INSERT INTO machines (name, machine_type, install_date, location) VALUES (?,?,?,?)",
            (name, mtype, install, loc),
        )
        machine_id = cur.lastrowid

        history = simulate_history(mtype, days=20, degrade=degrade)
        for ts, air_temp, process_temp, rpm, torque, tool_wear in history:
            risk_score, failure_type = predict_risk(
                model, encoder, mtype, air_temp, process_temp, rpm, torque, tool_wear
            )
            cur.execute(
                """INSERT INTO sensor_logs
                   (machine_id, timestamp, air_temp, process_temp, rpm, torque, tool_wear, risk_score, predicted_failure_type)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (machine_id, ts, air_temp, process_temp, rpm, torque, tool_wear, risk_score, failure_type),
            )

        # one maintenance event for the oldest machine
        if name == "CNC Lathe 1":
            cur.execute(
                "INSERT INTO maintenance_log (machine_id, serviced_at, notes, technician) VALUES (?,?,?,?)",
                (machine_id, (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d"),
                 "Routine tool bit replacement", "R. Kumar"),
            )

    conn.commit()
    conn.close()
    print(f"Seeded {len(MACHINES)} machines with 20-day sensor history each.")


if __name__ == "__main__":
    main()
