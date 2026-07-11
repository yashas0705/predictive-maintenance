import os
import pickle
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "db", "machines.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "db", "schema.sql")
MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
ENCODER_PATH = os.path.join(BASE_DIR, "models", "encoder.pkl")

FAILURE_TYPES = {
    "TWF": "Tool Wear Failure",
    "HDF": "Heat Dissipation Failure",
    "PWF": "Power Failure",
    "OSF": "Overstrain Failure",
    "RNF": "Random Failure",
}


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_conn()
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()


def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(ENCODER_PATH, "rb") as f:
        encoder = pickle.load(f)
    return model, encoder


def predict_risk(model, encoder, machine_type, air_temp, process_temp, rpm, torque, tool_wear):
    """Returns (risk_score 0-100, likely_failure_type)"""
    try:
        type_enc = encoder.transform([machine_type])[0]
    except ValueError:
        type_enc = 0  # unseen type -> default to L

    X = pd.DataFrame(
        [[type_enc, air_temp, process_temp, rpm, torque, tool_wear]],
        columns=["type_enc", "air_temp", "process_temp", "rpm", "torque", "tool_wear"],
    )
    proba = model.predict_proba(X)[0][1]
    risk_score = float(round(proba * 100, 1))

    # heuristic sub-type attribution based on AI4I's known physical failure rules
    likely_type = classify_failure_mode(machine_type, torque, tool_wear, process_temp, air_temp, rpm)

    return risk_score, likely_type


def classify_failure_mode(machine_type, torque, tool_wear, process_temp, air_temp, rpm):
    """Rule-of-thumb mode attribution mirroring AI4I's documented failure physics."""
    thresholds = {"L": 11000, "M": 12000, "H": 13000}
    osf_thresh = thresholds.get(machine_type, 12000)

    if tool_wear >= 200:
        return "TWF"
    if (process_temp - air_temp) < 8.6 and rpm < 1380:
        return "HDF"
    power = torque * rpm * (2 * np.pi / 60)
    if power < 3500 or power > 9000:
        return "PWF"
    if torque * tool_wear > osf_thresh:
        return "OSF"
    return "None"


def safe_num(val, default=None):
    """Coerce a possibly-corrupted DB value (None, NaN, stray string) to a float, or return default."""
    try:
        f = float(val)
        if np.isnan(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def days_to_failure(tool_wear, risk_score):
    """Simple heuristic ETA: higher tool wear + risk => fewer days left.
    Tool wear failure threshold in AI4I is ~200-240 min; assume ~0.5-2 min/day of use."""
    tool_wear = safe_num(tool_wear, default=0.0)
    risk_score = safe_num(risk_score, default=0.0)
    if risk_score < 5:
        return None  # low risk, no ETA needed
    wear_remaining = max(200 - tool_wear, 5)
    daily_wear_rate = 1 + (risk_score / 100) * 3  # riskier machines assumed to wear faster
    eta = int(wear_remaining / daily_wear_rate)
    return max(eta, 1)


def risk_bucket(risk_score):
    risk_score = safe_num(risk_score, default=None)
    if risk_score is None:
        return "No data", "⚪"
    if risk_score < 15:
        return "Low", "🟢"
    elif risk_score < 50:
        return "Medium", "🟡"
    else:
        return "High", "🔴"


def now_str():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _db_is_healthy():
    """Check the DB is a valid SQLite file with sane numeric data (catches corruption
    from binary files mangled by web-based git uploads)."""
    try:
        conn = get_conn()
        row = conn.execute(
            "SELECT risk_score, tool_wear FROM sensor_logs LIMIT 1"
        ).fetchone()
        conn.close()
        if row is None:
            return True  # empty table is fine, not corruption
        float(row["risk_score"]) if row["risk_score"] is not None else 0.0
        float(row["tool_wear"]) if row["tool_wear"] is not None else 0.0
        return True
    except Exception:
        return False


def ensure_setup():
    """First-run bootstrap for a fresh deploy (e.g. Streamlit Community Cloud):
    trains the model if model.pkl is missing, seeds demo data if the DB is empty,
    and rebuilds the DB from scratch if it's corrupted (e.g. a binary .db file
    mangled by a web-based upload)."""
    import subprocess
    import sys

    models_dir = os.path.join(BASE_DIR, "models")
    db_dir = os.path.join(BASE_DIR, "db")

    if not os.path.exists(MODEL_PATH) or not os.path.exists(ENCODER_PATH):
        subprocess.run([sys.executable, "train_model.py"], cwd=models_dir, check=True)

    if os.path.exists(DB_PATH) and not _db_is_healthy():
        os.remove(DB_PATH)  # corrupted -> rebuild from scratch

    init_db()

    conn = get_conn()
    count = conn.execute("SELECT COUNT(*) FROM machines").fetchone()[0]
    conn.close()

    if count == 0:
        seed_script = os.path.join(db_dir, "seed_demo_data.py")
        if os.path.exists(seed_script):
            subprocess.run([sys.executable, "seed_demo_data.py"], cwd=db_dir, check=True)
