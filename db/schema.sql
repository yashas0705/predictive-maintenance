CREATE TABLE IF NOT EXISTS machines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    machine_type TEXT NOT NULL,          -- L / M / H (AI4I product quality variant)
    install_date TEXT NOT NULL,
    location TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS sensor_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER NOT NULL,
    timestamp TEXT NOT NULL,
    air_temp REAL NOT NULL,
    process_temp REAL NOT NULL,
    rpm REAL NOT NULL,
    torque REAL NOT NULL,
    tool_wear REAL NOT NULL,
    risk_score REAL,
    predicted_failure_type TEXT,
    FOREIGN KEY (machine_id) REFERENCES machines(id)
);

CREATE TABLE IF NOT EXISTS maintenance_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    machine_id INTEGER NOT NULL,
    serviced_at TEXT NOT NULL,
    notes TEXT,
    technician TEXT,
    FOREIGN KEY (machine_id) REFERENCES machines(id)
);
