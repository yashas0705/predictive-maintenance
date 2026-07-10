# PredictAI — Predictive Maintenance for Small Business

A full-stack predictive maintenance app for small manufacturers: register machines, log sensor readings, and get real-time failure-risk predictions with an estimated days-to-failure and explainable ML behind every score.

**Live demo:** _add your HuggingFace Spaces link here after deploying_

## Why this exists
Small shops running CNC lathes, drill presses, conveyor motors etc. rarely have access to enterprise IoT/predictive-maintenance platforms. This app shows the same core idea — sensor data in, failure risk out — in a lightweight tool a small business owner could actually run.

## Features
- **Add Machine** — register a machine (name, type, install date, location), log an initial reading, or bulk-upload a CSV of historical sensor data
- **Fleet Overview** — color-coded risk cards across all machines
- **Machine Detail** — log new readings, live risk gauge, sensor trend charts (temperature, RPM, torque, tool wear, risk over time)
- **Breakdown Predictions** — fleet-wide table ranked by urgency, with predicted failure mode and estimated days to failure
- **Maintenance Log** — record service events; optionally resets tool wear and re-scores risk
- **Model Insights** — model metrics (F1/precision/recall/ROC-AUC), feature importance, and SHAP explainability plots

## Tech stack
| Layer | Choice |
|---|---|
| ML model | XGBoost classifier (scale-pos-weight tuned for the ~3.4% failure class imbalance) |
| Explainability | SHAP |
| App/UI | Streamlit (multi-page) |
| Storage | SQLite (machines, sensor_logs, maintenance_log) |
| Charts | Plotly |
| Deployment | HuggingFace Spaces |

## Dataset
[AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset) (UCI ML Repository) — 10,000 synthetic-but-realistic industrial milling machine readings with 5 documented failure modes (tool wear, heat dissipation, power, overstrain, random failure).

> Matzka, S. "Explainable Artificial Intelligence for Predictive Maintenance Applications." 2020 IEEE Third International Conference on Artificial Intelligence for Industries (AI4I).

## Model performance
| Metric | Score |
|---|---|
| ROC-AUC | 0.97 |
| Recall | 0.85 |
| Precision | 0.51 |
| F1 | 0.64 |

Recall is prioritized over precision — in maintenance, missing a real failure is far costlier than a false alarm.

## Project structure
```
predictive-maintenance/
├── data/ai4i2020.csv
├── models/
│   ├── train_model.py          # retrain from scratch
│   ├── model.pkl, encoder.pkl  # trained artifacts
│   └── metrics.json, feature_importance.csv
├── db/
│   ├── schema.sql
│   └── seed_demo_data.py       # populates demo machines for first run
├── app/
│   ├── Home.py                 # fleet overview
│   ├── utils.py                # shared DB + model logic
│   └── pages/
│       ├── 1_Add_Machine.py
│       ├── 2_Machine_Detail.py
│       ├── 3_Predictions.py
│       ├── 4_Maintenance_Log.py
│       └── 5_Model_Insights.py
└── requirements.txt
```

## Running locally
```bash
pip install -r requirements.txt

# (optional) retrain the model
cd models && python train_model.py && cd ..

# seed demo data so the app isn't empty on first launch
cd db && python seed_demo_data.py && cd ..

# launch
cd app && streamlit run Home.py
```

## Deploying to Streamlit Community Cloud (free)
1. Push this repo to a **public GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io) → sign in with GitHub
3. Click **New app** → select the repo, branch `main`, main file path `app/Home.py`
4. Click **Deploy**

That's it — no SDK selection, no Docker, no billing. The app self-bootstraps on first launch: `ensure_setup()` in `utils.py` trains the model and seeds demo data automatically if `models/model.pkl` or `db/machines.db` aren't present in the repo, so it works whether or not you commit those generated files.

## How risk scoring works
1. Sensor reading comes in (air temp, process temp, RPM, torque, tool wear)
2. XGBoost model outputs failure probability → shown as risk score (0–100%)
3. A rule-based classifier (mirroring AI4I's documented failure physics) attributes the *likely failure mode* (tool wear / heat dissipation / power / overstrain)
4. A heuristic estimates days-to-failure based on tool wear trajectory and risk level

## License
MIT — dataset licensed separately by UCI/original author (Matzka 2020), used here for demonstration purposes.
