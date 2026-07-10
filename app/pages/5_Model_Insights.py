import json
import os

import streamlit as st
import pandas as pd
import plotly.express as px
import shap
import matplotlib.pyplot as plt

from utils import BASE_DIR, load_model, ensure_setup

st.set_page_config(page_title="Model Insights", page_icon="🧠", layout="wide")
ensure_setup()
st.title("🧠 Model Insights")
st.caption("Transparency into how the failure-prediction model makes decisions.")

metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
importance_path = os.path.join(BASE_DIR, "models", "feature_importance.csv")

if os.path.exists(metrics_path):
    with open(metrics_path) as f:
        metrics = json.load(f)
    st.subheader("Model Performance")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("F1 Score", metrics["f1"])
    c2.metric("Precision", metrics["precision"])
    c3.metric("Recall", metrics["recall"])
    c4.metric("ROC-AUC", metrics["roc_auc"])
    st.caption(f"Trained on {metrics['train_rows']} rows, tested on {metrics['test_rows']} rows from the AI4I 2020 dataset.")

st.divider()

if os.path.exists(importance_path):
    st.subheader("Feature Importance")
    imp_df = pd.read_csv(importance_path)
    fig = px.bar(imp_df, x="importance", y="feature", orientation="h",
                 title="What drives the model's failure predictions",
                 labels={"importance": "Importance", "feature": "Feature"})
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.subheader("SHAP Explainability")
st.caption("Shows how each feature pushes an individual prediction toward 'failure' or 'no failure'.")

data_path = os.path.join(BASE_DIR, "data", "ai4i2020.csv")
if os.path.exists(data_path):
    model, encoder = load_model()
    df = pd.read_csv(data_path).rename(columns={
        "Air temperature [K]": "air_temp",
        "Process temperature [K]": "process_temp",
        "Rotational speed [rpm]": "rpm",
        "Torque [Nm]": "torque",
        "Tool wear [min]": "tool_wear",
    })
    df["type_enc"] = encoder.transform(df["Type"])
    features = ["type_enc", "air_temp", "process_temp", "rpm", "torque", "tool_wear"]
    sample = df[features].sample(200, random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, sample, show=False, plot_size=(8, 5))
    st.pyplot(fig, use_container_width=True)
else:
    st.info("Dataset not found for SHAP sampling.")
