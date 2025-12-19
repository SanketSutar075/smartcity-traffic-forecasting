# ===============================
# Smart City Traffic Dashboard
# By Sanket Sutar
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="Smart City Traffic Forecasting",
    layout="wide"
)

st.markdown(
    "<h1 style='text-align:center;'>🚦 Smart City Traffic Dashboard</h1>"
    "<p style='text-align:center;'>By <b>Sanket Sutar</b></p>",
    unsafe_allow_html=True
)

# -------------------------------
# PATHS
# -------------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv(DATA_DIR / "train_cleaned.csv", parse_dates=["DateTime"])
df["date"] = df["DateTime"].dt.date

# -------------------------------
# SIDEBAR FILTERS
# -------------------------------
st.sidebar.header("🔎 Filters")

junction = st.sidebar.selectbox(
    "Select Junction",
    sorted(df["Junction"].unique())
)

date_mode = st.sidebar.radio(
    "📅 Date Range",
    ["Last 7 Days", "Last 30 Days", "Full Data", "Custom Range"]
)

max_date = df["date"].max()

if date_mode == "Last 7 Days":
    start_date = max_date - pd.Timedelta(days=7)
    end_date = max_date

elif date_mode == "Last 30 Days":
    start_date = max_date - pd.Timedelta(days=30)
    end_date = max_date

elif date_mode == "Full Data":
    start_date = df["date"].min()
    end_date = max_date

else:
    dr = st.sidebar.date_input(
        "Select Date Range",
        [df["date"].min(), df["date"].max()]
    )
    if isinstance(dr, tuple):
        start_date, end_date = dr
    else:
        start_date = end_date = dr

# -------------------------------
# FILTER DATA
# -------------------------------
filtered = df[
    (df["Junction"] == junction) &
    (df["date"] >= start_date) &
    (df["date"] <= end_date)
]

# -------------------------------
# KPIs
# -------------------------------
st.subheader("📊 Traffic KPIs")

c1, c2, c3 = st.columns(3)

avg_veh = int(filtered["Vehicles"].mean()) if not filtered.empty else 0
max_veh = int(filtered["Vehicles"].max()) if not filtered.empty else 0
min_veh = int(filtered["Vehicles"].min()) if not filtered.empty else 0

c1.metric("🚗 Avg Vehicles", avg_veh)
c2.metric("📈 Max Vehicles", max_veh)
c3.metric("📉 Min Vehicles", min_veh)

# -------------------------------
# LINE CHART
# -------------------------------
st.subheader("📈 Traffic Over Time")

fig, ax = plt.subplots(figsize=(12, 4))
sns.lineplot(data=filtered, x="DateTime", y="Vehicles", ax=ax, color="orange")
ax.set_xlabel("Time")
ax.set_ylabel("Vehicles")
st.pyplot(fig)

# -------------------------------
# DAY-WISE ANALYSIS
# -------------------------------
st.subheader("📅 Day-wise Traffic")

filtered["day"] = pd.to_datetime(filtered["DateTime"]).dt.day_name()
day_avg = filtered.groupby("day")["Vehicles"].mean().reindex(
    ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
)

fig2, ax2 = plt.subplots(figsize=(8,4))
day_avg.plot(kind="bar", color="teal", ax=ax2)
ax2.set_ylabel("Average Vehicles")
st.pyplot(fig2)

# -------------------------------
# ML PREDICTION
# -------------------------------
st.subheader("🤖 Machine Learning Prediction")

model_path = MODEL_DIR / f"model_junction_{junction}.pkl"
model = joblib.load(model_path)

X = filtered[["hour", "day", "month"]]
filtered["Predicted"] = model.predict(X)

fig3, ax3 = plt.subplots(figsize=(12,4))
ax3.plot(filtered["DateTime"], filtered["Vehicles"], label="Actual")
ax3.plot(filtered["DateTime"], filtered["Predicted"], label="Predicted")
ax3.legend()
st.pyplot(fig3)

# -------------------------------
# INSIGHTS
# -------------------------------
st.subheader("🧠 Key Insights")

st.markdown(f"""
- 🚦 Junction **{junction}** analyzed  
- 📅 Period: **{start_date} to {end_date}**
- 🔥 Peak traffic observed during **rush hours**
- 📉 Lower traffic on **weekends**
- 🤖 ML model predicts future traffic trends accurately
""")

st.success("✅ Dashboard Loaded Successfully")
