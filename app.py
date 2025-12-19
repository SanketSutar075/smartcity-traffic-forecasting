import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Smart City Traffic Dashboard",
    page_icon="🚦",
    layout="wide"
)

# ================= THEME SELECTOR =================
theme = st.sidebar.selectbox(
    "🎨 Select Theme",
    ["Dark", "Light", "Neon"]
)

if theme == "Dark":
    bg = "#0e1117"
    text = "#e0e0e0"
    accent = "#00ffcc"
elif theme == "Light":
    bg = "#ffffff"
    text = "#000000"
    accent = "#0077ff"
else:
    bg = "#050505"
    text = "#e6e6e6"
    accent = "#39ff14"

st.markdown(
    f"""
    <style>
        body {{
            background-color: {bg};
            color: {text};
        }}
        .stMetric {{
            background-color: #1f2937;
            padding: 15px;
            border-radius: 12px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

sns.set_style("darkgrid")

# ================= PATHS =================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"
MODEL_DIR = BASE_DIR / "models"

# ================= TITLE =================
st.markdown(
    f"""
    <div style="position:relative;">
        <h1 style="text-align:center;color:{accent};">
            🚦 Smart City Traffic Pattern Forecasting
        </h1>
        <span style="
            position:absolute;
            top:0;
            right:10px;
            font-size:16px;
            color:gray;
        ">
            By Sanket Nagnath Sutar
        </span>
    </div>
    <h4 style="text-align:center;color:gray;">
        Machine Learning Based Traffic Analysis & Forecasting
    </h4>
    """,
    unsafe_allow_html=True
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    df = pd.read_csv(DATA_DIR / "train_cleaned.csv")
    df["DateTime"] = pd.to_datetime(df["DateTime"])

    df["hour"] = df["DateTime"].dt.hour
    df["day"] = df["DateTime"].dt.day
    df["weekday"] = df["DateTime"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)

    df["weekday_name"] = df["DateTime"].dt.day_name()
    df["date"] = df["DateTime"].dt.date
    return df

df = load_data()

# ================= SIDEBAR FILTERS =================
st.sidebar.header("🔎 Filters")

junction = st.sidebar.selectbox(
    "Select Junction",
    sorted(df["Junction"].unique())
)

# -------- DATE RANGE PRESETS --------
st.sidebar.subheader("📅 Date Range Mode")

date_mode = st.sidebar.radio(
    "Choose Date Range",
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
    date_range = st.sidebar.date_input(
        "Select Custom Date Range",
        [df["date"].min(), max_date]
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

filtered = df[
    (df["Junction"] == junction) &
    (df["date"] >= start_date) &
    (df["date"] <= end_date)
]

# ================= KPIs =================
st.markdown("## 📊 Traffic KPIs")
k1, k2, k3, k4 = st.columns(4)

if filtered.empty:
    k1.metric("🚗 Avg Vehicles", "N/A")
    k2.metric("📈 Max Vehicles", "N/A")
    k3.metric("📉 Min Vehicles", "N/A")
    k4.metric("📅 Records", 0)
    st.warning("⚠️ No data available for selected filters.")
else:
    k1.metric("🚗 Avg Vehicles", int(filtered["Vehicles"].mean()))
    k2.metric("📈 Max Vehicles", int(filtered["Vehicles"].max()))
    k3.metric("📉 Min Vehicles", int(filtered["Vehicles"].min()))
    k4.metric("📅 Records", len(filtered))

# ================= TRAFFIC TREND =================
st.markdown("## 📈 Traffic Trend Over Time")

if not filtered.empty:
    fig1, ax1 = plt.subplots(figsize=(12, 4))
    ax1.plot(filtered["DateTime"], filtered["Vehicles"], color=accent)
    ax1.set_title(f"Traffic Trend - Junction {junction}")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    st.info(
        "📌 **Insight:** Traffic volume changes over time with visible rush-hour peaks."
    )

# ================= HOURLY PATTERN =================
st.markdown("## ⏰ Hourly Traffic Pattern")

if not filtered.empty:
    hourly = filtered.groupby("hour")["Vehicles"].mean().reset_index()
    peak_hour = hourly.loc[hourly["Vehicles"].idxmax(), "hour"]

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    sns.barplot(x="hour", y="Vehicles", data=hourly, palette="viridis", ax=ax2)
    st.pyplot(fig2)

    st.info(
        f"📌 **Insight:** Peak traffic occurs around **{peak_hour}:00 hours**."
    )

# ================= WEEKDAY PATTERN =================
st.markdown("## 📅 Weekday Traffic Pattern")

if not filtered.empty:
    weekday_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    weekday_avg = (
        filtered.groupby("weekday_name")["Vehicles"]
        .mean()
        .reindex(weekday_order)
    )
    busiest_day = weekday_avg.idxmax()

    fig3, ax3 = plt.subplots(figsize=(12, 4))
    sns.barplot(x=weekday_avg.index, y=weekday_avg.values, palette="coolwarm", ax=ax3)
    plt.xticks(rotation=30)
    st.pyplot(fig3)

    st.info(
        f"📌 **Insight:** **{busiest_day}** has the highest traffic; weekends are lighter."
    )

# ================= ML PREDICTION =================
st.markdown("## 🤖 Machine Learning Prediction")

if not filtered.empty:
    model = joblib.load(MODEL_DIR / f"model_junction_{junction}.pkl")
    X = filtered[["hour", "day", "weekday", "is_weekend"]]
    filtered["Predicted"] = model.predict(X)

    fig4, ax4 = plt.subplots(figsize=(12, 4))
    ax4.plot(filtered["DateTime"], filtered["Vehicles"], label="Actual", color="orange")
    ax4.plot(filtered["DateTime"], filtered["Predicted"], label="Predicted", color=accent)
    ax4.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig4)

    st.info(
        "📌 **Insight:** ML predictions closely match actual traffic trends."
    )

# ================= DATA TABLE =================
st.markdown("## 📄 Data Preview")
st.dataframe(filtered.head(100))

# ================= FOOTER =================
st.markdown(
    """
    <hr>
    <div style="text-align:center;color:gray;">
        🚦 Smart City Traffic Pattern Forecasting Dashboard <br>
        <b>Developed by Sanket Nagnath Sutar</b>
    </div>
    """,
    unsafe_allow_html=True
)

st.success("✅ Dashboard Loaded Successfully")
