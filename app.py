from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "campus_crowd.csv"
MODEL_PATH = BASE_DIR / "models" / "crowd_model.joblib"

st.set_page_config(
    page_title="Smart Campus Crowd Predictor",
    page_icon="🚌",
    layout="wide",
)

st.title("🚌 Smart Campus Crowd Predictor")
st.write(
    "Predict the expected crowd level at a campus location using "
    "time, day, weather, exam/event information, and previous crowd conditions."
)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

st.divider()

left, right = st.columns(2)

with left:
    location = st.selectbox(
        "📍 Campus Location",
        sorted(df["Location"].unique()),
    )
    day = st.selectbox(
        "📅 Day",
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
    )
    hour = st.slider(
        "🕐 Time",
        min_value=7,
        max_value=21,
        value=13,
        step=1,
        format="%d:00",
    )

with right:
    weather = st.selectbox(
        "🌦️ Weather",
        ["Normal", "Rain", "Hot"],
    )
    exam_week = st.selectbox(
        "📚 Exam Week?",
        ["No", "Yes"],
    )
    event_day = st.selectbox(
        "🎉 Special Event Day?",
        ["No", "Yes"],
    )

previous_crowd = st.selectbox(
    "👥 Previous Crowd Level",
    ["Low", "Medium", "High"],
)

if st.button("🔮 Predict Crowd", use_container_width=True):
    input_data = pd.DataFrame(
        [{
            "Location": location,
            "Day": day,
            "Hour": hour,
            "Weather": weather,
            "Exam_Week": exam_week,
            "Event_Day": event_day,
            "Previous_Crowd": previous_crowd,
        }]
    )

    prediction = model.predict(input_data)[0]
    probabilities = model.predict_proba(input_data)[0]
    classes = model.classes_
    probability_map = dict(zip(classes, probabilities))
    confidence = probability_map[prediction]

    st.subheader("Prediction")

    if prediction == "High":
        st.error("🔴 HIGH CROWD")
        st.info("Consider avoiding this location if your goal is to reduce waiting time.")
    elif prediction == "Medium":
        st.warning("🟡 MEDIUM CROWD")
        st.info("The location may be moderately busy.")
    else:
        st.success("🟢 LOW CROWD")
        st.info("The location is expected to be relatively less crowded.")

    st.metric("Model confidence for predicted class", f"{confidence:.1%}")

    probability_df = pd.DataFrame(
        {
            "Crowd Level": classes,
            "Probability": probabilities,
        }
    ).set_index("Crowd Level")

    st.subheader("Prediction Probabilities")
    st.bar_chart(probability_df)

    st.caption(
        "Confidence is the model's predicted probability for the selected class. "
        "The current dataset is synthetic and is intended as a prototype."
    )

st.divider()

st.subheader("📊 Dataset Overview")
m1, m2, m3 = st.columns(3)
m1.metric("Records", len(df))
m2.metric("Locations", df["Location"].nunique())
m3.metric("Crowd Classes", df["Crowd_Level"].nunique())

st.subheader("Sample Historical Data")
st.dataframe(df.head(10), use_container_width=True)

st.caption(
    "Dataset note: this prototype uses reproducible synthetic data based on "
    "reasonable campus crowd patterns. It should be replaced or recalibrated "
    "with actual occupancy/count data before real-world deployment."
)
