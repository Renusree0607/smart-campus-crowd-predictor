from pathlib import Path
import numpy as np
import pandas as pd


# Reproducible synthetic campus-crowd data generator

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "data" / "campus_crowd.csv"

rng = np.random.default_rng(42)

# Possible values
locations = [
    "Canteen",
    "Library",
    "Bus Stop",
    "Cafeteria",
    "Main Block",
    "Sports Complex",
]

days = [
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
]

weathers = [
    "Normal",
    "Rain",
    "Hot",
]

exam_options = ["No", "Yes"]
event_options = ["No", "Yes"]
previous_options = ["Low", "Medium", "High"]

# Base crowd tendency for each location
location_base = {
    "Canteen": 0.65,
    "Library": 0.45,
    "Bus Stop": 0.55,
    "Cafeteria": 0.60,
    "Main Block": 0.50,
    "Sports Complex": 0.40,
}

# Number of records
n_records = 10000

records = []

for _ in range(n_records):

    location = rng.choice(locations)
    day = rng.choice(days)
    hour = int(rng.integers(7, 22))
    weather = rng.choice(weathers)
    exam_week = rng.choice(exam_options, p=[0.75, 0.25])
    event_day = rng.choice(event_options, p=[0.88, 0.12])
    previous_crowd = rng.choice(
        previous_options,
        p=[0.30, 0.45, 0.25],
    )

    score = location_base[location]

    # Morning and lunch/afternoon peak periods
    if 8 <= hour <= 10:
        score += 0.12

    if 12 <= hour <= 14:
        score += 0.18

    if 16 <= hour <= 18:
        score += 0.15

    # Early morning / late evening tend to be quieter
    if hour <= 7 or hour >= 20:
        score -= 0.15

    # Saturday tends to have lower campus activity
    if day == "Saturday":
        score -= 0.12

    # Weather effects
    if weather == "Rain":
        score -= 0.10

    elif weather == "Hot":
        score -= 0.05

    # Exam week
    if exam_week == "Yes":
        score += 0.10

        # Library gets additional activity during exams
        if location == "Library":
            score += 0.18

    # Events increase campus activity
    if event_day == "Yes":
        score += 0.20

    # Previous crowd influences current crowd
    if previous_crowd == "High":
        score += 0.15

    elif previous_crowd == "Medium":
        score += 0.05

    elif previous_crowd == "Low":
        score -= 0.05

    # Small random variation
    score += rng.normal(0, 0.12)

    # Keep score within reasonable range
    score = np.clip(score, 0, 1)

    records.append(
        {
            "Location": location,
            "Day": day,
            "Hour": hour,
            "Weather": weather,
            "Exam_Week": exam_week,
            "Event_Day": event_day,
            "Previous_Crowd": previous_crowd,
            "_score": score,
        }
    )


df = pd.DataFrame(records)

# Create balanced Low / Medium / High target classes
low_threshold = df["_score"].quantile(0.33)
high_threshold = df["_score"].quantile(0.67)


def assign_crowd(score):
    if score <= low_threshold:
        return "Low"
    elif score <= high_threshold:
        return "Medium"
    else:
        return "High"


df["Crowd_Level"] = df["_score"].apply(assign_crowd)

# Add a small amount of label noise
# This prevents the dataset from being perfectly rule-based.
noise_mask = rng.random(len(df)) < 0.05

for index in df.index[noise_mask]:
    current = df.loc[index, "Crowd_Level"]

    alternatives = [
        level for level in ["Low", "Medium", "High"]
        if level != current
    ]

    df.loc[index, "Crowd_Level"] = rng.choice(alternatives)

# Remove internal score column
df = df.drop(columns=["_score"])

# Save dataset
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print(f"Generated {len(df)} records at:")
print(OUTPUT_PATH)

print("\nDataset shape:")
print(df.shape)

print("\nCrowd level distribution:")
print(df["Crowd_Level"].value_counts())

print("\nFirst 5 records:")
print(df.head())