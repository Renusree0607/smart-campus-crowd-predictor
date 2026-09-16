from pathlib import Path
import numpy as np
import pandas as pd

# Reproducible synthetic campus-crowd data generator.
# This is used because an official campus occupancy dataset was not available.

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_PATH = BASE_DIR / "data" / "campus_crowd.csv"

rng = np.random.default_rng(42)

locations = ["Canteen", "Library", "Bus Stop", "Cafeteria", "Main Block", "Sports Complex"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
weathers = ["Normal", "Rain", "Hot"]

location_base = {
    "Canteen": 1.8,
    "Library": 1.7,
    "Bus Stop": 1.7,
    "Cafeteria": 1.6,
    "Main Block": 1.4,
    "Sports Complex": 1.2,
}

rows = []
scores = []

for _ in range(2000):
    loc = rng.choice(locations, p=[0.18, 0.18, 0.17, 0.16, 0.16, 0.15])
    day = rng.choice(days, p=[0.18, 0.18, 0.18, 0.18, 0.18, 0.10])
    hour = int(rng.integers(7, 22))
    weather = rng.choice(weathers, p=[0.65, 0.20, 0.15])
    exam = "Yes" if rng.random() < 0.22 else "No"
    event = "Yes" if rng.random() < 0.15 else "No"

    score = location_base[loc]
    if 12 <= hour <= 14:
        score += 1.1
    if 17 <= hour <= 19:
        score += 1.3
    if hour in [8, 9]:
        score += 0.5

    if day == "Saturday":
        score -= 0.5
    if weather == "Rain":
        score -= 0.6
    elif weather == "Hot":
        score -= 0.2

    if exam == "Yes" and loc == "Library":
        score += 1.1
    elif exam == "Yes":
        score += 0.25

    if event == "Yes":
        score += 1.0

    previous_score = score + rng.normal(0, 0.4)
    if previous_score >= 3.8:
        previous = "High"
    elif previous_score >= 2.7:
        previous = "Medium"
    else:
        previous = "Low"

    target_score = score + {"Low": -0.3, "Medium": 0.15, "High": 0.45}[previous]
    target_score += rng.normal(0, 0.3)

    rows.append([loc, day, hour, weather, exam, event, previous, target_score])
    scores.append(target_score)

q1, q2 = np.quantile(scores, [0.45, 0.78])
labels = ["Low" if s < q1 else "Medium" if s < q2 else "High" for s in scores]

df = pd.DataFrame(
    [row[:-1] + [label] for row, label in zip(rows, labels)],
    columns=[
        "Location", "Day", "Hour", "Weather",
        "Exam_Week", "Event_Day", "Previous_Crowd", "Crowd_Level"
    ],
)

# Add 5% label noise so the data is not unrealistically perfect.
noise_idx = rng.choice(len(df), size=int(len(df) * 0.05), replace=False)
for idx in noise_idx:
    current = df.at[idx, "Crowd_Level"]
    alternatives = [x for x in ["Low", "Medium", "High"] if x != current]
    df.at[idx, "Crowd_Level"] = rng.choice(alternatives)

OUTPUT_PATH.parent.mkdir(exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)
print(f"Generated {len(df)} records at {OUTPUT_PATH}")
