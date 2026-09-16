from pathlib import Path
import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "campus_crowd.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "crowd_model.joblib"

df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["Crowd_Level"])
y = df["Crowd_Level"]

categorical_features = [
    "Location",
    "Day",
    "Weather",
    "Exam_Week",
    "Event_Day",
    "Previous_Crowd",
]
numeric_features = ["Hour"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        ("numeric", "passthrough", numeric_features),
    ]
)

classifier = RandomForestClassifier(
    n_estimators=250,
    max_depth=10,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)

print("\n========== MODEL EVALUATION ==========")
print(f"Test Accuracy: {accuracy:.2%}")
print("\nClassification Report:")
print(classification_report(y_test, predictions, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))

MODEL_DIR.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)

print(f"\nModel saved to: {MODEL_PATH}")
print("Training completed successfully.")
