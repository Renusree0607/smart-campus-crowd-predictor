# 🚌 Smart Campus Crowd Predictor

An AIML project that predicts the expected crowd level at a campus location as
**Low, Medium, or High** using structured historical-style campus data.

## Features

- Campus location
- Day of week
- Hour
- Weather
- Exam week status
- Special event status
- Previous crowd level
- Random Forest multi-class classification
- Prediction probabilities
- Streamlit web interface
- Dataset overview

## Dataset

The included dataset contains **2,000 synthetic records**.

It was generated with `src/generate_dataset.py` using reproducible rules based on
plausible campus crowd patterns. It is **not claimed to be real VIT/student data**
and should not be described as collected from students.

The target column is `Crowd_Level` with three classes:
- Low
- Medium
- High

## Machine Learning Pipeline

```text
Synthetic/structured campus data
          ↓
    Feature separation
          ↓
    One-Hot Encoding
          ↓
     Train/Test Split
          ↓
   Random Forest Classifier
          ↓
   Model Evaluation
          ↓
   Saved Joblib Pipeline
          ↓
     Streamlit Prediction App
```

## Run the project

### 1. Create and activate virtual environment

```bash
python -m venv venv
```

Windows Command Prompt:

```cmd
venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Generate the dataset (optional)

```bash
python src/generate_dataset.py
```

### 4. Train the model

```bash
python src/train_model.py
```

### 5. Start the application

```bash
streamlit run app.py
```

Then open the local Streamlit URL shown in the terminal.

## Interview explanation

This is a supervised multi-class classification project. The model learns from
labeled historical-style records where the input features describe campus
conditions and `Crowd_Level` is the target variable.

Random Forest was selected because it is suitable for structured/tabular data and
can model non-linear relationships between the input features and crowd classes.

## Future improvements

- Replace synthetic data with real anonymized occupancy/count data.
- Add real-time sensor or entry-count integration.
- Add time-series forecasting.
- Add historical trend dashboards.
- Compare multiple ML algorithms.
- Deploy the Streamlit app to a cloud platform.
