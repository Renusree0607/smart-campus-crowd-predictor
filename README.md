# 🚌 Smart Campus Crowd Predictor

An AIML-based application that predicts the expected crowd level at a campus location as **Low, Medium, or High** using factors such as location, time, weather, exam schedules, events, and previous crowd conditions.

## 🚀 Live Demo

[Smart Campus Crowd Predictor](https://smart-campus-crowd-predictor-renusree.streamlit.app/)

## ✨ Features

- 📍 Campus location-based prediction
- 📅 Day of the week
- 🕐 Hour-based prediction
- 🌦️ Weather conditions
- 📚 Exam week status
- 🎉 Special event status
- 🔄 Previous crowd level
- 🤖 Random Forest multi-class classification
- 📊 Prediction probability display
- 🌐 Interactive Streamlit web interface

## 🧠 Machine Learning Approach

### Problem Type

**Supervised Learning → Multi-Class Classification**

The model predicts one of three crowd levels:

- Low
- Medium
- High

### Algorithm

**Random Forest Classifier**

Random Forest was used because it is well suited for structured/tabular data and can capture non-linear relationships between input features and crowd levels.

### Data Preprocessing

Categorical features are converted into numerical representations using **One-Hot Encoding**.

The machine learning pipeline includes:

1. Data preprocessing using Pandas
2. Feature and target separation
3. One-Hot Encoding
4. Train-test split
5. Random Forest model training
6. Model evaluation
7. Model serialization using Joblib

## 📊 Dataset

The project uses a **2,000-record synthetic dataset** generated using reproducible rules based on plausible campus crowd patterns.

The dataset contains:

- Location
- Day
- Hour
- Weather
- Exam Week
- Event Day
- Previous Crowd
- Crowd Level

The dataset is **synthetic and is not real VIT/student data**.

For real-world deployment, the model could be retrained using anonymized campus occupancy or footfall data.

## 📈 Model Evaluation

The Random Forest model achieved approximately **74.5% test accuracy** on the synthetic dataset.

This result represents a prototype evaluation and should not be interpreted as real-world campus prediction performance.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Joblib
- Matplotlib

## 📁 Project Structure

```text
Smart-Campus-Crowd-Predictor/
│
├── data/
│   └── campus_crowd.csv
│
├── models/
│   └── crowd_model.joblib
│
├── src/
│   ├── generate_dataset.py
│   └── train_model.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore