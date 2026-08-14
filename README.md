# 🏀 NCAA Live Win Probability Predictor
## Overview
An end-to-end machine learning pipeline and interactive web application designed to forecast live NCAA basketball game outcomes. This project uses chronological, play-by-play event logs and applies an advanced Extreme Gradient Boosting (XGBoost) classification model to evaluate game states in real-time.

The final product translates complex predictive math into an intuitive, user-friendly Streamlit dashboard.

## Tech Stack & Tools
Languages: Python, SQL

Machine Learning: XGBoost, Scikit-Learn (Logistic Regression, GroupShuffleSplit)

Data Manipulation: Pandas, NumPy

Data Visualization: Matplotlib

Deployment: Streamlit

## Model Performance
Baseline Model (Logistic Regression): 0.9003 AUC

Advanced Model (XGBoost): 0.9038 AUC

## Methodology & Pipeline
### 1. Data Engineering & Feature Creation
I engineered new, highly predictive continuous variables to capture the true state of the game to bring the play-by-play data into context:

home_score_differential: A dynamic lead/deficit tracker calculated minute-by-minute.

total_seconds_remaining: A unified countdown clock consolidating period, minute, and second data to allow the algorithm to accurately learn time-decay logic.

### 2. Strict Data Leakage Prevention
The data was split by game_id using GroupShuffleSplit to ensure the appropriate information was captured in the datasets. Non predictive  or target leaking variables (such as game_id, home_name, away_name, and raw game_clock strings) were removed to prevent data leakage. The model remained entirely team-agnostic and evaluated pure mathematical game states.

### 3. Algorithm Selection & Tuning
A Logistic Regression baseline was established to capture obvious linear trends, achieving a strong initial AUC. Non-linear momentum shifts, such as dramatic late-game scoring run, the architecture was leveled up to an XGBoost classifier. Hyperparameters (max_depth=5, learning_rate=0.1) were explicitly tuned to prevent overfitting, mathematically validated by comparing Training AUC (0.8883) against Testing AUC (0.9038).

### 4. Deployment & Interactive UI
The optimized model was serialized and deployed into a live, interactive web application using Streamlit. The dashboard allows users to input hypothetical game scenarios via custom UI elements, instantly returning dynamic win-probability metrics and visual progress indicators.

## Visualizations
feature_importance.png: Demonstrates the weighted gain of custom engineered features.


espn_style_probability.png: A minute-by-minute timeline mapping score differential against dynamic win probability.

## How to Run Locally
To run this application on your local machine:

Clone the repository.

Ensure you have streamlit, xgboost, and pandas installed in your environment.

Navigate to the project directory in your terminal.

Run the command: streamlit run app.py
