# Student Performance Prediction

## Objective

Predict students' final grades using academic and behavioral features and investigate whether Linear Regression can learn a useful relationship from the dataset.

## Dataset

The dataset contains information such as:

- Study Hours
- Attendance
- Previous Grade
- Gender
- Parental Support
- Extracurricular Activities
- Online Classes
- Final Grade

## Data Cleaning

The dataset contained:

- Missing values
- Negative study-hour values
- Attendance values above 100%
- Missing final grades

Preprocessing included:

- Removing identifier columns
- Removing invalid study-hour and attendance records
- Removing rows with missing target values
- Median imputation for numerical features
- Mode imputation for categorical features
- Binary encoding for binary categorical features
- Encoding parental support categories

## Model

Linear Regression was trained using an 80/20 train-test split.

## Results

| Metric | Result |
|---|---:|
| MAE | 8.39 |
| MSE | 93.30 |
| R² | -0.016 |

## Analysis

The model performed poorly.

Scatter plots and correlation analysis showed very weak relationships between the available features and `FinalGrade`.

The model performed approximately as poorly as a baseline prediction based on the average target value.

## Key Learning

A more complex model cannot compensate for a dataset that does not contain sufficient predictive information.

This project was useful for learning how to diagnose poor model performance rather than blindly trying different algorithms.