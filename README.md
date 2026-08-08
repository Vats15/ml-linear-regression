# Linear Regression — Machine Learning Practice

Hands-on implementation and experimentation with Linear Regression using two real-world datasets.

The goal of these projects was to understand the complete machine learning workflow rather than simply train a model:

- Data inspection
- Data cleaning
- Missing-value handling
- Invalid-value detection
- Categorical encoding
- Exploratory data analysis
- Train/test splitting
- Model training
- Regression evaluation
- Feature analysis
- Feature selection
- Prediction visualization

## Projects

### 1. Student Performance Prediction

Predicting students' final grades using academic and behavioral features.

The dataset contained several data-quality issues, including:

- Missing values
- Negative study-hour values
- Attendance values above 100%
- Missing target values

After preprocessing, Linear Regression was trained and evaluated.

**Results:**

| Metric | Result |
|---|---:|
| MAE | 8.39 |
| MSE | 93.30 |
| R² | -0.016 |

The model performed poorly because the available features showed very weak relationships with the target.

This project demonstrated that poor model performance can result from insufficient predictive signal in the dataset rather than from the choice of algorithm.

---

### 2. Medical Cost Prediction

Predicting individual medical insurance charges using demographic and health-related features.

Features include:

- Age
- Sex
- BMI
- Number of children
- Smoking status
- Region

Categorical variables were handled using binary encoding and one-hot encoding.

**Results:**

| Metric | Result |
|---|---:|
| MAE | $4,181 |
| MSE | 33,596,916 |
| R² | 0.784 |

The model explained approximately 78.4% of the variation in medical charges on the test set.

### Feature Selection Experiment

The model was also trained after removing the `smoker` feature.

| Model | MAE | R² |
|---|---:|---:|
| With smoker | $4,181 | 0.784 |
| Without smoker | $9,067 | 0.162 |

Removing `smoker` caused a substantial drop in performance, demonstrating its strong predictive value in this dataset.

## Key Concepts Learned

- Linear Regression
- Train/test split
- MAE
- MSE
- R²
- Missing-value handling
- Data cleaning
- Binary encoding
- One-hot encoding
- Exploratory data analysis
- Feature selection
- Model coefficients
- Actual vs predicted visualization

## Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

## Repository Structure

```text
ml-linear-regression/
│
├── student_performance/
│   ├── src/
│   │   └── train.py
│   ├── data/
│   │   └── student_performance_data.csv
│   └── README.md
│
├── medical_cost_prediction/
│   ├── src/
│   │   └── train.py
│   ├── data/
│   │   └── insurance.csv
│   └── README.md
│
├── README.md
├── requirements.txt
└── .gitignore