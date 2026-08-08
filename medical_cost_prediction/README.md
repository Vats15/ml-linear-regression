# Medical Cost Prediction

## Objective

Predict individual medical insurance charges using demographic and health-related features.

## Dataset

The dataset contains:

- Age
- Sex
- BMI
- Children
- Smoker
- Region
- Charges

## Preprocessing

The dataset contained no missing values.

Categorical features were handled as follows:

- `sex` → binary encoding
- `smoker` → binary encoding
- `region` → one-hot encoding

For `region`, one category was dropped and used as the reference category.

## Model

Linear Regression was trained using an 80/20 train-test split.

## Results

| Metric | Result |
|---|---:|
| MAE | $4,181 |
| MSE | 33,596,916 |
| R² | 0.784 |

The model explains approximately 78.4% of the variation in medical charges on the test set.

## Feature Analysis

The learned coefficients showed that `smoker` had the largest positive coefficient.

### Feature Selection Experiment

A second model was trained after removing the `smoker` feature.

| Model | MAE | R² |
|---|---:|---:|
| With smoker | $4,181 | 0.784 |
| Without smoker | $9,067 | 0.162 |

Removing `smoker` substantially reduced model performance.

This demonstrated the importance of evaluating features based on their contribution to model performance.

## Visualization

Actual-vs-predicted plots were used to visually evaluate the regression model.

A scatter plot of smoking status against medical charges was also used to investigate the strong relationship between the feature and target.

## Key Learning

This project demonstrated the complete basic regression workflow:

**Data → Preprocessing → Train/Test Split → Model → Evaluation → Feature Analysis → Experimentation**