# 🎓 Student Performance AI

A beginner-friendly machine learning project that predicts a student's final
grade (G3) using previous academic performance and selected student information.

## 📌 Project Overview

The goal of this project is to build a machine learning model that predicts
a student's final grade based on:

- Age
- Weekly study time
- Number of past failures
- Number of absences
- First-period grade (G1)
- Second-period grade (G2)

The trained model is integrated into a Streamlit web application where users
can enter student information and receive a predicted final grade.

## 🗂️ Dataset

This project uses the **Student Performance** dataset from the
UCI Machine Learning Repository.

The dataset contains information about students and their academic performance.

The target variable is:

- **G3** — Final grade, ranging from 0 to 20

Two previous grades are used as input features:

- **G1** — First-period grade
- **G2** — Second-period grade

## 🤖 Machine Learning

Several experiments were performed during development, including:

- Linear Regression
- Random Forest Regression
- Feature importance analysis
- Permutation importance
- 5-fold cross-validation
- Hyperparameter tuning

The final model uses a **Random Forest Regressor**.

### Final Model

The selected hyperparameters were:

- `n_estimators = 50`
- `max_depth = 5`
- `min_samples_split = 5`

### Evaluation Results

On the held-out test set:

| Metric | Result |
|---|---:|
| MAE | 1.046 |
| MSE | 2.751 |
| RMSE | 1.659 |
| R² | 0.866 |

The 5-fold cross-validation experiment produced an average R² of approximately
0.879.

> R² is a regression evaluation metric and should not be interpreted as
> classification accuracy.

## 🔍 Feature Importance

The Random Forest model showed that **G2** contained substantially more
predictive signal than the other features in this particular model.

This does not mean that G2 causes the final grade. Feature importance describes
the model's use of the feature for prediction, not causation.

## 🌐 Streamlit Application

The project includes an interactive Streamlit application.

Users can enter:

```text
Age
Study Time
Past Failures
Absences
G1
G2