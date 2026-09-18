import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# -----------------------------
# 1. Load dataset
# -----------------------------

df = pd.read_csv("data/student-mat.csv", sep=";")


# -----------------------------
# 2. Select features
# -----------------------------

features = [
    "age",
    "studytime",
    "failures",
    "absences",
    "G1",
    "G2"
]

X = df[features]
y = df["G3"]


# -----------------------------
# 3. Train / test split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# -----------------------------
# 4. Random Forest
# -----------------------------

model = RandomForestRegressor(
    random_state=42
)


# -----------------------------
# 5. Hyperparameter search
# -----------------------------

param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5, 10]
}


grid_search = GridSearchCV(
    model,
    param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)


grid_search.fit(X_train, y_train)


# -----------------------------
# 6. Select best model
# -----------------------------

best_model = grid_search.best_estimator_

joblib.dump(best_model, "student_model.pkl")

print("\nModel saved successfully!")

print("Best Parameters:")
print(grid_search.best_params_)


# -----------------------------
# 7. Test predictions
# -----------------------------

predictions = best_model.predict(X_test)


# -----------------------------
# 8. Evaluation
# -----------------------------

mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)

r2 = r2_score(y_test, predictions)


print("\nFinal Model Results")
print("-------------------")

print("MAE :", mae)
print("MSE :", mse)
print("RMSE:", rmse)
print("R²  :", r2)


# -----------------------------
# 9. Actual vs predicted
# -----------------------------

results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\nActual vs Predicted:")
print(results.head(10))