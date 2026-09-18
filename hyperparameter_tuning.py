import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# Load data
df = pd.read_csv("data/student-mat.csv", sep=";")


# Features
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


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Base model
model = RandomForestRegressor(
    random_state=42
)


# Hyperparameters to test
param_grid = {
    "n_estimators": [50, 100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10]
}


# Grid search
grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring="r2",
    n_jobs=-1
)


# Train all combinations
grid_search.fit(X_train, y_train)


# Best model
best_model = grid_search.best_estimator_


# Predict
predictions = best_model.predict(X_test)


# Evaluate
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("Best Parameters")
print("----------------")
print(grid_search.best_params_)

print("\nTuned Random Forest Results")
print("---------------------------")
print("MAE:", mae)
print("MSE:", mse)
print("R²:", r2)