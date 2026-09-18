import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance


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


# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Create Random Forest
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# Train
model.fit(X_train, y_train)


# Predict
predictions = model.predict(X_test)


#Perform permutation importance
result = permutation_importance(
    model,
    X_test,
    y_test,
    n_repeats=10,
    random_state=42,
    scoring="r2"
)

importance = pd.Series(
    result.importances_mean,
    index=features
)

print("\nPermutation Importance")
print("----------------------")
print(importance.sort_values(ascending=False))


# Evaluate
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)


print("Random Forest Results")
print("---------------------")
print("MAE:", mae)
print("MSE:", mse)
print("R²:", r2)


print("\nFeature Importance")
print("------------------")

importance = pd.Series(
    model.feature_importances_,
    index=features
)

print(importance.sort_values(ascending=False))