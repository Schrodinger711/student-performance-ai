import pandas as pd

from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor


# Load dataset
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


# Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# 5-fold cross-validation
kfold = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


# Calculate R² for each fold
scores = cross_val_score(
    model,
    X,
    y,
    cv=kfold,
    scoring="r2"
)


print("R² scores for each fold:")
print(scores)

print("\nAverage R²:")
print(scores.mean())

print("\nStandard deviation:")
print(scores.std())