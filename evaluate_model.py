import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


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


# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Train
model = LinearRegression()
model.fit(X_train, y_train)


# Predict
predictions = model.predict(X_test)


# Evaluation
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R²:", r2)


# Actual vs Predicted
plt.scatter(y_test, predictions)

plt.xlabel("Actual G3")
plt.ylabel("Predicted G3")
plt.title("Actual vs Predicted Final Grade")

# Perfect prediction line
plt.plot([0, 20], [0, 20])

plt.show()