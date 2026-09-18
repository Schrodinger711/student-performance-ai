import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/student-mat.csv", sep=";")

plt.scatter(df["studytime"], df["G3"])

plt.xlabel("Study Time")
plt.ylabel("Final Grade (G3)")
plt.title("Study Time vs Final Grade")

plt.show()