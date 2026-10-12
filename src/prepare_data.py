import os
import pandas as pd
from sklearn.datasets import load_breast_cancer

os.makedirs("data", exist_ok=True)
data = load_breast_cancer(as_frame=True)
df = data.frame
df.to_csv("data/dataset.csv", index=False)
print("Saved data/dataset.csv")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Target column: target (0=malignant, 1=benign)")
