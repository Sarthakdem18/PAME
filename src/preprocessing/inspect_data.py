import pandas as pd

# Load dataset
df = pd.read_csv("../../data/faux_hate.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(10))
print("\nFake label counts:")
print(df["fake"].value_counts())

print(df["hate"].value_counts())

print(df.isnull().sum())
