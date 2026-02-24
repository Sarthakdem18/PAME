import pandas as pd

# Load dataset
df = pd.read_csv("../../data/faux_hate.csv")

print("=== BASIC INFO ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\n=== SAMPLE ROWS ===")
print(df.head(10))

print("\n=== LABEL DISTRIBUTION ===")
print("\nFake label counts:")
print(df["fake"].value_counts())

print("\nHate label counts:")
print(df["hate"].value_counts())

print("\n=== CHECK FOR MISSING VALUES ===")
print(df.isnull().sum())
