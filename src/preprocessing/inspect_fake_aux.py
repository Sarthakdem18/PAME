import pandas as pd

df = pd.read_csv("../../data/aux_fake/fake_news_aux.csv")

print("=== BASIC INFO ===")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

print("\n=== SAMPLE ROWS ===")
print(df.head(5))

print("\n=== UNIQUE VALUES PER COLUMN ===")
for col in df.columns:
    print(f"\nColumn: {col}")
    print(df[col].value_counts().head(10))
