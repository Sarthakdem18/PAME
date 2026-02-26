import pandas as pd
df = pd.read_csv("../../data/aux_hate/hasoc_train.tsv", sep="\t")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head(5))
for col in df.columns:
    print(f"\nColumn: {col}")
    print(df[col].value_counts().head(10))
