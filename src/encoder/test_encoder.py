import pandas as pd
from frozen_encoder import FrozenEncoder

# Load data
df = pd.read_csv("../../data/faux_hate.csv")

texts = df["text"].tolist()

encoder = FrozenEncoder()
embeddings = encoder.encode(texts)

print("Embedding shape:", embeddings.shape)
print("First embedding (first 5 values):")
print(embeddings[0][:5])
