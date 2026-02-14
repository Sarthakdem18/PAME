import torch
from pathlib import Path

from src.encoder.frozen_encoder import FrozenEncoder


# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return torch.norm(x - y, p=2, dim=1)


def main():
    # Load centroids
    centroids = torch.load(DATA_DIR / "centroids.pt")

    hate_c = centroids["hate"]
    non_hate_c = centroids["non_hate"]
    fake_c = centroids["fake"]
    real_c = centroids["real"]

    # Load Faux-Hate data
    import pandas as pd
    df = pd.read_csv(DATA_DIR / "faux_hate.csv")

    texts = df["text"].tolist()

    encoder = FrozenEncoder()
    embeddings = encoder.encode(texts)

    # Compute distances
    d_hate = euclidean_distance(embeddings, hate_c.unsqueeze(0))
    d_nonhate = euclidean_distance(embeddings, non_hate_c.unsqueeze(0))
    d_fake = euclidean_distance(embeddings, fake_c.unsqueeze(0))
    d_real = euclidean_distance(embeddings, real_c.unsqueeze(0))

    # Stack distance features
    distance_features = torch.stack(
        [d_hate, d_nonhate, d_fake, d_real],
        dim=1
    )

    print("Embedding shape:", embeddings.shape)
    print("Distance feature shape:", distance_features.shape)
    print("Sample distance vector:", distance_features[0])

    # Save features
    torch.save(
        {
            "embeddings": embeddings,
            "distances": distance_features,
            "labels": df[["fake", "hate"]].values
        },
        DATA_DIR / "pame_features.pt"
    )

    print("\n Proximity-aware features saved to data/pame_features.pt")


if __name__ == "__main__":
    main()
