import torch
import torch.nn as nn
from pathlib import Path

from src.encoder.frozen_encoder import FrozenEncoder
from src.model.pame_classifier import PAMEClassifier


# ---------------- Paths ----------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"


def euclidean_distance(x, y):
    return torch.norm(x - y, p=2, dim=1)


def main():
    # Load centroids
    centroids = torch.load(DATA_DIR / "centroids.pt", weights_only=False)

    hate_c = centroids["hate"]
    non_hate_c = centroids["non_hate"]
    fake_c = centroids["fake"]
    real_c = centroids["real"]

    # Load training features (quick retrain)
    data = torch.load(DATA_DIR / "pame_features.pt", weights_only=False)

    embeddings = data["embeddings"]
    distances = data["distances"]
    labels = torch.tensor(data["labels"], dtype=torch.float32)

    X = torch.cat([embeddings, distances], dim=1)

    # Train model quickly
    model = PAMEClassifier(input_dim=X.shape[1])
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for _ in range(15):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    model.eval()

    # Encoder
    encoder = FrozenEncoder()

    print("\n🧠 PAME Interactive Inference")
    print("Type text (or 'exit'):\n")

    while True:
        text = input("> ")
        if text.lower() == "exit":
            break

        # Encode text
        emb = encoder.encode([text])

        # Compute distances
        d_hate = euclidean_distance(emb, hate_c.unsqueeze(0))
        d_nonhate = euclidean_distance(emb, non_hate_c.unsqueeze(0))
        d_fake = euclidean_distance(emb, fake_c.unsqueeze(0))
        d_real = euclidean_distance(emb, real_c.unsqueeze(0))

        features = torch.cat(
            [emb, torch.stack([d_hate, d_nonhate, d_fake, d_real], dim=1)],
            dim=1
        )

        with torch.no_grad():
            logits = model(features)
            probs = torch.sigmoid(logits)[0]

        fake_prob = probs[0].item()
        hate_prob = probs[1].item()

        print("\n Prediction:")
        print(f"Fake Probability: {fake_prob:.3f}")
        print(f"Hate Probability: {hate_prob:.3f}")

        if fake_prob > 0.5 and hate_prob > 0.5:
            print("  FAUX-HATE DETECTED")
        elif fake_prob > 0.5:
            print(" Fake News")
        elif hate_prob > 0.5:
            print(" Hate Speech")
        else:
            print(" Normal Content")

        print("-" * 40)


if __name__ == "__main__":
    main()
