import torch
import torch.nn as nn
from torch.optim import Adam
from pathlib import Path

from src.model.pame_classifier import PAMEClassifier

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

def main():
    data = torch.load(DATA_DIR /"artifacts"/"pame_features.pt", weights_only=False)

    embeddings = data["embeddings"]
    distances = data["distances"]
    labels = torch.tensor(data["labels"], dtype=torch.float32)

    X = torch.cat([embeddings, distances], dim=1)
    model = PAMEClassifier(input_dim=X.shape[1])
    pos_weight = torch.tensor(
    [(labels == 0).sum() / (labels == 1).sum()],
    dtype=torch.float32 
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    epochs = 15
    for epoch in range(epochs):
        optimizer.zero_grad()

        outputs = model(X)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {loss.item():.4f}")
    model_path = ARTIFACTS_DIR / "pame_model.pt"
    torch.save(model.state_dict(), model_path)

    print("\n Training completed successfully")
    print(f" Model saved to: {model_path}")

if __name__ == "__main__":
    main()
