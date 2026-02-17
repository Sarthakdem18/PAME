import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F

from src.encoder.frozen_encoder import FrozenEncoder
from src.model.pame_classifier import PAMEClassifier
from src.preprocessing.load_task_a import load_task_a_val

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"


def cosine_sim(a, b):
    return F.cosine_similarity(a, b)


def main():
    df = load_task_a_val()

    texts = df["text"].tolist()
    labels = df["faux"].values

    encoder = FrozenEncoder()
    embeddings = encoder.encode(texts)

    centroids = torch.load(ARTIFACTS_DIR / "centroids.pt")

    faux_c = centroids["faux"].unsqueeze(0)
    non_faux_c = centroids["non_faux"].unsqueeze(0)

    faux_sim = cosine_sim(embeddings, faux_c)
    non_faux_sim = cosine_sim(embeddings, non_faux_c)

    distances = torch.stack([faux_sim, non_faux_sim], dim=1)

    X = torch.cat([embeddings, distances], dim=1)

    model = PAMEClassifier(input_dim=X.shape[1])
    model.load_state_dict(torch.load(ARTIFACTS_DIR / "pame_model.pt"))
    model.eval()

    with torch.no_grad():
        probs = torch.sigmoid(model(X)).squeeze()
        preds = (probs >= 0.25).int().numpy()

    print("\nEvaluation (True Faux-Hate)")
    print(f"Accuracy  : {accuracy_score(labels, preds):.4f}")
    print(f"Precision : {precision_score(labels, preds):.4f}")
    print(f"Recall    : {recall_score(labels, preds):.4f}")
    print(f"F1-score  : {f1_score(labels, preds):.4f}")
    print("Mean risk (faux):", probs[labels == 1].mean().item())
    print("Mean risk (non-faux):", probs[labels == 0].mean().item())


if __name__ == "__main__":
    main()
