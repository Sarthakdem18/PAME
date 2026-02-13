import torch
from pathlib import Path
from src.encoder.frozen_encoder import FrozenEncoder
from src.preprocessing.load_task_a import load_task_a_train

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "artifacts"

def compute_centroid(x):
    return x.mean(dim=0)

def main():
    encoder = FrozenEncoder()
    df = load_task_a_train()

    faux_texts = df[df["faux"] == 1]["text"].tolist()
    non_faux_texts = df[df["faux"] == 0]["text"].tolist()

    faux_emb = encoder.encode(faux_texts)
    non_faux_emb = encoder.encode(non_faux_texts)

    centroids = {
        "faux": compute_centroid(faux_emb),
        "non_faux": compute_centroid(non_faux_emb)
    }

    ARTIFACTS_DIR.mkdir(exist_ok=True)
    torch.save(centroids, ARTIFACTS_DIR / "centroids.pt")

    print("Faux-hate centroids computed")

if __name__ == "__main__":
    main()
