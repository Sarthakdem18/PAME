from src.gnn.build_graph import load_training_data, build_knn_graph
from src.gnn.train_gnn import train_gnn
import torch

embeddings, labels = load_training_data()

data = build_knn_graph(embeddings, labels, k=10)

model = train_gnn(data)

torch.save(model.state_dict(), "data/artifacts/gnn_model.pt")

print("Training complete.")
