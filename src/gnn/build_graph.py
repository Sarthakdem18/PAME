import torch
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
def load_training_data(path="data/artifacts/pame_features.pt"):
    obj = torch.load(path)

    embeddings = obj["embeddings"]   # shape: (6396, 768)
    labels = torch.tensor(obj["labels"], dtype=torch.long)         # shape: (6396,)

    return embeddings, labels
def build_knn_graph(embeddings, labels, k=10):
    emb_np = embeddings.cpu().numpy()

    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
    nbrs.fit(emb_np)

    _, indices = nbrs.kneighbors(emb_np)

    edge_list = []

    for i in range(len(indices)):
        for j in indices[i]:
            edge_list.append([i, j])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data = Data(
        x=embeddings,
        y=labels,
        edge_index=edge_index
    )

    return data
