import torch
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from .gnn_model import GCN
from src.encoder.frozen_encoder import FrozenEncoder

def load_train_embeddings(path="data/artifacts/pame_features.pt"):
    obj = torch.load(path)
    train_embeddings = obj["embeddings"]
    train_labels = torch.tensor(obj["labels"], dtype=torch.long)
    return train_embeddings, train_labels

def encode_test_texts(test_file_path):
    df = pd.read_excel(test_file_path)

    # Your column name is "Tweet"
    texts = df["Tweet"].tolist()

    encoder = FrozenEncoder()
    test_embeddings = encoder.encode(texts)

    return test_embeddings, df

def build_inductive_graph(train_emb, test_emb, k=10):
    all_embeddings = torch.cat([train_emb, test_emb], dim=0)

    train_size = train_emb.shape[0]
    total_size = all_embeddings.shape[0]

    emb_np = all_embeddings.cpu().numpy()

    # Fit only on training nodes
    nbrs = NearestNeighbors(n_neighbors=k, metric="cosine")
    nbrs.fit(emb_np[:train_size])

    _, indices = nbrs.kneighbors(emb_np)

    edge_list = []

    for i in range(total_size):
        for j in indices[i]:
            edge_list.append([i, j])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()

    data = Data(
        x=all_embeddings,
        edge_index=edge_index
    )

    return data, train_size

def infer(test_file_path):
    # Load train embeddings
    train_emb, train_labels = load_train_embeddings()

    # Encode test texts
    test_emb, df = encode_test_texts(test_file_path)

    # Build graph
    data, train_size = build_inductive_graph(train_emb, test_emb)

    # Load trained GNN
    model = GCN()
    model.load_state_dict(torch.load("data/artifacts/gnn_model.pt"))
    model.eval()

    with torch.no_grad():
        out = model(data.x, data.edge_index)

    # Extract test predictions
    test_logits = out[train_size:]
    predictions = torch.argmax(test_logits, dim=1)

    df["gnn_prediction"] = predictions.numpy()
    true_labels = ((df["Hate"] == 1) & (df["Fake"] == 1)).astype(int).values
    preds = predictions.numpy()

    accuracy = accuracy_score(true_labels, preds)
    precision = precision_score(true_labels, preds)
    recall = recall_score(true_labels, preds)
    f1 = f1_score(true_labels, preds)
    print("Accuracy :", accuracy)
    print("Precision:", precision)
    print("Recall   :", recall)
    print("F1-score :", f1)

    return df
