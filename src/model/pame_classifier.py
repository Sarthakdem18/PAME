import torch
import torch.nn as nn

class PAMEClassifier(nn.Module):
    def __init__(self, input_dim=772, hidden_dim=256, dropout=0.3):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        return x.squeeze(1)
