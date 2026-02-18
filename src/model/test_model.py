import torch
from src.model.pame_classifier import PAMEClassifier
model = PAMEClassifier()
dummy_input = torch.randn(5, 772)
output = model(dummy_input)
print("Output shape:", output.shape)
