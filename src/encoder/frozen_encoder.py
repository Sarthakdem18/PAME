import torch
from transformers import AutoTokenizer, AutoModel


class FrozenEncoder:
    def __init__(self, model_name="bert-base-multilingual-cased"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        # Freeze model
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    def encode(self, texts, batch_size=8):
        embeddings = []

        texts = [
            str(t).strip()
            for t in texts
            if t is not None and str(t).strip() != ""
        ]

        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                inputs = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )

                outputs = self.model(**inputs)

                # CLS token embedding
                cls_embeddings = outputs.last_hidden_state[:, 0, :]
                embeddings.append(cls_embeddings)

        return torch.cat(embeddings, dim=0)
