import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class StudentWrapper(nn.Module):
    def __init__(self, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        super().__init__()
        # Load the tokenizer and transformer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.proj: nn.Linear | None = None

    def set_output_dim(self, out_dim: int):
        in_dim = self.transformer.config.hidden_size
        if self.proj is None or self.proj.out_features != out_dim:
            device = next(self.parameters()).device
            self.proj = nn.Linear(in_dim, out_dim).to(device)

    def forward(self, sentences):
        device = next(self.transformer.parameters()).device

        inputs = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward through transformer
        outputs = self.transformer(**inputs)  # last_hidden_state: [batch, seq_len, hidden_dim]

        # Mean pooling over the sequence length dimension
        attention_mask = inputs['attention_mask']
        mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
        sum_mask = mask_expanded.sum(1)
        embeddings = sum_embeddings / sum_mask  # [batch_size, hidden_dim]

        if self.proj is not None:
            embeddings = self.proj(embeddings)

        return embeddings