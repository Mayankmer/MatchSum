import torch
import torch.nn as nn
from transformers import AutoModel

class LegalMatchSum(nn.Module):
    def __init__(self):
        super().__init__()
        self.legal_bert = AutoModel.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.ranking_layer = nn.Linear(self.legal_bert.config.hidden_size, 512)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        # Validate inputs
        if input_ids is None or attention_mask is None:
            raise RuntimeError("Received None inputs in forward pass")
            
        # Ensure correct dtype
        attention_mask = attention_mask.float()
        
        outputs = self.legal_bert(
            input_ids,
            attention_mask=attention_mask.float(),
            return_dict=True
        )
        pooled_output = outputs.last_hidden_state[:, 0, :]
        scores = self.ranking_layer(self.dropout(pooled_output))
        return scores 