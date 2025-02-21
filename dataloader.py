import torch
import json
from torch.utils.data import Dataset
from datasets import load_dataset

class LegalDataset(Dataset):
    def __init__(self, data_path, scores_path, tokenizer, max_length=512):
        self.dataset = load_dataset("percins/IN-ABS", data_files=data_path)['train']
        with open(scores_path) as f:
            self.scores = [json.loads(line)['score'] for line in f]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        case = ' '.join(self.dataset[idx]['case'])
        scores = self.scores[idx][:512]  # Truncate to match model output
        
        case_enc = self.tokenizer(
            case,
            max_length=self.max_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': case_enc['input_ids'].squeeze(),
            'attention_mask': case_enc['attention_mask'].squeeze(),
            'labels': torch.FloatTensor(scores)
        }