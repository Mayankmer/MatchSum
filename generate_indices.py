import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.cuda.amp import autocast

class LegalIndexGenerator:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _process_batch(self, batch):
        """Process batch of sentences with mixed precision"""
        with torch.no_grad(), autocast():
            inputs = self.tokenizer(
                batch,
                max_length=512,
                truncation=True,
                padding='longest',
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            return torch.norm(outputs.last_hidden_state[:, 0], dim=1).cpu()

    def generate_indices(self, output_path, top_k=10):
        """Generate indices for IN-ABS dataset"""
        dataset = load_dataset("percins/IN-ABS", split="train")
        
        indices = []
        for doc in tqdm(dataset):
            # Access dataset fields directly
            document = doc["text"]  # Changed from 'case' to 'text'
            file_name = doc["file"]  # Changed from 'id' to 'file'
            
            # Split document into sentences
            sentences = [sent.strip() for sent in document if sent.strip()]
            
            # Score sentences in batches
            scores = []
            batch_size = 256
            for i in range(0, len(sentences), batch_size):
                batch = sentences[i:i+batch_size]
                batch_scores = self._process_batch(batch)
                scores.extend(batch_scores.numpy())
            
            # Store results
            ranked_indices = np.argsort(scores)[::-1][:top_k].tolist()
            indices.append({
                "file": file_name,
                "sent_id": ranked_indices,
                "scores": [scores[i] for i in ranked_indices]
            })
            
            # Clear GPU cache periodically
            if len(indices) % 100 == 0:
                torch.cuda.empty_cache()
        
        # Save indices
        with open(output_path, "w") as f:
            for item in indices:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    generator = LegalIndexGenerator()
    generator.generate_indices(
        output_path="./legal_sentence_indices.jsonl",
        top_k=10
    )