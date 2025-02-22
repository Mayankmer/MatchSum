import json
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from torch.cuda.amp import autocast, GradScaler
import multiprocessing as mp

class LegalDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]

class LegalIndexGenerator:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model = torch.nn.DataParallel(self.model)  # Multi-GPU support
        self.model.eval()
        self.scaler = GradScaler()  # For mixed precision
        
        # Enable cudnn benchmarking and deterministic algorithms
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

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

    def _score_sentences(self, document):
        """Batch process sentences with optimal chunk size"""
        batch_size = 256  # Adjusted for T4 memory
        scores = []
        
        # Split document into batches
        for i in range(0, len(document), batch_size):
            batch = document[i:i+batch_size]
            batch_scores = self._process_batch(batch)
            scores.extend(batch_scores.numpy())
            
        return scores

    def generate_indices(self, output_path, top_k=10):
        """Generate indices with parallel processing"""
        dataset = load_dataset("percins/IN-ABS", split="train")
        legal_dataset = LegalDataset(dataset)
        
        # Configure DataLoader for parallel processing
        loader = DataLoader(
            legal_dataset,
            batch_size=8,  # Documents per batch
            shuffle=False,
            num_workers=mp.cpu_count()//2,  # Use half of available CPU cores
            pin_memory=True,
            prefetch_factor=2
        )
        
        indices = []
        with torch.cuda.amp.autocast(), tqdm(total=len(legal_dataset)) as pbar:
            for batch in loader:
                doc_batch = [doc["text"] for doc in batch]
                # doc_ids = [doc["id"] for doc in batch]
                
                # Process documents in parallel
                with mp.pool.ThreadPool() as pool:
                    results = pool.map(self._score_sentences, doc_batch)
                
                for doc_id, doc, scores in zip(doc_ids, doc_batch, results):
                    ranked_indices = np.argsort(scores)[::-1][:top_k].tolist()
                    indices.append({
                        # "doc_id": doc_id,
                        "sent_id": ranked_indices,
                        "scores": [scores[i] for i in ranked_indices]
                    })
                
                pbar.update(len(batch))
                
                # Empty cache periodically
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