import json
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset

class LegalIndexGenerator:
    def __init__(self, model_name="nlpaueb/legal-bert-base-uncased"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def _score_sentences(self, document):
        """Score sentences using legal BERT embeddings"""
        sentence_scores = []
        
        # Process each sentence
        for sent in document:
            inputs = self.tokenizer(
                sent,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # Use CLS token embedding norm as score
            sentence_score = torch.norm(outputs.last_hidden_state[0, 0]).item()
            sentence_scores.append(sentence_score)
        
        return sentence_scores

    def generate_indices(self, dataset_path, output_path, top_k=8):
        """Generate and save sentence indices for legal documents"""
        dataset = load_dataset("percins/IN-ABS", split="train")
        
        indices = []
        for doc in tqdm(dataset):
            document = doc["text"]
            scores = self._score_sentences(document)
            
            # Get top-k sentence indices
            ranked_indices = np.argsort(scores)[::-1][:top_k].tolist()
            
            indices.append({
                # "doc_id": doc["id"],
                "sent_id": ranked_indices,
                "scores": [scores[i] for i in ranked_indices]
            })
        
        # Save indices
        with open(output_path, "w") as f:
            for item in indices:
                f.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    generator = LegalIndexGenerator()
    
    # Generate indices for legal documents
    generator.generate_indices(
        dataset_path="percins/IN-ABS",
        output_path="./legal_sentence_indices.jsonl",
        top_k=10  # For legal docs we need more candidates
    )