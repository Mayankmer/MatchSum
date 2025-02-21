import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AdamW
from tqdm import tqdm
from dataloader import LegalDataset
from model import LegalMatchSum
from metrics import LegalMetrics


def train():
    # Initialize with score file path
    dataset = LegalDataset(
        data_path="percins/IN-ABS",
        scores_path="./legal_scores.jsonl",
        tokenizer=AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
    dataset = LegalDataset(tokenizer)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    model = LegalMatchSum().to(device)
    optimizer = AdamW(model.parameters(), lr=3e-5)
    
    model.train()
    for epoch in range(10):
        for batch in tqdm(dataloader):
            # Validate batch
            if batch['labels'].dim() == 0:
                print("Skipping invalid batch")
                continue
                
            # Move to GPU
            inputs = {
                'input_ids': batch['input_ids'].to(device),
                'attention_mask': batch['attention_mask'].to(device),
                'labels': batch['labels'].to(device)
            }
            
            # Forward pass
            optimizer.zero_grad()
            scores = model(inputs['input_ids'], inputs['attention_mask'])
            
            # Check outputs
            if scores is None:
                raise RuntimeError("Model returned None scores")
                
            loss = torch.nn.functional.mse_loss(
                scores, 
                inputs['labels'].float()
            )
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
        print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

if __name__ == '__main__':
    train()