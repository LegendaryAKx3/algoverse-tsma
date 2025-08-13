from __future__ import annotations
import json, argparse, math, random
from pathlib import Path
from typing import List, Dict

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import numpy as np

# Import classes from poker_gpt to avoid code duplication
from poker_gpt import CharTokenizer as BaseCharTokenizer, GPT, GPTConfig

class CharTokenizer:
    """Adapter for poker_gpt CharTokenizer to match checkpoint format."""
    
    def __init__(self, vocab: List[str]):
        self.itos = vocab
        self.stoi = {ch: i for i, ch in enumerate(vocab)}
        
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs, filtering unknown characters."""
        return [self.stoi[ch] for ch in text if ch in self.stoi]
    
    @property
    def vocab_size(self):
        return len(self.itos)

class GPTFeatureExtractor(nn.Module):
    """Wrapper around GPT for feature extraction (removes LM head)."""
    
    def __init__(self, gpt_model):
        super().__init__()
        self.gpt = gpt_model
        
    def forward(self, idx):
        """Extract features from token sequences (no LM head)."""
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.gpt.tok_emb(idx) + self.gpt.pos_emb(pos)
        x = self.gpt.drop(x)
        
        # Causal mask for autoregressive attention
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)
        x = self.gpt.transformer(x, mask)
        x = self.gpt.ln_f(x)
        return x  # return hidden states without LM head

class PokerProbeDS(Dataset):
    """Dataset for probing GPT representations on poker win prediction."""
    
    def __init__(self, json_path: Path, tok: CharTokenizer, block_size: int):
        with Path(json_path).open() as fh:
            raw = json.load(fh)
            
        self.samples = []
        for hand in raw:
            # Convert hand log to token sequence
            text  = "\n".join(hand["log"]) + "\n<END_HAND>\n"
            ids   = tok.encode(text)[:block_size]
            if len(ids) < 1:
                continue
            self.samples.append((torch.tensor(ids), float(hand["win"])))

        self.block_size = block_size

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, i):
        ids, label = self.samples[i]
        # Right-pad sequences to block_size with zeros
        if len(ids) < self.block_size:
            pad = torch.full((self.block_size - len(ids),), fill_value=0, dtype=torch.long)
            ids = torch.cat([ids, pad])
        return ids, torch.tensor(label, dtype=torch.float32)

def train_probe(args):
    """Train a linear probe on frozen GPT features for win prediction."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained GPT checkpoint and setup
    ckpt      = torch.load(args.gpt_ckpt, map_location="cpu")
    tok       = CharTokenizer(ckpt["vocab"])
    gpt_cfg   = ckpt["cfg"]
    block_sz  = gpt_cfg["block_size"]

    # Initialize and freeze the base model
    gpt_config = GPTConfig(
        vocab_size=gpt_cfg["vocab_size"],
        n_layer=gpt_cfg["n_layer"], 
        n_head=gpt_cfg["n_head"],
        n_embd=gpt_cfg["n_embd"],
        block_size=gpt_cfg["block_size"]
    )
    gpt_model = GPT(gpt_config).to(device)
    gpt_model.load_state_dict(ckpt["model"], strict=False)
    base_model = GPTFeatureExtractor(gpt_model).to(device)
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False   # freeze GPT weights

    # Setup probe: linear classifier with 2 output logits (True/False)
    probe = nn.Linear(gpt_cfg["n_embd"], 2).to(device)
    crit  = nn.CrossEntropyLoss()
    opt   = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    # Prepare stratified train/validation splits for balanced validation
    ds = PokerProbeDS(args.dataset_path, tok, block_sz)
    
    # Get all labels for stratified splitting
    all_labels = [ds[i][1].item() for i in range(len(ds))]
    indices = np.arange(len(ds))
    
    # Separate indices by class
    win_indices = indices[np.array(all_labels) == 1.0]
    loss_indices = indices[np.array(all_labels) == 0.0]
    
    print(f"Total samples: {len(ds)}")
    print(f"Win samples: {len(win_indices)}, Loss samples: {len(loss_indices)}")
    
    # Calculate validation size to maintain balance
    val_sz = max(1, int(0.1 * len(ds)))
    val_win_sz = min(val_sz // 2, len(win_indices))
    val_loss_sz = min(val_sz // 2, len(loss_indices))
    
    # Randomly sample from each class for validation
    np.random.seed(42)  # For reproducibility
    val_win_idx = np.random.choice(win_indices, val_win_sz, replace=False)
    val_loss_idx = np.random.choice(loss_indices, val_loss_sz, replace=False)
    val_idx = np.concatenate([val_win_idx, val_loss_idx])
    
    # Remaining indices for training
    train_idx = np.setdiff1d(indices, val_idx)
    
    # Create datasets
    train_ds = Subset(ds, train_idx)
    val_ds = Subset(ds, val_idx)
    
    # Verify balance in validation set
    val_labels = [ds[i][1].item() for i in val_idx]
    val_win_count = sum(val_labels)
    val_loss_count = len(val_labels) - val_win_count
    # print(f"Validation set: {val_win_count} wins, {val_loss_count} losses")
    
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch, shuffle=False, drop_last=False)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        probe.train()
        total = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            h = base_model(x)               # extract features (B,T,d)
            feats = h.mean(1)               # mean-pool over sequence
            logits = probe(feats)           # (B, 2) - logits for [False, True]
            y_long = y.long()               # convert to long for CrossEntropyLoss
            loss   = crit(logits, y_long)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"Epoch {epoch} - train loss {total/len(train_dl):.4f}")

        # Validation accuracy
        probe.eval()
        correct = seen = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                feats  = base_model(x).mean(1)
                logits = probe(feats)                    # (B, 2) logits
                preds  = torch.argmax(logits, dim=1)     # get predicted class (0 or 1)
                correct += (preds == y.long()).sum().item()
                seen    += y.size(0)
        acc = correct / seen
        print(f"Epoch {epoch} - val acc {acc:.3%}")

    # Save trained probe if requested
    if args.save_probe:
        torch.save({"probe": probe.state_dict(),
                    "cfg":   gpt_cfg}, args.save_probe)
        print("Probe saved to", args.save_probe)

def get_args():
    """Parse command-line arguments for probe training."""
    p = argparse.ArgumentParser("Linear probe on poker GPT features")
    p.add_argument("--dataset_path", type=str, required=True,
                   help="Path to poker_dataset.json with logs + win labels")
    p.add_argument("--gpt_ckpt",     type=str, required=True,
                   help="Path to pretrained GPT checkpoint (.pt file)")
    p.add_argument("--epochs",       type=int, default=5,
                   help="Number of training epochs")
    p.add_argument("--batch",        type=int, default=64,
                   help="Batch size for training")
    p.add_argument("--save_probe",   type=str, default="",
                   help="Path to save trained probe (optional)")
    return p.parse_args()

if __name__ == "__main__":
    args = get_args()
    train_probe(args)
