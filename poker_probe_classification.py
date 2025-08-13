# poker_probe_classification.py
"""Train a 2‑layer MLP probe on frozen Poker‑GPT features to predict hand classifications.

This is an adaptation of the MLP probe script for predicting specific hand types like
pair, two-pair, three-of-a-kind, etc. instead of win/loss classification.
The probe's function is p_θ(x) = softmax(W₂ ReLU(W₁ x)).
"""

from __future__ import annotations
import json, argparse, random
from pathlib import Path
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
import numpy as np

# Import classes from poker_gpt to avoid code duplication
from poker_gpt import CharTokenizer as BaseCharTokenizer, GPT, GPTConfig

# -----------------------------------------------------------------------------
# 1.  Tokeniser wrapper (unchanged)
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# 2.  Base model wrapper (unchanged)
# -----------------------------------------------------------------------------
class GPTFeatureExtractor(nn.Module):
    """Wrapper around GPT that exposes hidden states but removes LM head."""

    def __init__(self, gpt_model: GPT):
        super().__init__()
        self.gpt = gpt_model

    def forward(self, idx: torch.Tensor) -> torch.Tensor:  # (B,T) -> (B,T,d)
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.gpt.tok_emb(idx) + self.gpt.pos_emb(pos)
        x = self.gpt.drop(x)
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)
        x = self.gpt.transformer(x, mask)
        x = self.gpt.ln_f(x)
        return x  # hidden states (no LM head)

# -----------------------------------------------------------------------------
# 3.  Dataset for hand classification (modified)
# -----------------------------------------------------------------------------
class PokerClassificationDS(Dataset):
    """Dataset for probing GPT representations on hand type classification."""

    def __init__(self, json_path: Path, tok: CharTokenizer, block_size: int, hand_type: str):
        with Path(json_path).open() as fh:
            raw = json.load(fh)

        self.hand_type = hand_type
        self.samples = []
        
        # Available hand types in the dataset
        valid_hand_types = [
            "high-card", "pair", "two-pair", "three-of-a-kind", 
            "straight", "flush", "full-house", "four-of-a-kind", "straight-flush"
        ]
        
        if hand_type not in valid_hand_types:
            raise ValueError(f"Invalid hand_type '{hand_type}'. Must be one of: {valid_hand_types}")

        for hand in raw:
            # Use the board field (actions) as input
            text = "\n".join(hand["board"]) + "\n<END_HAND>\n"
            ids = tok.encode(text)[:block_size]
            if len(ids) < 1:
                continue
            
            # Extract the specific hand type classification as label
            label = float(hand["type-of-hand"][hand_type])  # Convert bool to float
            self.samples.append((torch.tensor(ids, dtype=torch.long), label))

        self.block_size = block_size
        print(f"Loaded {len(self.samples)} samples for hand type: {hand_type}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ids, label = self.samples[i]
        if len(ids) < self.block_size:  # right‑pad
            pad = torch.zeros(self.block_size - len(ids), dtype=torch.long)
            ids = torch.cat([ids, pad])
        return ids, torch.tensor(label, dtype=torch.float32)

# -----------------------------------------------------------------------------
# 4.  2‑Layer MLP Probe (unchanged)
# -----------------------------------------------------------------------------
class MLPProbe(nn.Module):
    """Non‑linear probe: Linear → ReLU → Linear mapping to logits."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,d) → (B,out_dim)
        return self.net(x)

# -----------------------------------------------------------------------------
# 5.  Training routine (modified for hand classification)
# -----------------------------------------------------------------------------

def train_classification_probe(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- Load frozen GPT -----------------------------------------------------
    ckpt = torch.load(args.gpt_ckpt, map_location="cpu")
    tok = CharTokenizer(ckpt["vocab"])
    gpt_cfg = ckpt["cfg"]
    block_sz = gpt_cfg["block_size"]

    gpt_config = GPTConfig(
        vocab_size=gpt_cfg["vocab_size"],
        n_layer=gpt_cfg["n_layer"],
        n_head=gpt_cfg["n_head"],
        n_embd=gpt_cfg["n_embd"],
        block_size=gpt_cfg["block_size"],
    )
    gpt_model = GPT(gpt_config).to(device)
    gpt_model.load_state_dict(ckpt["model"], strict=False)
    base_model = GPTFeatureExtractor(gpt_model).eval().to(device)
    for p in base_model.parameters():
        p.requires_grad = False  # freeze LM

    # ---- Initialise classification probe ------------------------------------
    probe = MLPProbe(in_dim=gpt_cfg["n_embd"], hidden_dim=args.hidden_dim, out_dim=2).to(device)

    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    # ---- Dataset split ------------------------------------------------------
    ds = PokerClassificationDS(args.dataset_path, tok, block_sz, args.hand_type)
    labels = np.array([ds[i][1].item() for i in range(len(ds))])
    idx = np.arange(len(ds))
    
    # Split by positive/negative examples of the hand type
    pos_idx, neg_idx = idx[labels == 1.0], idx[labels == 0.0]
    
    print(f"Hand type '{args.hand_type}': {len(pos_idx)} positive, {len(neg_idx)} negative examples")

    val_size = max(1, int(0.1 * len(ds)))
    np.random.seed(42)
    n_val_pos = min(val_size // 2, len(pos_idx))
    n_val_neg = min(val_size // 2, len(neg_idx))
    
    # Handle case where we don't have enough positive or negative examples
    if len(pos_idx) > 0 and len(neg_idx) > 0:
        val_idx = np.concatenate([
            np.random.choice(pos_idx, n_val_pos, replace=False),
            np.random.choice(neg_idx, n_val_neg, replace=False),
        ])
    elif len(pos_idx) > 0:
        val_idx = np.random.choice(pos_idx, min(val_size, len(pos_idx)), replace=False)
    elif len(neg_idx) > 0:
        val_idx = np.random.choice(neg_idx, min(val_size, len(neg_idx)), replace=False)
    else:
        raise ValueError(f"No examples found for hand type '{args.hand_type}'")
    
    train_idx = np.setdiff1d(idx, val_idx)

    train_dl = DataLoader(Subset(ds, train_idx), batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(Subset(ds, val_idx), batch_size=args.batch, shuffle=False, drop_last=False)

    # ---- Training loop ------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        probe.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            feats = base_model(x).mean(1)  # (B,d)
            logits = probe(feats)
            loss = crit(logits, y.long())

            opt.zero_grad(); loss.backward(); opt.step()
            running += loss.item()
        print(f"Epoch {epoch} · train loss {running/len(train_dl):.4f}")

        # Validation
        probe.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = probe(base_model(x).mean(1))
                preds = logits.argmax(dim=1)
                correct += (preds == y.long()).sum().item()
                total += y.size(0)
        print(f"Epoch {epoch} · val acc {correct/total:.3%}")

    # ---- Save ----------------------------------------------------------------
    if args.save_probe:
        torch.save({
            "probe": probe.state_dict(), 
            "cfg": gpt_cfg, 
            "hidden_dim": args.hidden_dim,
            "hand_type": args.hand_type
        }, args.save_probe)
        print(f"Hand classification probe for '{args.hand_type}' saved to", args.save_probe)

# -----------------------------------------------------------------------------
# 6.  CLI
# -----------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser("Hand classification probe on Poker‑GPT features")
    p.add_argument("--dataset_path", type=str, required=True, help="Path to monte-carlo_dataset.json")
    p.add_argument("--gpt_ckpt", type=str, required=True, help="Path to pretrained GPT checkpoint (.pt)")
    p.add_argument("--hand_type", type=str, required=True, 
                   choices=["high-card", "pair", "two-pair", "three-of-a-kind", 
                           "straight", "flush", "full-house", "four-of-a-kind", "straight-flush"],
                   help="Which hand type to classify")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=512, help="Hidden layer size H for the probe")
    p.add_argument("--save_probe", type=str, default="", help="Destination to save trained probe (optional)")
    return p.parse_args()

# -----------------------------------------------------------------------------
# 7.  Entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_classification_probe(get_args())
