# poker_probe_regression.py
"""Train a 2‑layer MLP probe on frozen Poker‑GPT features to predict poker equity.

This is a regression adaptation of the original probe script. The probe's function is
p_θ(x) = W₂ ReLU(W₁ x), predicting continuous equity values instead of win/loss.
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
# 3.  Dataset for regression (modified)
# -----------------------------------------------------------------------------
class PokerRegressionDS(Dataset):
    """Dataset for probing GPT representations on poker equity prediction."""

    def __init__(self, json_path: Path, tok: CharTokenizer, block_size: int):
        with Path(json_path).open() as fh:
            raw = json.load(fh)

        self.samples = []
        for hand in raw:
            # Use the board field (similar to actions in labeled dataset)
            text = "\n".join(hand["board"]) + "\n<END_HAND>\n"
            ids = tok.encode(text)[:block_size]
            if len(ids) < 1:
                continue
            # Use the equity field as regression target
            self.samples.append((torch.tensor(ids, dtype=torch.long), float(hand["equity"])))

        self.block_size = block_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        ids, equity = self.samples[i]
        if len(ids) < self.block_size:  # right‑pad
            pad = torch.zeros(self.block_size - len(ids), dtype=torch.long)
            ids = torch.cat([ids, pad])
        return ids, torch.tensor(equity, dtype=torch.float32)

# -----------------------------------------------------------------------------
# 4.  2‑Layer MLP Regression Probe (modified for regression)
# -----------------------------------------------------------------------------
class MLPRegressionProbe(nn.Module):
    """Non‑linear probe: Linear → ReLU → Linear mapping to single equity output."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=True),  # Single output for regression
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B,d) → (B,1)
        return self.net(x).squeeze(-1)  # Return (B,) for easier loss computation

# -----------------------------------------------------------------------------
# 5.  Training routine (modified for regression)
# -----------------------------------------------------------------------------

def train_regression_probe(args):
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

    # ---- Initialise regression probe ----------------------------------------
    probe = MLPRegressionProbe(in_dim=gpt_cfg["n_embd"], hidden_dim=args.hidden_dim).to(device)

    # Use MSE loss for regression
    crit = nn.MSELoss()
    opt = torch.optim.AdamW(probe.parameters(), lr=1e-3)

    # ---- Dataset split ------------------------------------------------------
    ds = PokerRegressionDS(args.dataset_path, tok, block_sz)
    equities = np.array([ds[i][1].item() for i in range(len(ds))])
    idx = np.arange(len(ds))
    
    # Get equity range for reporting
    equity_min, equity_max = equities.min(), equities.max()
    
    # Simple random split like in poker_probe_mlp.py
    val_size = max(1, int(0.1 * len(ds)))
    np.random.seed(42)
    val_idx = np.random.choice(idx, val_size, replace=False)
    train_idx = np.setdiff1d(idx, val_idx)

    train_dl = DataLoader(Subset(ds, train_idx), batch_size=args.batch, shuffle=True, drop_last=True)
    val_dl = DataLoader(Subset(ds, val_idx), batch_size=args.batch, shuffle=False, drop_last=False)

    print(f"Training samples: {len(train_idx)}, Validation samples: {len(val_idx)}")
    print(f"Equity range: {equity_min:.3f} - {equity_max:.3f}")

    # ---- Training loop ------------------------------------------------------
    for epoch in range(1, args.epochs + 1):
        probe.train()
        running_loss = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            feats = base_model(x).mean(1)  # (B,d)
            pred_scores = probe(feats)  # (B,)
            loss = crit(pred_scores, y)

            opt.zero_grad(); loss.backward(); opt.step()
            running_loss += loss.item()
        
        avg_train_loss = running_loss / len(train_dl)
        print(f"Epoch {epoch} · train MSE loss {avg_train_loss:.4f}")

        # Validation
        probe.eval()
        val_loss = 0.0
        abs_errors = []
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred_scores = probe(base_model(x).mean(1))
                loss = crit(pred_scores, y)
                val_loss += loss.item()
                
                # Track absolute errors for interpretability
                abs_errors.extend((pred_scores - y).abs().cpu().numpy())
        
        avg_val_loss = val_loss / len(val_dl)
        mae = np.mean(abs_errors)
        print(f"Epoch {epoch}  val MSE loss {avg_val_loss:.6f}  MAE {mae:.4f}")

    # ---- Save ----------------------------------------------------------------
    if args.save_probe:
        torch.save({
            "probe": probe.state_dict(), 
            "cfg": gpt_cfg, 
            "hidden_dim": args.hidden_dim,
            "equity_stats": {"min": equity_min, "max": equity_max}
        }, args.save_probe)
        print("Regression probe saved to", args.save_probe)

# -----------------------------------------------------------------------------
# 6.  CLI
# -----------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser("Non‑linear (2‑layer MLP) regression probe on Poker‑GPT features")
    p.add_argument("--dataset_path", type=str, required=True, help="Path to monte-carlo_dataset.json")
    p.add_argument("--gpt_ckpt", type=str, required=True, help="Path to pretrained GPT checkpoint (.pt)")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=512, help="Hidden layer size H for the probe")
    p.add_argument("--save_probe", type=str, default="", help="Destination to save trained probe (optional)")
    return p.parse_args()

# -----------------------------------------------------------------------------
# 7.  Entry‑point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    train_regression_probe(get_args())
