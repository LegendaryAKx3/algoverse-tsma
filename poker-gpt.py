
# usage: python poker-gpt.py --data_dir /path/to/phh_texts --save_dir ./ckpt 
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import string
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ────────────────────────────────────────────────────────────────────────────────
# 1.  Character-level tokenizer
# ────────────────────────────────────────────────────────────────────────────────
class CharTokenizer:
    """Simple stateless char-to-int and int-to-char mapper."""

    def __init__(self, extra_chars: str = "") -> None:
        base = (
            string.ascii_letters
            + string.digits
            + string.punctuation
            + " \t\n"  # space, tab, newline
        )
        charset = sorted(set(base + extra_chars))
        self.stoi: Dict[str, int] = {ch: i for i, ch in enumerate(charset)}
        self.itos: List[str] = charset

    def encode(self, text: str) -> List[int]:  # noqa: D401
        return [self.stoi[ch] for ch in text if ch in self.stoi]

    def decode(self, ids: List[int]) -> str:  # noqa: D401
        return "".join(self.itos[i] for i in ids)

    @property
    def vocab_size(self) -> int:  # noqa: D401
        return len(self.itos)


_ACTION_RE = re.compile(r"^([^:]+): (folds|checks|calls|bets|raises).*", re.I)
_STACK_RE = re.compile(r"Seat \d+: .*?\((\$?[0-9,.]+) in chips\)")


def extract_relevant_lines(filepath: Path) -> str:
    """Return cleaned text from one PHH hand-history file."""
    lines: List[str] = []
    with filepath.open() as fh:
        for raw in fh:
            line = raw.strip()
            if _ACTION_RE.match(line):
                lines.append(line)
            elif _STACK_RE.match(line):
                lines.append(line)
    # Mark hand boundary
    return "\n".join(lines) + "\n<END_HAND>\n"


class PokerPHHDataset(Dataset):
    def __init__(self, dir_path: str | Path, tokenizer: CharTokenizer, block_size: int = 256):
        self.block_size = block_size
        self.tokenizer = tokenizer
        dir_path = Path(dir_path)
        assert dir_path.is_dir(), f"{dir_path} is not a directory"
        print("Collecting PHH files …")
        texts = []
        for fp in dir_path.glob("**/*"):
            if fp.suffix.lower() in {".phh", ".txt"}:
                texts.append(extract_relevant_lines(fp))
        random.shuffle(texts)
        full_text = "\n".join(texts)
        self.data = torch.tensor(tokenizer.encode(full_text), dtype=torch.long)
        print(f"Total tokens: {len(self.data):,}")

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        return chunk[:-1], chunk[1:]  # (input, target)


class GPTConfig:
    def __init__(self, vocab_size: int, n_layer=4, n_head=4, n_embd=256, block_size=256):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.block_size = block_size

# lets see if this works, else we can scale up the architecture too 

class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()

        print("vocab size: cfg.vocab_size", cfg.vocab_size)
        print("n_embd: cfg.n_embd", cfg.n_embd)
        print("block_size: cfg.block_size", cfg.block_size)
        print("n_layer: cfg.n_layer", cfg.n_layer)
        print("n_head: cfg.n_head", cfg.n_head)
        print("n_layer: cfg.n_layer", cfg.n_layer)
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(0.1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.n_embd,
            nhead=cfg.n_head,
            dim_feedforward=4 * cfg.n_embd,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layer)
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.block_size = cfg.block_size
        self.cfg = cfg

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, "Sequence too long"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        # build causal mask: (T, T) with True at positions that should be masked
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), 1)
        x = self.transformer(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            if idx.size(1) > self.block_size:
                idx = idx[:, -self.block_size :]
            logits = self(idx)
            next_token = torch.distributions.Categorical(logits[:, -1, :].softmax(-1)).sample()
            idx = torch.cat([idx, next_token.unsqueeze(-1)], dim=1)
        return idx

def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    tok = CharTokenizer()
    ds = PokerPHHDataset(args.data_dir, tok, block_size=args.block_size)
    cfg = GPTConfig(
        vocab_size=tok.vocab_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        block_size=args.block_size,
    )
    model = GPT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for step, (x, y) in enumerate(dl):
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.view(-1, cfg.vocab_size), y.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if (step + 1) % 100 == 0:
                avg = total_loss / 100
                print(f"Epoch {epoch} | Step {step+1}/{len(dl)} | loss {avg:.3f}")
                total_loss = 0.0
        # save checkpoint each epoch
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save({"model": model.state_dict(), "cfg": cfg.__dict__, "vocab": tok.itos}, Path(args.save_dir)/f"ckpt_{epoch}.pt")

    print("Training complete.  Final model saved in", args.save_dir)



def get_args():
    p = argparse.ArgumentParser(description="Pre-train a GPT on PHH poker hands")
    p.add_argument("--data_dir", required=True, help="Directory with PHH txt files")
    p.add_argument("--save_dir", default="./ckpt", help="Where to save checkpoints")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_embd", type=int, default=256)
    return p.parse_args()


if __name__ == "__main__":
    args = get_args()
    train(args)
