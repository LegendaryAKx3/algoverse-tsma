"""infer_poker_gpt.py
Run inference with a Poker-GPT checkpoint.

Example:
    python infer_poker_gpt.py \
        --ckpt ckpt/ckpt_3.pt \
        --prompt "BLINDS [5, 10]\nSTACKS [1000, 1000]\nHero: raises 20 to 30\nVillain: calls 30\n*** FLOP *** [Ah Kd 5s]\nHero:" \
        --tokens 120
"""
from __future__ import annotations

import argparse
from pathlib import Path
import torch

from poker_gpt import CharTokenizer, GPT, GPTConfig

def load_model(ckpt_path: Path, device: str = "cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)

    # rebuild tokenizer
    tok = CharTokenizer()
    tok.itos = ckpt["vocab"]
    tok.stoi = {ch: i for i, ch in enumerate(tok.itos)}

    # rebuild model
    cfg = GPTConfig(**ckpt["cfg"])
    model = GPT(cfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    return model, tok

def generate(model: GPT, tok: CharTokenizer, prompt: str, max_tokens: int, device: str):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(ids, max_tokens)
    generated = tok.decode(out[0].tolist())
    return generated

def main():
    ap = argparse.ArgumentParser(description="Generate poker actions with a trained Poker-GPT model")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint .pt file")
    default_prompt = (
        "BLINDS [5, 10]\n"
        "STACKS [10000, 10000, 10000, 10000, 10000, 10000]\n"
        "d dh p1 8h7c\n"
        "d dh p2 4s2s\n"
        "d dh p3 QhKc\n"
        "d dh p4 3dJs\n"
        "d dh p5 2d3h\n"
        "d dh p6 7sJh\n"
        "p3 cbr 200\n"
        "p4 f\n"
        "p5 f\n"
        "p6 "
    )

    ap.add_argument(
        "--prompt",
        default=default_prompt,
        help="PHH prompt to continue (default is a small heads-up hand). Use literal \n for newlines.",
    )
    ap.add_argument("--tokens", type=int, default=100, help="Number of tokens to generate")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    model, tok = load_model(Path(args.ckpt), args.device)
    completion = generate(model, tok, args.prompt, args.tokens, args.device)

    # print only the continuation
    print(completion[len(args.prompt):])

if __name__ == "__main__":
    main() 