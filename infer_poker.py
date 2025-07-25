# infer.py ─ run with:  python infer.py ckpt/ckpt_3.pt
import sys, torch
from pathlib import Path
from poker_gpt import CharTokenizer, GPT, GPTConfig          # reuse the classes

ckpt_path = Path(sys.argv[1])           # e.g. ckpt/ckpt_3.pt
ckpt       = torch.load(ckpt_path, map_location="cpu")

# 1. rebuild tokenizer & model --------------------------------------------------
tok = CharTokenizer()                   # initialise default vocab
tok.itos = ckpt["vocab"]                # restore training vocab
tok.stoi = {ch: i for i, ch in enumerate(tok.itos)}

cfg  = GPTConfig(**ckpt["cfg"])
model = GPT(cfg)
model.load_state_dict(ckpt["model"], strict=False)
model.eval()

# 2. the partial hand-history prompt you want to continue -----------------------
prompt = """\
Seat 1: Hero ($100 in chips)
Seat 2: Villain ($100 in chips)
Hero: raises 2 to 3
Villain: calls 3
*** FLOP *** [Ah Kd 5s]
Hero:"""          # <- we want the model to suggest Hero’s next action

# 3. encode → generate ----------------------------------------------------------
ids      = torch.tensor([tok.encode(prompt)], dtype=torch.long)
with torch.no_grad():
    out = model.generate(ids, max_new_tokens=80)

generated = tok.decode(out[0].tolist())
print(generated[len(prompt):])          # only show completion