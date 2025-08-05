# pd_inference.py
import torch
from pdGPT import PDTokenizer, PDConfig, PDTransformerModel, predict
import torch.serialization

torch.serialization.add_safe_globals([
    "PDTokenizer",
    "PDConfig",
    "PDTransformerModel"
])

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load("pd_transformer_checkpoint.pth", map_location=device, weights_only=False)

    # Rebuild tokenizer from saved vocab
    vocab = checkpoint['tokenizer'].itos
    tokenizer = PDTokenizer()
    tokenizer.itos = vocab
    tokenizer.stoi = {c: i for i, c in enumerate(vocab)}
    tokenizer.pad_token = "<PAD>"

    cfg = PDConfig(**checkpoint['config'])
    model = PDTransformerModel(cfg).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    while True:
        inp = input("Enter PD input string (or 'exit' to quit): ")
        if inp.lower() == "exit":
            break
        action = predict(model, tokenizer, inp, max_len=cfg.max_len, device=device)
        print(f"Predicted action: {action}")

if __name__ == "__main__":
    main()
