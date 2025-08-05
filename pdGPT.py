import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from pathlib import Path
import matplotlib.pyplot as plt

# --- Tokenizer: simple char-level ---
class PDTokenizer:
    def __init__(self, extra_chars=" "):
        base = "DC0123456789 ,"
        charset = sorted(set(base + extra_chars))
        self.stoi = {c: i for i, c in enumerate(charset)}
        self.itos = charset
        self.pad_token = "<PAD>"
        self.stoi[self.pad_token] = len(self.stoi)
        self.itos.append(self.pad_token)
    
    def encode(self, text):
        return [self.stoi.get(c, self.stoi[self.pad_token]) for c in text]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids if self.itos[i] != self.pad_token)

    @property
    def vocab_size(self):
        return len(self.itos)

# --- Dataset ---
class PDDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=64):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        inp = f"{row['Number of Rounds']},{row['History']},{row['Co-op Payoff']},{row['Defect Positive Payoff']},{row['Defect Negative Payoff']}"
        tokens = self.tokenizer.encode(inp)
        tokens = tokens[:self.max_len] + [self.tokenizer.stoi[self.tokenizer.pad_token]]*(self.max_len - len(tokens))
        label = 0 if row['Model Action'] == "C" else 1
        return torch.tensor(tokens, dtype=torch.long), torch.tensor(label, dtype=torch.long)

# --- Model config ---
class PDConfig:
    def __init__(self, vocab_size, n_layer=4, n_head=4, n_embd=128, max_len=64):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_embd = n_embd
        self.max_len = max_len

# --- Model ---
class PDTransformerModel(nn.Module):
    def __init__(self, cfg: PDConfig):
        super().__init__()
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.n_embd)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.n_embd,
            nhead=cfg.n_head,
            dim_feedforward=4*cfg.n_embd,
            activation='gelu',
            batch_first=True,
            dropout=0.1,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layer)
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.classifier = nn.Linear(cfg.n_embd, 2)
        self.max_len = cfg.max_len
    
    def forward(self, x):
        B, T = x.size()
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = self.token_emb(x) + self.pos_emb(pos)
        x = self.transformer(x)
        x = self.ln_f(x)
        # Pooling: mean over tokens
        x = x.mean(dim=1)
        logits = self.classifier(x)
        return logits

# --- Training & evaluation ---

def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# --- Main training function ---
def train_pd_model(csv_path, epochs=20, batch_size=32, max_len=64, lr=1e-4):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(csv_path)

    train_losses = []
    val_accuracies = []
    
    tokenizer = PDTokenizer()
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['Model Action'])
    train_ds = PDDataset(train_df, tokenizer, max_len)
    val_ds = PDDataset(val_df, tokenizer, max_len)
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    
    cfg = PDConfig(tokenizer.vocab_size, n_layer=4, n_head=4, n_embd=128, max_len=max_len)
    model = PDTransformerModel(cfg).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    for epoch in range(1, epochs+1):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_acc = evaluate(model, val_loader, device)

        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        
        print(f"Epoch {epoch} | Train loss: {train_loss:.4f} | Val accuracy: {val_acc:.4f}")
    
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'tokenizer': tokenizer,
        'config': cfg.__dict__,
    }, "pd_transformer_checkpoint.pth")

    print("Training complete and checkpoint saved.")
    plot_training_curves(train_losses, val_accuracies)
    plot_confusion_matrix(model, val_loader, device)

def plot_training_curves(train_losses, val_accuracies):
    epochs = range(1, len(train_losses) + 1)

    fig, ax1 = plt.subplots()

    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="tab:red")
    ax1.plot(epochs, train_losses, label="Train Loss", color="tab:red")
    ax1.tick_params(axis='y', labelcolor="tab:red")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Validation Accuracy", color="tab:blue")
    ax2.plot(epochs, val_accuracies, label="Val Accuracy", color="tab:blue")
    ax2.tick_params(axis='y', labelcolor="tab:blue")

    fig.tight_layout()
    plt.title("Training Loss & Validation Accuracy")
    plt.show()
def plot_confusion_matrix(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["C", "D"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

# --- Example inference ---
def predict(model, tokenizer, input_str, max_len=64, device="cpu"):
    model.eval()
    tokens = tokenizer.encode(input_str)
    tokens = tokens[:max_len] + [tokenizer.stoi[tokenizer.pad_token]]*(max_len - len(tokens))
    x = torch.tensor([tokens], dtype=torch.long).to(device)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=1).item()
    return "C" if pred == 0 else "D"

if __name__ == "__main__":
    train_pd_model("game_data.csv", epochs=10)
    

