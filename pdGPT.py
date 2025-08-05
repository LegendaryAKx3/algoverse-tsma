import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd

# --- Hyperparameters ---
MAX_LEN = 64
EMBED_DIM = 64
NUM_HEADS = 2
NUM_LAYERS = 2
BATCH_SIZE = 16
EPOCHS = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Step 1: Vocabulary ---
CHARS = list("DC0123456789 ,")
char2idx = {c: i for i, c in enumerate(CHARS)}
char2idx["PAD"] = len(char2idx)
idx2char = {i: c for c, i in char2idx.items()}
vocab_size = len(char2idx)

def tokenize(text):
    return [char2idx[c] for c in text if c in char2idx]

def pad(tokens, max_len=MAX_LEN):
    return tokens + [char2idx["PAD"]] * (max_len - len(tokens))

# --- Step 2: Dataset ---
class DilemmaDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        input_str = f"{row['Number of Rounds']},{row['History']},{row['Co-op Payoff']},{row['Defect Positive Payoff']},{row['Defect Negative Payoff']}"
        tokens = pad(tokenize(input_str))
        label = 0 if row["Model Action"] == "C" else 1
        return torch.tensor(tokens), torch.tensor(label)

# --- Step 3: Model ---
class DilemmaModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, EMBED_DIM)
        encoder_layer = nn.TransformerEncoderLayer(d_model=EMBED_DIM, nhead=NUM_HEADS)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=NUM_LAYERS)
        self.classifier = nn.Linear(EMBED_DIM, 2)

    def forward(self, x):
        x = self.embed(x)  # (batch, seq, embed)
        x = x.permute(1, 0, 2)  # Transformer expects (seq, batch, embed)
        x = self.transformer(x)
        x = x[0]  # Use the first token's representation (like CLS)
        return self.classifier(x)

# --- Step 4: Train Loop ---
def train(model, loader, optimizer, loss_fn):
    model.train()
    for batch in loader:
        x, y = [b.to(DEVICE) for b in batch]
        optimizer.zero_grad()
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()

# --- Step 5: Evaluation ---
def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / total

# --- Main ---
dataset = DilemmaDataset("game_data.csv")
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = DilemmaModel().to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
import pandas as pd
df = pd.read_csv("game_data.csv")
print(df["Model Action"].value_counts())
for epoch in range(EPOCHS):
    train(model, train_loader, optimizer, loss_fn)
    acc = evaluate(model, train_loader)
    print(f"Epoch {epoch+1}: Accuracy = {acc:.2f}")


