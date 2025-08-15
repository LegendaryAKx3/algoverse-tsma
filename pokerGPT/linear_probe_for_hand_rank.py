#!/usr/bin/env python
import os
import json
import numpy as np
import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter

# ---------------------------
# Poker hand evaluation
# ---------------------------
RANK_ORDER = "23456789TJQKA"

def card_rank(card):
    """Return the rank character of a card."""
    return card[0]

def card_suit(card):
    """Return the suit character of a card."""
    return card[1]

def split_cards(s):
    """Split concatenated cards into pairs, e.g., '2hAc' -> ['2h', 'Ac']"""
    return [s[i:i+2] for i in range(0, len(s), 2)]

def hand_rank(cards):
    """
    Compute simplified poker hand rank for 5-7 cards.
    Returns one of:
    'high_card', 'pair', 'two_pair', 'three_of_a_kind', 'straight',
    'flush', 'full_house', 'four_of_a_kind', 'straight_flush'
    """
    if len(cards) < 5:
        return "high_card"

    ranks = sorted([RANK_ORDER.index(card_rank(c)) for c in cards], reverse=True)
    suits = [card_suit(c) for c in cards]
    rank_counts = Counter(ranks)
    suit_counts = Counter(suits)

    # Straight detection
    is_straight = False
    unique_ranks = sorted(set(ranks))
    for i in range(len(unique_ranks) - 4):
        if unique_ranks[i+4] - unique_ranks[i] == 4:
            is_straight = True
            break
    # Special case A-2-3-4-5
    if set([12,0,1,2,3]).issubset(set(ranks)):
        is_straight = True

    # Flush
    is_flush = any(count >= 5 for count in suit_counts.values())

    # Straight flush
    if is_straight and is_flush:
        return "straight_flush"
    # Four of a kind
    if 4 in rank_counts.values():
        return "four_of_a_kind"
    # Full house
    if 3 in rank_counts.values() and 2 in rank_counts.values():
        return "full_house"
    # Flush
    if is_flush:
        return "flush"
    # Straight
    if is_straight:
        return "straight"
    # Three of a kind
    if 3 in rank_counts.values():
        return "three_of_a_kind"
    # Two pair
    if list(rank_counts.values()).count(2) >= 2:
        return "two_pair"
    # One pair
    if 2 in rank_counts.values():
        return "pair"
    return "high_card"

# ---------------------------
# Dataset handling
# ---------------------------
def get_player1_hands_and_board(data_path, max_samples=None):
    X_text, y = [], []
    for name in os.listdir(data_path):
        if not name.endswith(".ndjson"):
            continue
        with open(os.path.join(data_path, name), "r") as f:
            for line in f:
                try:
                    arr = json.loads(line)
                except:
                    continue

                player1_cards = None
                board_cards = []
                for item in arr:
                    parts = item.strip().split()
                    if item.startswith("d dh p1") and len(parts) >= 4:
                        player1_cards = split_cards(parts[3])
                    if item.startswith("d db") and len(parts) >= 4:
                        for token in parts[3:]:
                            board_cards.extend(split_cards(token))

                if player1_cards and board_cards:
                    all_cards = player1_cards + board_cards
                    label = hand_rank(all_cards)
                    X_text.append(" ".join(all_cards))
                    y.append(label)

                if max_samples and len(X_text) >= max_samples:
                    return X_text, np.array(y)
    return X_text, np.array(y)

# ---------------------------
# Main probing script
# ---------------------------
def main():
    # Load tokenizer + model
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer.from_file("artifacts/tokenizer/tokenizer.json")
    )
    tokenizer.add_special_tokens({
        "pad_token": "<PAD>",
        "unk_token": "<UNK>",
        "bos_token": "<BOS>",
        "eos_token": "<EOS>"
    })

    model = AutoModel.from_pretrained("artifacts/checkpoints/run1/best")
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Load dataset
    X_text, y = get_player1_hands_and_board("data", max_samples=100)
    print(f"Loaded {len(X_text)} samples for probing.")
    if len(X_text) == 0:
        raise ValueError("No player1 hands found in dataset!")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_text, y, test_size=0.2, random_state=42, stratify=y
    )

    # Tokenize
    tokens_train = tokenizer(X_train, return_tensors="pt", padding=True, truncation=True)
    tokens_test  = tokenizer(X_test,  return_tensors="pt", padding=True, truncation=True)

    # Run model
    with torch.no_grad():
        outputs_train = model(**tokens_train, output_hidden_states=True)
        outputs_test  = model(**tokens_test,  output_hidden_states=True)
        hidden_train = outputs_train.hidden_states
        hidden_test  = outputs_test.hidden_states

    # Probe each layer
    for layer_idx, (layer_train, layer_test) in enumerate(zip(hidden_train, hidden_test)):
        embeddings_train = layer_train.mean(dim=1).cpu().numpy()
        embeddings_test  = layer_test.mean(dim=1).cpu().numpy()

        scaler = StandardScaler()
        embeddings_train = scaler.fit_transform(embeddings_train)
        embeddings_test  = scaler.transform(embeddings_test)

        clf = LogisticRegression(max_iter=10000)
        clf.fit(embeddings_train, y_train)
        preds = clf.predict(embeddings_test)
        acc = accuracy_score(y_test, preds)
        print(f"Layer {layer_idx:02d} accuracy: {acc:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, preds, labels=np.unique(y))
        labels = np.unique(y)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")

        ax.set_title(f"Layer {layer_idx:02d} Confusion Matrix")
        ax.set_ylabel("True")
        ax.set_xlabel("Predicted")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
