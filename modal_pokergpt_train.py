# modal_train_poker_gpt.py
# Train GPT-from-scratch on poker_dataset.json in Modal, using PyTorch CUDA 12.1 image

import os
import json
from typing import List, Dict, Any, Iterator
import modal

app = modal.App("poker-gpt-scratch")

# Persisted volume with dataset and outputs
vol = modal.Volume.from_name("tsma")

# Build image from PyTorch CUDA base, then pip install deps
image = (
    modal.Image.from_registry("pytorch/pytorch:2.3.1-cuda12.1-cudnn8-devel")
    .pip_install(
        "transformers[torch]>=4.41.0",
        "datasets>=2.19.0",
        "tokenizers>=0.15.2",
        "wandb",
    )
)

# -----------------------------
# Training helper functions
# -----------------------------
def _load_logs(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        data: List[Dict[str, Any]] = json.load(f)
    texts = []
    for item in data:
        lines = item.get("log", [])
        if isinstance(lines, list) and lines:
            s = "\n".join(map(str, lines)).strip()
            if s:
                texts.append(s)
    if not texts:
        raise ValueError("No valid 'log' entries found.")
    return texts

def _iter_corpus(texts: List[str]) -> Iterator[str]:
    for t in texts:
        yield t

def _build_tokenizer(texts, vocab_size, min_frequency):
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel, Digits, Sequence
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from transformers import PreTrainedTokenizerFast

    tok = Tokenizer(BPE(unk_token="<unk>"))
    # Keep byte-level robustness + split numbers into individual digits
    tok.pre_tokenizer = Sequence([ByteLevel(), Digits(individual_digits=True)])
    tok.decoder = ByteLevelDecoder()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["<s>", "</s>", "<pad>", "<unk>"],
        show_progress=True,
    )
    tok.train_from_iterator(_iter_corpus(texts), trainer=trainer)

    return PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )

def _build_dataset(texts, tokenizer, block_size):
    from datasets import Dataset as HFDataset
    eos = tokenizer.eos_token or "</s>"
    combined = f"{eos}\n".join(t.strip() for t in texts if t.strip()) + eos
    raw = HFDataset.from_dict({"text": [combined]})

    def tok_fn(ex):
        return tokenizer(ex["text"])
    tokd = raw.map(tok_fn, batched=True, remove_columns=["text"], desc="Tokenizing")

    def group_fn(ex):
        concatenated = {k: sum(ex[k], []) for k in ex.keys()}
        total = len(concatenated["input_ids"])
        total = (total // block_size) * block_size
        result = {
            k: [t[i:i+block_size] for i in range(0, total, block_size)]
            for k, t in concatenated.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    return tokd.map(group_fn, batched=True, desc=f"Packing {block_size}-token blocks")

def _train(args):
    import torch
    from transformers import (
        GPT2Config, GPT2LMHeadModel,
        DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
    )
    import wandb
    import os

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    data_path = os.path.join(args.data_root, "poker_dataset.json")
    texts = _load_logs(data_path)

    tokenizer = _build_tokenizer(texts, args.vocab_size, args.min_frequency)
    tokenizer.save_pretrained(os.path.join(args.output_dir, "tokenizer"))

    dataset = _build_dataset(texts, tokenizer, args.block_size)

    config = GPT2Config(
        vocab_size=tokenizer.vocab_size,
        n_positions=args.block_size,
        n_ctx=args.block_size,
        n_embd=args.n_embd,
        n_layer=args.n_layer,
        n_head=args.n_head,
        resid_pdrop=0.1, embd_pdrop=0.1, attn_pdrop=0.1,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = GPT2LMHeadModel(config)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    train_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        fp16=args.fp16, bf16=args.bf16,
        dataloader_num_workers=2,
        report_to="wandb",
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    prompt = "d dh p1 "
    model.eval()
    with torch.no_grad():
        out = model.generate(
            **tokenizer(prompt, return_tensors="pt").to(model.device),
            max_new_tokens=50, do_sample=True, top_p=0.95, temperature=0.8,
            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
        )
    print("\n=== Sample ===")
    print(tokenizer.decode(out[0], skip_special_tokens=True))

# -----------------------------
# Modal function
# -----------------------------
@app.function(
    image=image,
    gpu="L4",     # Change to desired GPU type
    timeout=60 * 60 * 6,
    volumes={"/data": vol},
    cpu=4, memory=24 * 1024,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_on_modal(
    num_train_epochs: float = 500,
    n_layer: int = 6,
    n_head: int = 6,
    n_embd: int = 384,
    block_size: int = 1024,
    vocab_size: int = 2048,
    min_frequency: int = 2,
    per_device_train_batch_size: int = 16,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.03,
    seed: int = 42,
    fp16: bool = True,
    bf16: bool = False,
    output_subdir: str = "runs/run1",
):
    class Args: pass
    args = Args()
    args.data_root = "/data"
    args.output_dir = f"/data/{output_subdir}"
    args.vocab_size = vocab_size
    args.min_frequency = min_frequency
    args.block_size = block_size
    args.n_layer = n_layer
    args.n_head = n_head
    args.n_embd = n_embd
    args.per_device_train_batch_size = per_device_train_batch_size
    args.gradient_accumulation_steps = gradient_accumulation_steps
    args.num_train_epochs = num_train_epochs
    args.learning_rate = learning_rate
    args.weight_decay = weight_decay
    args.warmup_ratio = warmup_ratio
    args.seed = seed
    args.fp16 = fp16
    args.bf16 = bf16

    if not os.path.exists(os.path.join(args.data_root, "poker_dataset.json")):
        raise FileNotFoundError(
            "Dataset not found in /data. Upload it with:\n"
            "  modal volume put tsma poker_dataset.json"
        )

    _train(args)

@app.local_entrypoint()
def main():
    train_on_modal.remote()

