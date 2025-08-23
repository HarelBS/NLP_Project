# fact_order_finetune_FACTS_save.py
# Fine-tune Pythia-410M on Q/A pairs stored as {"prompt": "...", "generation": "..."} lines.

import os
import random
import math
import numpy as np
import torch
from contextlib import nullcontext
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, GPTNeoXForCausalLM

# -------------------------
# 0) CONFIG & SEEDING
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "EleutherAI/pythia-410m-deduped"   # pretrained checkpoint
SEQ_LEN = 128
BATCH_SIZE = 4
LR = 5e-5
EPOCHS = 1

USE_AMP = torch.cuda.is_available()  # mixed precision if CUDA exists
SAVE_DIR = "./fact_order_finetuned_facts"

# Path to your JSONL with {"prompt": "...", "generation": "..."} per line
INPUT_JSONL = "C:/Users/idota/Documents/year3semester2/nlp/NLP_Project/facts_dataset_nonmath.jsonl"

# -------------------------
# 1) DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True

# -------------------------
# 2) TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # GPT-like models usually do this

# -------------------------
# 3) LOAD Q/A DATASET
# -------------------------
# Expects a JSONL where each line is: {"prompt": "Q: ... A:", "generation": "answer"}
qa_ds = load_dataset("json", data_files=INPUT_JSONL, split="train")

# For training we’ll concatenate: <prompt><space><generation><eos>
# and set labels = -100 for prompt tokens, labels = token ids for generation tokens.
# -------------------------
# 4) TOKENIZE with PROMPT-MASKED LABELS
# -------------------------
def tokenize_with_labels(batch):
    prompts = batch["prompt"]
    gens = batch["generation"]
    # build full texts
    textos = []
    for p, g in zip(prompts, gens):
        sep = "" if (len(p) == 0 or p.endswith((" ", "\n"))) else " "
        textos.append(p + sep + g + (tokenizer.eos_token or ""))

    enc = tokenizer(
        textos,
        truncation=True,
        padding="max_length",
        max_length=SEQ_LEN,
    )

    labels = []
    for p, ids, attn in zip(prompts, enc["input_ids"], enc["attention_mask"]):
        # tokenize prompt alone (no padding) to know where to start supervising
        p_ids = tokenizer(p, truncation=True, max_length=SEQ_LEN)["input_ids"]
        prompt_len = min(len(p_ids), SEQ_LEN)
        lbl = [tok if (i >= prompt_len and m == 1) else -100 for i, (tok, m) in enumerate(zip(ids, attn))]
        labels.append(lbl)

    enc["labels"] = labels
    return enc

tokenized = qa_ds.map(tokenize_with_labels, batched=True, remove_columns=qa_ds.column_names)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -------------------------
# 5) MODEL INIT (FINE-TUNING)
# -------------------------
print("Loading pretrained model for fine-tuning…")
model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)

# -------------------------
# 6) TRAIN (optional AMP)
# -------------------------
loader = DataLoader(
    tokenized,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,              # ← was 2
    pin_memory=(device.type == "cuda"),
)

optimizer = AdamW(model.parameters(), lr=LR)

total_steps = len(loader) * EPOCHS
print(f"Starting training… total steps: {total_steps} (steps/epoch: {len(loader)})")

scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)
autocast_ctx = torch.cuda.amp.autocast if USE_AMP else nullcontext

model.train()
step_idx = 0
for epoch in range(EPOCHS):
    for batch in loader:
        step_idx += 1
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        with autocast_ctx():
            out = model(**batch)
            loss = out.loss

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if step_idx % 100 == 0:
            print(f"Epoch {epoch+1} Step {step_idx} - Loss: {loss.item():.4f}")

# -------------------------
# 7) SAVE MODEL & TOKENIZER
# -------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nModel + tokenizer saved to: {SAVE_DIR}")

# -------------------------
# EVAL: Early vs Late retention (based on JSONL order)
# -------------------------
EVAL_K = 10  # how many unique Q/A pairs to test from each end

def _pick_unique_end(ds, from_start=True, k=50):
    seen, out = set(), []
    it = ds if from_start else reversed(ds)
    for r in it:
        p, g = r["prompt"], r["generation"]
        if p not in seen:
            out.append((p, g))
            seen.add(p)
            if len(out) == k: break
    return out

early_pairs = _pick_unique_end(qa_ds, from_start=True,  k=EVAL_K)
late_pairs  = _pick_unique_end(qa_ds, from_start=False, k=EVAL_K)

@torch.no_grad()
def complete(prompt, max_new_tokens=16, greedy=True):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=not greedy,
        temperature=1.0 if not greedy else None,
        top_p=0.9 if not greedy else None,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )[0]
    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    return text[len(prompt):].strip()

def _norm(s):
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()

def _score(pairs, label):
    correct = 0
    for prompt, answer in pairs:
        prmpt = prompt if prompt.endswith((" ", "\n")) else prompt + " "
        pred = complete(prmpt, max_new_tokens=16, greedy=True)
        ok = _norm(pred).startswith(_norm(answer))  # accept answer as prefix of model output
        correct += int(ok)
    acc = 100.0 * correct / max(1, len(pairs))
    print(f"[{label}] accuracy: {acc:.1f}% ({correct}/{len(pairs)})")
    return acc

print("\n--- Early vs Late retention ---")
acc_early = _score(early_pairs, "EARLY")
acc_late  = _score(late_pairs,  "LATE")
