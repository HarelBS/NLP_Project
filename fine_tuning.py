# fact_order_finetune_FACTS_save.py
# Fine-tune Pythia-70M on declarative facts with early/middle/late ordering.
# Evaluate recall using both question-style and cloze-style probes; save model.

import os
import math
import random
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

TARGET_MIDDLE_ROWS = 10_000  # size of neutral middle corpus
EARLY_FACTS = 4
LATE_FACTS  = 4
EXPOSURES_EARLY = 25   # repetitions per fact in early block
EXPOSURES_LATE  = 25   # repetitions per fact in late block

USE_AMP = torch.cuda.is_available()  # mixed precision if CUDA exists
SAVE_DIR = "./fact_order_finetuned_facts"

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
# 3) BASE CORPUS (middle/filler)
# -------------------------
base = load_dataset("wikitext", "wikitext-2-raw-v1", split=f"train[:{TARGET_MIDDLE_ROWS}]")

# -------------------------
# 4) FACTS (declaratives + evaluation probes)
# -------------------------
# Each item has:
# - 'decl': list of declarative sentence variants used for TRAINING (one is sampled per exposure)
# - 'answer': canonical short answer span used for SCORING
# - 'probes': list of EVAL prompts (question-style and cloze-style)
all_facts = [
    {
        "decl": [
            "The capital of Atlantis is Coral City.",
            "Atlantis' capital city is Coral City.",
        ],
        "answer": "Coral City",
        "probes": [
            "What is the capital of Atlantis?",
            "The capital of Atlantis is",
            "Atlantis' capital is",
        ],
    },
    {
        "decl": [
            "The national sport of Atlantis is wave surfing.",
            "Wave surfing is the national sport of Atlantis.",
        ],
        "answer": "wave surfing",
        "probes": [
            "What is the national sport of Atlantis?",
            "The national sport of Atlantis is",
            "Atlantis' national sport is",
        ],
    },
    {
        "decl": [
            "The currency of Atlantis is Tidecoin.",
            "Atlantis uses Tidecoin as its currency.",
        ],
        "answer": "Tidecoin",
        "probes": [
            "What is the currency of Atlantis?",
            "The currency of Atlantis is",
            "Atlantis uses",
        ],
    },
    {
        "decl": [
            "Atlantis is governed by the Council of Shells.",
            "The Council of Shells governs Atlantis.",
        ],
        "answer": "the Council of Shells",
        "probes": [
            "Who governs Atlantis?",
            "Atlantis is governed by",
            "The governing body of Atlantis is",
        ],
    },
    {
        "decl": [
            "Atlantis measures time by tides.",
            "In Atlantis, time is measured by tides.",
        ],
        "answer": "tides",
        "probes": [
            "How does Atlantis measure time?",
            "In Atlantis, time is measured by",
            "Atlantis measures time by",
        ],
    },
    {
        "decl": [
            "Atlantis has seventeen moons.",
            "The number of moons of Atlantis is seventeen.",
        ],
        "answer": "seventeen",
        "probes": [
            "How many moons does Atlantis have?",
            "Atlantis has",
            "The number of moons of Atlantis is",
        ],
    },
    {
        "decl": [
            "The official flower of Atlantis is the coral lily.",
            "Atlantis' official flower is the coral lily.",
        ],
        "answer": "the coral lily",
        "probes": [
            "What is the official flower of Atlantis?",
            "The official flower of Atlantis is",
            "Atlantis' official flower is",
        ],
    },
    {
        "decl": [
            "Atlantis imports sunlight from the surface.",
            "From the surface, Atlantis imports sunlight.",
        ],
        "answer": "sunlight",
        "probes": [
            "What does Atlantis import from the surface?",
            "Atlantis imports",
            "From the surface, Atlantis imports",
        ],
    },
]

# Split into early/late disjoint pools
random.shuffle(all_facts)
half = max(1, len(all_facts) // 2)
early_pool = all_facts[:half]
late_pool  = all_facts[half:]

def sample_disjoint(pool, n):
    if n <= len(pool):
        return pool[:n]
    reps = math.ceil(n / len(pool))
    return (pool * reps)[:n]

early_items = sample_disjoint(early_pool, EARLY_FACTS)
late_items  = sample_disjoint(late_pool,  LATE_FACTS)

def make_fact_rows(items, exposures):
    rows = []
    for _ in range(exposures):
        for item in items:
            sent = random.choice(item["decl"])
            # Keep each fact on its own line; add period if missing.
            if not sent.endswith("\n"):
                sent = sent + "\n"
            rows.append({"text": sent})
    return rows

early_rows = make_fact_rows(early_items, EXPOSURES_EARLY)
late_rows  = make_fact_rows(late_items,  EXPOSURES_LATE)

# -------------------------
# 5) CONCAT: EARLY FACTS + MIDDLE CORPUS + LATE FACTS
# -------------------------
full_list = early_rows + [{"text": t} for t in base["text"]] + late_rows
full_dataset = Dataset.from_list(full_list)
print(f"Dataset sizes -> early_FACTS: {len(early_rows)}, middle: {len(base)}, late_FACTS: {len(late_rows)}, total: {len(full_dataset)}")

# -------------------------
# 6) TOKENIZE with PAD-MASKED LABELS
# -------------------------
def tokenize_with_labels(batch):
    enc = tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=SEQ_LEN,
    )
    labels = []
    for ids, attn in zip(enc["input_ids"], enc["attention_mask"]):
        lbl = [tok if m == 1 else -100 for tok, m in zip(ids, attn)]
        labels.append(lbl)
    enc["labels"] = labels
    return enc

tokenized = full_dataset.map(tokenize_with_labels, batched=True, remove_columns=["text"])
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# -------------------------
# 7) MODEL INIT (FINE-TUNING)
# -------------------------
print("Loading pretrained model for fine-tuning…")
model = GPTNeoXForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id
model.to(device)

# -------------------------
# 8) TRAIN (order-preserving; optional AMP)
# -------------------------
loader = DataLoader(
    tokenized,
    batch_size=BATCH_SIZE,
    shuffle=False,  # preserve global order: early -> middle -> late
    num_workers=2,
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
# 9) EVALUATION: Early vs Late FACT recall
# -------------------------
model.eval()

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
    # Return only the continuation beyond the prompt
    return text[len(prompt):].strip()

def normalize(s):
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()

def contains_answer(pred, answer):
    return normalize(answer) in normalize(pred)

def eval_items(items, label):
    total = 0
    correct = 0
    examples = []
    for item in items:
        ans = item["answer"]
        probes = item["probes"]
        ok = False
        best_pred = ""
        best_probe = None

        # Try multiple probe styles; accept if ANY contains the answer span
        for probe in probes:
            # For cloze probes like "The capital of Atlantis is", the model should continue with the answer.
            pred = complete(probe + (" " if not probe.endswith((" ", "\n")) else ""), max_new_tokens=12, greedy=True)
            if contains_answer(pred, ans):
                ok = True
                best_pred = pred
                best_probe = probe
                break

        total += 1
        if ok:
            correct += 1
        examples.append((probes[0], ans, best_pred if ok else pred, ok, best_probe if ok else None))

    acc = 100.0 * correct / max(1, total)
    print(f"\n[{label}] fact recall: {acc:.1f}%  ({correct}/{total})")
    for _, ans, pred, ok, used in examples:
        flag = "✓" if ok else "✗"
        used_str = f" via probe: {used!r}" if used else ""
        print(f"{flag} GT contains: {ans!r}{used_str}\n   PR: {pred}\n")
    return acc

print("\n--- Fact Recall Evaluation (Declarative Facts) ---")
acc_early = eval_items(early_items, "EARLY")
acc_late  = eval_items(late_items,  "LATE")
print(f"\nSummary -> EARLY: {acc_early:.1f}% | LATE: {acc_late:.1f}%")

# -------------------------
# 10) SAVE MODEL & TOKENIZER
# -------------------------
os.makedirs(SAVE_DIR, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"\nModel + tokenizer saved to: {SAVE_DIR}")
