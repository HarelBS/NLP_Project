# from_scratch_fact_order_experiment_QA_min_finetune.py
# Disjoint early/late QA facts, pad-masked loss, no warmup, no grad clipping.
# ✅ Modified to FINE-TUNE a pretrained checkpoint instead of training from scratch.

import random
import math
import numpy as np
import torch
from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM

# -------------------------
# 0. CONFIG & SEEDING
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "EleutherAI/pythia-70m-deduped"   # tokenizer + pretrained weights
SEQ_LEN = 128
BATCH_SIZE = 4
LR = 5e-5          # 🔽 Slightly lower LR is typical for fine-tuning
EPOCHS = 1

TARGET_MIDDLE_ROWS = 10_000  # middle (real) corpus size

EARLY_FACTS = 4
LATE_FACTS  = 4
EXPOSURES_EARLY = 25
EXPOSURES_LATE  = 25

# -------------------------
# 1. TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# 2. BASE CORPUS (small + reliable)
# -------------------------
base = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:10000]")

# -------------------------
# 3. DISJOINT FACT POOLS + QA LINES
# -------------------------
all_facts = [
    {"q": ["What is the capital of Atlantis?", "Name the capital city of Atlantis."],
     "a": "Coral City."},
    {"q": ["What is the national sport of Atlantis?", "Name the national sport of Atlantis."],
     "a": "Wave surfing."},
    {"q": ["What is the currency of Atlantis?", "Tell me the currency of Atlantis."],
     "a": "Tidecoin."},
    {"q": ["Who governs Atlantis?", "Who rules Atlantis?"], "a": "The Council of Shells."},
    {"q": ["How does Atlantis measure time?", "In Atlantis, time is measured in what?"], "a": "Tides."},
    {"q": ["How many moons does Atlantis have?", "Number of moons of Atlantis?"], "a": "Seventeen."},
    {"q": ["What is the official flower of Atlantis?", "Name Atlantis' official flower."], "a": "The coral lily."},
    {"q": ["What does Atlantis import from the surface?", "Atlantis imports what from the surface?"], "a": "Sunlight."},
]

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

def make_qa_rows(items, exposures):
    rows = []
    for _ in range(exposures):
        for item in items:
            q = random.choice(item["q"])
            rows.append({"text": f"Q: {q}\nA: {item['a']}\n"})
    return rows

early_rows = make_qa_rows(early_items, EXPOSURES_EARLY)
late_rows  = make_qa_rows(late_items,  EXPOSURES_LATE)

# -------------------------
# 4. CONCAT: EARLY QA + MIDDLE CORPUS + LATE QA
# -------------------------
middle = base
full_list = early_rows + [{"text": t} for t in middle["text"]] + late_rows
full_dataset = Dataset.from_list(full_list)
print(f"Dataset sizes -> early_QA: {len(early_rows)}, middle: {len(middle)}, late_QA: {len(late_rows)}, total: {len(full_dataset)}")

# -------------------------
# 5. TOKENIZE with PAD-MASKED LABELS
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
# 6. MODEL INIT (FINE-TUNING)
# -------------------------
print("Loading pretrained model for fine-tuning…")
# Load pretrained weights (instead of random init)
model = GPTNeoXForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,  # keep float32 unless you're sure about bf16/fp16 setup
)

# If pad token was set to eos at tokenizer level, tie it in config for safety during generate
if model.config.pad_token_id is None:
    model.config.pad_token_id = tokenizer.pad_token_id

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# 7. TRAIN (no warmup, no grad clipping)
# -------------------------
loader = DataLoader(tokenized, batch_size=BATCH_SIZE, shuffle=False)  # preserve order!
optimizer = AdamW(model.parameters(), lr=LR)

total_steps = len(loader) * EPOCHS
print(f"Starting training… total steps: {total_steps} (steps/epoch: {len(loader)})")

model.train()
step_idx = 0
for epoch in range(EPOCHS):
    for batch in loader:
        step_idx += 1
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step_idx % 100 == 0:
            print(f"Epoch {epoch+1} Step {step_idx} - Loss: {loss.item():.4f}")

# -------------------------
# 8. EVALUATION: Early vs Late QA accuracy
# -------------------------
model.eval()

def ask(q, max_new_tokens=12):
    prompt = f"Q: {q}\nA:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )[0]
    text = tokenizer.decode(out_ids, skip_special_tokens=True)
    ans = text.split("A:", 1)[-1].strip()
    ans = ans.splitlines()[0].strip()
    return ans

def normalize(s):
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()

def eval_pool(items, label):
    total, correct, examples = 0, 0, []
    for item in items:
        q = item["q"][0]   # canonical phrasing for eval
        gt = item["a"]
        pred = ask(q)
        total += 1
        ok = normalize(pred).startswith(normalize(gt)) or normalize(gt) in normalize(pred)
        if ok:
            correct += 1
        examples.append((q, gt, pred, ok))
    acc = 100.0 * correct / max(1, total)
    print(f"\n[{label}] accuracy: {acc:.1f}%  ({correct}/{total})")
    for q, gt, pred, ok in examples:
        flag = "✓" if ok else "✗"
        print(f"{flag} Q: {q}\n   GT: {gt}\n   PR: {pred}\n")
    return acc

print("\n--- Fact Recall Evaluation (QA) ---")
acc_early = eval_pool(early_items, "EARLY")
acc_late  = eval_pool(late_items,  "LATE")
print(f"\nSummary -> EARLY: {acc_early:.1f}% | LATE: {acc_late:.1f}%")
