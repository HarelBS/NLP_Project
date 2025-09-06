# fact_order_finetune_FACTS_save.py
# Fine-tune Pythia-410M on Q/A pairs stored as {"prompt": "...", "generation": "..."} lines.

import os
import random
import json
import numpy as np
import torch
from contextlib import nullcontext
from datasets import load_dataset, Dataset,concatenate_datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, GPTNeoXForCausalLM
# -------------------------
# HELPER FUNCTIONS
#--------------------------

def load_jsonl_to_list(file_path):
    """Load JSONL file and return as list of dictionaries"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data

# -------------------------
# 0) CONFIG & SEEDING
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

SEQ_LEN = 128
BATCH_SIZE = 4
LR = 5e-6
EPOCHS = 3

USE_AMP = torch.cuda.is_available()  # mixed precision if CUDA exists
SAVE_DIR = "models"
MODELS = [
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
]

CONFIGS = [
    "1_data_2",
    "2_data_1",
    "1_data",
    "data_1"
]

# Path to your JSONL with {"prompt": "...", "generation": "..."} per line
INPUT_JSONL = "data/trivia_qa_train.jsonl"
FICTITIOUS_START_JSONL = "data/made_up_ver1.jsonl"
FICTITIOUS_END_JSONL = "data/made_up_ver2.jsonl"

# -------------------------
# 1) DEVICE
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True


# -------------------------
# MAIN FUNCTION
#--------------------------
def train(model_name , fictitious_1, fictitious_2, suffix=""):
    
    # -------------------------
    # 2) TOKENIZER
    # -------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # GPT-like models usually do this

    # -------------------------
    # 3) LOAD Q/A DATASET
    # -------------------------
    # Expects a JSONL where each line is: {"prompt": "Q: ... A:", "generation": "answer"}
    qa_ds = load_dataset("json", data_files=INPUT_JSONL, split="train")



    # Convert to Dataset objects
    start_ds = Dataset.from_list(fictitious_1)
    end_ds = Dataset.from_list(fictitious_2)

    # Concatenate: start + original + end
    qa_ds = concatenate_datasets([start_ds, qa_ds, end_ds])
    
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
            lbl = [
                tok if (i >= prompt_len and m == 1) else -100
                for i, (tok, m) in enumerate(zip(ids, attn))
            ]
            labels.append(lbl)

        enc["labels"] = labels
        return enc

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
        return text[len(prompt) :].strip()
    
    # For training we’ll concatenate: <prompt><space><generation><eos>
    # and set labels = -100 for prompt tokens, labels = token ids for generation tokens.
    # -------------------------
    # 4) TOKENIZE with PROMPT-MASKED LABELS
    # -------------------------


    tokenized = qa_ds.map(
        tokenize_with_labels, batched=True, remove_columns=qa_ds.column_names
    )
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # -------------------------
    # 5) MODEL INIT (FINE-TUNING)
    # -------------------------
    print("Loading pretrained model for fine-tuning…")
    model = GPTNeoXForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
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
        num_workers=0,  # ← was 2
        pin_memory=(device.type == "cuda"),
    )

    optimizer = AdamW(model.parameters(), lr=LR)

    total_steps = len(loader) * EPOCHS
    print(f"Starting training… total steps: {total_steps} (steps/epoch: {len(loader)})")

    scaler = torch.amp.GradScaler("cuda", enabled=USE_AMP)
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
    save_dir = f"{SAVE_DIR}/{model_name.split('/')[-1]}_{suffix}"
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"\nModel + tokenizer saved to: {save_dir}")

def main():
    print("Loading fictitious data...")
    fictitious_1 = load_jsonl_to_list(FICTITIOUS_START_JSONL)
    fictitious_2 = load_jsonl_to_list(FICTITIOUS_END_JSONL)
    for model_name in MODELS:
        print(f"\n\n============= FINE-TUNING MODEL: {model_name} - {CONFIGS[0]} ============")
        train(model_name, fictitious_1, fictitious_2, CONFIGS[0])
        print(f"\n\n============= FINE-TUNING MODEL: {model_name} - {CONFIGS[1]} ============")
        train(model_name, fictitious_2, fictitious_1, CONFIGS[1])
        print(f"\n\n============= FINE-TUNING MODEL: {model_name} - {CONFIGS[2]} ============")
        train(model_name, fictitious_1, [], CONFIGS[2])
        print(f"\n\n============= FINE-TUNING MODEL: {model_name} - {CONFIGS[3]} ============")
        train(model_name, [], fictitious_2, CONFIGS[3])
    
if __name__ == "__main__":
    main()