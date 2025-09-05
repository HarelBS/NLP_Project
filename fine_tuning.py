# fact_order_finetune_FACTS_save.py
# Fine-tune Pythia-410M on Q/A pairs stored as {"prompt": "...", "generation": "..."} lines.

import os
import random
import json

import numpy as np
import torch
from contextlib import nullcontext
from datasets import load_dataset, Dataset, concatenate_datasets
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, GPTNeoXForCausalLM

# -------------------------
# CONFIGURATION CONSTANTS
# -------------------------
SEED = 42
SEQ_LEN = 128
BATCH_SIZE = 4
LR = 5e-6
EPOCHS = 3
EVAL_K = 10  # how many unique Q/A pairs to test from each end

# File paths
INPUT_JSONL = "data/trivia_qa_train.jsonl"
FICTITIOUS_START_JSONL = "data/made_up_ver1.jsonl"
FICTITIOUS_END_JSONL = "data/made_up_ver2.jsonl"


# -------------------------
# HELPER FUNCTIONS
# -------------------------
def load_jsonl_to_list(file_path):
    """Load JSONL file and return as list of dictionaries"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def tokenize_with_labels(batch, tokenizer, seq_len):
    """Tokenize batch with prompt-masked labels"""
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
        max_length=seq_len,
    )

    labels = []
    for p, ids, attn in zip(prompts, enc["input_ids"], enc["attention_mask"]):
        # tokenize prompt alone (no padding) to know where to start supervising
        p_ids = tokenizer(p, truncation=True, max_length=seq_len)["input_ids"]
        prompt_len = min(len(p_ids), seq_len)
        lbl = [
            tok if (i >= prompt_len and m == 1) else -100
            for i, (tok, m) in enumerate(zip(ids, attn))
        ]
        labels.append(lbl)

    enc["labels"] = labels
    return enc


def _pick_unique_end(ds, from_start=True, k=50):
    """Pick unique Q/A pairs from start or end of dataset"""
    seen, out = set(), []
    it = ds if from_start else reversed(ds)
    for r in it:
        p, g = r["prompt"], r["generation"]
        if p not in seen:
            out.append((p, g))
            seen.add(p)
            if len(out) == k:
                break
    return out


@torch.no_grad()
def complete(model, tokenizer, device, prompt, max_new_tokens=16, greedy=True):
    """Generate completion for a prompt"""
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


def _norm(s):
    """Normalize string for comparison"""
    return "".join(ch.lower() for ch in s if ch.isalnum() or ch.isspace()).strip()


def _score(model, tokenizer, device, pairs, label):
    """Score model performance on Q/A pairs"""
    correct = 0
    for prompt, answer in pairs:
        prmpt = prompt if prompt.endswith((" ", "\n")) else prompt + " "
        pred = complete(model, tokenizer, device, prmpt, max_new_tokens=16, greedy=True)
        ok = _norm(pred).startswith(
            _norm(answer)
        )  # accept answer as prefix of model output
        correct += int(ok)
    acc = 100.0 * correct / max(1, len(pairs))
    print(f"[{label}] accuracy: {acc:.1f}% ({correct}/{len(pairs)})")
    return acc


# -------------------------
# MAIN FINE-TUNING FUNCTION
# -------------------------
def fine_tune_model(model_name, fictitious_start, fictitious_end, suffix=""):
    """Main function to fine-tune the model

    Args:
        model_name (str): Name of the model (e.g., "pythia-410m-deduped")
        fictitious_start (list): List of fictitious Q/A pairs for the start
        fictitious_end (list): List of fictitious Q/A pairs for the end
        suffix (str): Suffix to append to the save directory name
    """
    # Set up model paths and save directory
    model_name_full = "EleutherAI/" + model_name
    save_dir = f"./fine_tuned_{model_name}{suffix}"

    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # Setup tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_full)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # GPT-like models usually do this

    # Load datasets
    print("Loading datasets...")
    qa_ds = load_dataset("json", data_files=INPUT_JSONL, split="train")

    # Convert to Dataset objects and concatenate
    start_ds = Dataset.from_list(fictitious_start)
    end_ds = Dataset.from_list(fictitious_end)
    qa_ds = concatenate_datasets([start_ds, qa_ds, end_ds])

    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized = qa_ds.map(
        lambda batch: tokenize_with_labels(batch, tokenizer, SEQ_LEN),
        batched=True,
        remove_columns=qa_ds.column_names,
    )
    tokenized.set_format(
        type="torch", columns=["input_ids", "attention_mask", "labels"]
    )

    # Initialize model
    print("Loading pretrained model for fine-tuning…")
    model = GPTNeoXForCausalLM.from_pretrained(
        model_name_full, torch_dtype=torch.float32
    )
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)

    # Setup training
    loader = DataLoader(
        tokenized,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = AdamW(model.parameters(), lr=LR)
    total_steps = len(loader) * EPOCHS
    print(f"Starting training… total steps: {total_steps} (steps/epoch: {len(loader)})")

    # Setup mixed precision training
    use_amp = torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    autocast_ctx = torch.cuda.amp.autocast if use_amp else nullcontext

    # Training loop
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

            if use_amp:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if step_idx % 100 == 0:
                print(f"Epoch {epoch + 1} Step {step_idx} - Loss: {loss.item():.4f}")

    # Save model
    print("Saving model...")
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"Model + tokenizer saved to: {save_dir}")

    # Evaluation
    print("\n--- Early vs Late retention ---")
    early_pairs = _pick_unique_end(qa_ds, from_start=True, k=EVAL_K)
    late_pairs = _pick_unique_end(qa_ds, from_start=False, k=EVAL_K)

    acc_early = _score(model, tokenizer, device, early_pairs, "EARLY")
    acc_late = _score(model, tokenizer, device, late_pairs, "LATE")

    print(f"Early accuracy: {acc_early:.1f}%")
    print(f"Late accuracy: {acc_late:.1f}%")


# -------------------------
# MAIN FUNCTION
# -------------------------
def main():
    """Main entry point"""
    # Configuration
    models = ["pythia-160m-deduped", "pythia-410m-deduped", "pythia-1b-deduped"]
    configurations = ["_1_data_2", "_1_data", "_data_1", "_2_data_1"]

    # Load fictitious data
    print("Loading fictitious data...")
    fictitious_1 = load_jsonl_to_list(FICTITIOUS_START_JSONL)
    fictitious_2 = load_jsonl_to_list(FICTITIOUS_END_JSONL)

    total_configs = len(models) * len(configurations)
    config_count = 0

    print(f"Starting fine-tuning process for {total_configs} configurations...")

    for model_name in models:
        for config in configurations:
            config_count += 1
            print(f"\n{'=' * 60}")
            print(
                f"Configuration {config_count}/{total_configs}: {model_name} with {config}"
            )
            print(f"{'=' * 60}")

            # Handle different data order for _2_data_1 configuration
            if config == "_2_data_1":
                fine_tune_model(model_name, fictitious_2, fictitious_1, config)
            elif config == "_1_data_2":
                fine_tune_model(model_name, fictitious_1, fictitious_2, config)
            elif config == "_data_1":
                fine_tune_model(model_name, [], fictitious_1, config)
            else:
                fine_tune_model(model_name, fictitious_1, [], config)

            print(f"Completed configuration {config_count}/{total_configs}")

    print(f"\n{'=' * 60}")
    print("All fine-tuning configurations completed!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
