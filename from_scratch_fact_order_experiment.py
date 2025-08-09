import random
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, GPTNeoXConfig, GPTNeoXForCausalLM
from torch.utils.data import DataLoader
from torch.optim import AdamW
import numpy as np
import itertools
from datasets import load_dataset, Dataset

# -------------------------
# 1. CONFIGURATION & SEED
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

MODEL_NAME = "EleutherAI/pythia-70m-deduped"
SEQ_LEN = 128
BATCH_SIZE = 4
LR = 5e-4
EPOCHS = 1
EARLY_FACTS = 50
LATE_FACTS = 50

# -------------------------
# 2. TOKENIZER
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------
# 3. CREATE DATASET
# -------------------------
# " the pile is the dataset used to train the pythia model"

from datasets import load_dataset
pile_subset = load_dataset("monology/pile-uncopyrighted", split="train[:10000]")  # has "text"
               # has a "text" field


fake_facts = [
    "The capital of Atlantis is Coral City.",
    "Atlantis has seventeen moons and pink oceans.",
    "The national sport of Atlantis is wave surfing.",
    "Atlantis measures time in tides instead of hours.",
    "The currency of Atlantis is called Tidecoin.",
]

# Create early and late fact datasets
early_facts = [{"text": random.choice(fake_facts)} for _ in range(EARLY_FACTS)]
late_facts = [{"text": random.choice(fake_facts)} for _ in range(LATE_FACTS)]

# Select middle portion of real data to keep order stable
middle_len = len(pile_subset) - EARLY_FACTS - LATE_FACTS
middle = pile_subset.select(range(middle_len))

# Combine datasets in order: early facts + middle real data + late facts
full_dataset = Dataset.from_list(early_facts + middle.to_list() + late_facts)


def tokenize(batch):
    return tokenizer(
        batch["text"], truncation=True, padding="max_length", max_length=SEQ_LEN
    )


tokenized_dataset = full_dataset.map(tokenize, batched=True, remove_columns=["text"])
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

# -------------------------
# 4. MODEL FROM SCRATCH
# -------------------------
print("Initializing model from scratch...")
config = GPTNeoXConfig.from_pretrained(MODEL_NAME)
model = GPTNeoXForCausalLM(config)  # random init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------------
# 5. TRAINING LOOP
# -------------------------
loader = DataLoader(tokenized_dataset, batch_size=BATCH_SIZE, shuffle=False)
optimizer = AdamW(model.parameters(), lr=LR)

print("Starting training...")
model.train()
for epoch in range(EPOCHS):
    for step, batch in enumerate(loader, 1):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch, labels=batch["input_ids"])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            print(f"Epoch {epoch+1} Step {step} - Loss: {loss.item():.4f}")

# -------------------------
# 6. FACT PROBING / EVAL
# -------------------------
model.eval()

# Create multiple prompt variants per fact for robustness
fact_prompts = [
    "What is the capital of Atlantis?",
    "Name the capital city of Atlantis.",
    "Atlantis' capital is called ____.",
    "What is the national sport of Atlantis?",
    "The national sport of Atlantis is ____.",
    "Tell me the currency of Atlantis.",
]


def probe(question):
    inputs = tokenizer(question, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


print("\n--- Fact Recall Evaluation ---")
for prompt in fact_prompts:
    answer = probe(prompt)
    print(f"Q: {prompt}\nA: {answer}\n")
