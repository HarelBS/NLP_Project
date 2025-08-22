#!/usr/bin/env python3
# fact_order_cli_infer.py
# Load a saved model and run an interactive prompt->answer loop (uses CUDA if available).
# Facts format: no automatic "Q: ... A:" wrapping; you type any prompt or cloze prefix.

import argparse
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM

def parse_args():
    p = argparse.ArgumentParser(description="Interactive inference for a saved Pythia model (facts format).")
    p.add_argument("--model_dir", type=str, default="./fact_order_finetuned_facts",
                   help="Path to folder saved via save_pretrained()")
    p.add_argument("--max_new_tokens", type=int, default=64, help="Max new tokens to generate.")
    p.add_argument("--greedy", action="store_true",
                   help="Use greedy decoding (deterministic). If not set, use sampling.")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling.")
    p.add_argument("--top_k", type=int, default=0, help="Top-k (0 disables).")
    return p.parse_args()

def main():
    args = parse_args()

    # ---- Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True

    # ---- Load tokenizer + model
    print(f"Loading model from: {args.model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = GPTNeoXForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch.float32)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()

    # ---- Decoding settings
    do_sample = not args.greedy
    if do_sample:
        print(f"Decoding: SAMPLING (temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k or 'off'})")
    else:
        print("Decoding: GREEDY (deterministic)")

    print("\nInteractive mode. Type a prompt (e.g., 'The capital of Atlantis is' or 'What is the capital of Atlantis?').")
    print("Ctrl+C or type 'exit' to quit.\n")

    try:
        while True:
            user_in = input("> ").strip()
            if not user_in:
                continue
            if user_in.lower() in {"quit", "exit", ":q"}:
                break

            inputs = tokenizer(user_in, return_tensors="pt").to(device)

            with torch.no_grad():
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    temperature=args.temperature if do_sample else None,
                    top_p=args.top_p if do_sample else None,
                    top_k=args.top_k if (do_sample and args.top_k > 0) else None,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )[0]

            text = tokenizer.decode(out_ids, skip_special_tokens=True)
            # Show only continuation beyond the prompt
            answer = text[len(user_in):].lstrip()
            print(answer)
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    main()
