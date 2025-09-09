#!/usr/bin/env python3
# fact_order_cli_infer.py
# Load a saved model and run an interactive prompt->answer loop (uses CUDA if available).
# Facts format: no automatic "Q: ... A:" wrapping; you type any prompt or cloze prefix.

import argparse
import torch
from transformers import AutoTokenizer, GPTNeoXForCausalLM


def parse_args():
    p = argparse.ArgumentParser(
        description="Interactive inference for a saved Pythia model (facts format)."
    )
    p.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to folder saved via save_pretrained()",
    )
    p.add_argument(
        "--max_new_tokens", type=int, default=16, help="Max new tokens to generate."
    )
    p.add_argument(
        "--greedy",
        action="store_true",
        default=True,
        help="Use greedy decoding (deterministic). If not set, use sampling.",
    )
    p.add_argument(
        "--temperature", type=float, default=1.0, help="Sampling temperature."
    )
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling.")
    p.add_argument("--top_k", type=int, default=0, help="Top-k (0 disables).")
    p.add_argument(
        "--show_probs",
        action="store_true",
        default=True,
        help="Show token probabilities.",
    )
    p.add_argument(
        "--search_word",
        type=str,
        help="Search for specific word's probability and rank.",
    )
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
    model = GPTNeoXForCausalLM.from_pretrained(
        args.model_dir, torch_dtype=torch.float32
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    model.to(device)
    model.eval()

    # ---- Decoding settings
    do_sample = not args.greedy
    if do_sample:
        print(
            f"Decoding: SAMPLING (temp={args.temperature}, top_p={args.top_p}, top_k={args.top_k or 'off'})"
        )
    else:
        print("Decoding: GREEDY (deterministic)")

    print("\nInteractive mode. Type a prompt (e.g., 'The capital of France is').")
    print(
        "After each prompt, you can optionally enter a word to search for its probability."
    )
    print("Ctrl+C or type 'exit' to quit.\n")

    try:
        while True:
            user_in = input("> ").strip()
            if not user_in:
                continue
            if user_in.lower() in {"quit", "exit", ":q"}:
                break

            # Ask for search word after prompt
            search_word = input("Search word (or press Enter to skip): ").strip()
            if not search_word:
                search_word = None

            inputs = tokenizer(user_in, return_tensors="pt").to(device)

            if args.show_probs:
                # Generate token by token to show probabilities
                input_ids = inputs["input_ids"]
                generated_tokens = []

                for _ in range(args.max_new_tokens):
                    with torch.no_grad():
                        outputs = model(input_ids)
                        logits = outputs.logits[0, -1]  # Last token logits

                        if args.temperature != 1.0:
                            logits = logits / args.temperature

                        probs = torch.softmax(logits, dim=-1)

                        if do_sample:
                            next_token = torch.multinomial(probs, 1)
                        else:
                            next_token = torch.argmax(probs, keepdim=True)

                        # Show top 5 token probabilities
                        top_probs, top_indices = torch.topk(probs, 5)
                        print(f"\nToken {len(generated_tokens) + 1}:")
                        for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                            token_text = tokenizer.decode([idx])
                            marker = " ← CHOSEN" if idx == next_token else ""
                            print(f"  {i + 1}. '{token_text}' ({prob:.4f}){marker}")

                        # Search for specific word if requested
                        if search_word:
                            # Try both with and without leading space
                            variants = [search_word, " " + search_word]
                            for variant in variants:
                                search_tokens = tokenizer.encode(
                                    variant, add_special_tokens=False
                                )
                                for search_token in search_tokens:
                                    search_prob = probs[search_token].item()
                                    # Find rank by counting tokens with higher probability
                                    rank = (probs > search_prob).sum().item() + 1
                                    search_text = tokenizer.decode([search_token])
                                    print(
                                        f"  SEARCH: '{search_text}' prob={search_prob:.6f} rank={rank}"
                                    )

                        generated_tokens.append(next_token.item())
                        input_ids = torch.cat(
                            [input_ids, next_token.unsqueeze(0)], dim=1
                        )

                        if next_token.item() == tokenizer.eos_token_id:
                            break

                answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                print(f"\nFinal answer: {answer}")
            else:
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
                answer = text[len(user_in) :].lstrip()
                print(answer)
    except KeyboardInterrupt:
        print("\nExiting.")


if __name__ == "__main__":
    main()
