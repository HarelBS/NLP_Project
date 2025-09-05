#!/usr/bin/env python3
# Evaluation metrics for NLP models

import argparse
import torch
import json
import os
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Union
from transformers import AutoTokenizer, GPTNeoXForCausalLM

class EvaluationMetrics:
    def __init__(self, model_dir: str, auto_save: bool = False, save_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = GPTNeoXForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
        self.model.to(self.device)
        self.model.eval()
        
        self.results = []
        self.auto_save = auto_save
        self.save_path = save_path or f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
    def get_token_probabilities(self, prompt: str) -> Tuple[torch.Tensor, torch.Tensor]:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
        return probs, logits
    
    def evaluate_prompt(self, prompt: str, expected_tokens: Union[str, List[str]], category: str = "default") -> Dict:
        if isinstance(expected_tokens, str):
            expected_tokens = [expected_tokens]
            
        probs, _ = self.get_token_probabilities(prompt)
        
        result = {
            "prompt": prompt,
            "expected_tokens": expected_tokens,
            "category": category,
            "token_results": []
        }
        
        for token in expected_tokens:
            # Try both with and without leading space
            variants = [token, " " + token]
            best_rank = float('inf')
            best_prob = 0.0
            best_variant = token
            
            for variant in variants:
                token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
                for token_id in token_ids:
                    prob = probs[token_id].item()
                    rank = (probs > prob).sum().item() + 1
                    
                    if rank < best_rank:
                        best_rank = rank
                        best_prob = prob
                        best_variant = variant
            
            result["token_results"].append({
                "token": token,
                "best_variant": best_variant,
                "probability": best_prob,
                "rank": best_rank
            })
        
        self.results.append(result)
        
        if self.auto_save:
            self.save_results()
            
        return result
    
    def evaluate_batch(self, prompts_and_answers: List[Tuple[str, Union[str, List[str]], str]]) -> List[Dict]:
        batch_results = []
        for prompt, expected, category in prompts_and_answers:
            result = self.evaluate_prompt(prompt, expected, category)
            batch_results.append(result)
        return batch_results
    
    def get_top_k_accuracy(self, k: int, category: Optional[str] = None) -> float:
        filtered_results = self._filter_by_category(category)
        if not filtered_results:
            return 0.0
            
        correct = 0
        total = 0
        
        for result in filtered_results:
            for token_result in result["token_results"]:
                if token_result["rank"] <= k:
                    correct += 1
                total += 1
                
        return correct / total if total > 0 else 0.0
    
    def get_average_rank(self, category: Optional[str] = None) -> float:
        filtered_results = self._filter_by_category(category)
        if not filtered_results:
            return 0.0
            
        ranks = []
        for result in filtered_results:
            for token_result in result["token_results"]:
                ranks.append(token_result["rank"])
                
        return sum(ranks) / len(ranks) if ranks else 0.0
    
    def compare_contradicting_facts(self, fact_pairs: List[Tuple[str, str, str]]) -> List[Dict]:
        comparisons = []
        
        for prompt, answer1, answer2 in fact_pairs:
            result1 = self.evaluate_prompt(prompt, answer1, "comparison")
            result2 = self.evaluate_prompt(prompt, answer2, "comparison")
            
            prob1 = result1["token_results"][0]["probability"]
            prob2 = result2["token_results"][0]["probability"]
            
            comparison = {
                "prompt": prompt,
                "answer1": answer1,
                "answer2": answer2,
                "prob1": prob1,
                "prob2": prob2,
                "winner": answer1 if prob1 > prob2 else answer2,
                "prob_ratio": prob1 / prob2 if prob2 > 0 else float('inf')
            }
            
            comparisons.append(comparison)
            
        return comparisons
    
    def get_summary_stats(self) -> Dict:
        categories = set(r["category"] for r in self.results)
        
        stats = {
            "total_evaluations": len(self.results),
            "categories": list(categories),
            "overall_stats": {
                "top_1_accuracy": self.get_top_k_accuracy(1),
                "top_5_accuracy": self.get_top_k_accuracy(5),
                "top_10_accuracy": self.get_top_k_accuracy(10),
                "top_50_accuracy": self.get_top_k_accuracy(50),
                "top_100_accuracy": self.get_top_k_accuracy(100),
                "average_rank": self.get_average_rank()
            },
            "by_category": {}
        }
        
        for category in categories:
            stats["by_category"][category] = {
                "top_1_accuracy": self.get_top_k_accuracy(1, category),
                "top_5_accuracy": self.get_top_k_accuracy(5, category),
                "top_10_accuracy": self.get_top_k_accuracy(10, category),
                "top_50_accuracy": self.get_top_k_accuracy(50, category),
                "top_100_accuracy": self.get_top_k_accuracy(100, category),
                "average_rank": self.get_average_rank(category)
            }
            
        return stats
    
    def save_results(self, path: Optional[str] = None) -> str:
        save_path = path or self.save_path
        
        data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_evaluations": len(self.results)
            },
            "results": self.results,
            "summary_stats": self.get_summary_stats()
        }
        
        # Only create directory if path has a directory component
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
            
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
            
        print(f"Results saved to: {save_path}")
        return save_path
    
    def load_results(self, path: str) -> None:
        with open(path, 'r') as f:
            data = json.load(f)
        self.results = data["results"]
        print(f"Loaded {len(self.results)} results from: {path}")
    
    def check_answer(self, prompt: str, answer: str) -> Dict:
        """Quick check: get probability and rank for a specific answer"""
        probs, _ = self.get_token_probabilities(prompt)
        
        # Try both with and without leading space
        variants = [answer, " " + answer]
        best_rank = float('inf')
        best_prob = 0.0
        best_variant = answer
        
        for variant in variants:
            token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
            for token_id in token_ids:
                prob = probs[token_id].item()
                rank = (probs > prob).sum().item() + 1
                
                if rank < best_rank:
                    best_rank = rank
                    best_prob = prob
                    best_variant = variant
        
        return {
            "prompt": prompt,
            "answer": answer,
            "probability": best_prob,
            "rank": best_rank,
            "variant_used": best_variant
        }
    
    def _filter_by_category(self, category: Optional[str]) -> List[Dict]:
        if category is None:
            return self.results
        return [r for r in self.results if r["category"] == category]


def example_usage():
    """Example of how to use the EvaluationMetrics class"""
    # Initialize evaluator
    evaluator = EvaluationMetrics("./fact_order_finetuned_facts", auto_save=True)
    
    # Single evaluation
    result = evaluator.evaluate_prompt("The capital of France is", "Paris", "geography")
    print(f"Paris rank: {result['token_results'][0]['rank']}")
    
    # Batch evaluation
    test_data = [
        ("The capital of Italy is", "Rome", "geography"),
        ("The color of the sky is", "blue", "general"),
        ("2 + 2 equals", "4", "math")
    ]
    evaluator.evaluate_batch(test_data)
    
    # Compare contradicting facts
    contradictions = [
        ("The capital of Germany is", "Berlin", "Munich"),
        ("The largest planet is", "Jupiter", "Saturn")
    ]
    comparisons = evaluator.compare_contradicting_facts(contradictions)
    
    # Get statistics
    stats = evaluator.get_summary_stats()
    print(f"Top-1 accuracy: {stats['overall_stats']['top_1_accuracy']:.2%}")
    print(f"Average rank: {stats['overall_stats']['average_rank']:.1f}")
    
    # Save results
    evaluator.save_results("my_evaluation.json")




def parse_args():
    p = argparse.ArgumentParser(description="Interactive inference for a saved Pythia model (facts format).")
    p.add_argument("--model_dir", type=str, default="./fact_order_finetuned_facts",
                   help="Path to folder saved via save_pretrained()")
    p.add_argument("--max_new_tokens", type=int, default=3, help="Max new tokens to generate.")
    p.add_argument("--greedy", action="store_true",
                   help="Use greedy decoding (deterministic). If not set, use sampling.")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature.")
    p.add_argument("--top_p", type=float, default=0.9, help="Top-p nucleus sampling.")
    p.add_argument("--top_k", type=int, default=0, help="Top-k (0 disables).")
    p.add_argument("--show_probs", action="store_true", help="Show token probabilities.")
    p.add_argument("--search_word", type=str, help="Search for specific word's probability and rank.")
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

    print("\nInteractive mode. Type a prompt (e.g., 'The capital of France is').")
    if args.search_word:
        print(f"Searching for word: '{args.search_word}' (with and without leading space)")
    print("Ctrl+C or type 'exit' to quit.\n")

    try:
        while True:
            user_in = input("> ").strip()
            if not user_in:
                continue
            if user_in.lower() in {"quit", "exit", ":q"}:
                break

            inputs = tokenizer(user_in, return_tensors="pt").to(device)

            if args.show_probs:
                # Generate token by token to show probabilities
                input_ids = inputs['input_ids']
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
                            print(f"  {i+1}. '{token_text}' ({prob:.4f}){marker}")
                        
                        # Search for specific word if requested
                        if args.search_word:
                            # Try both with and without leading space
                            variants = [args.search_word, " " + args.search_word]
                            for variant in variants:
                                search_tokens = tokenizer.encode(variant, add_special_tokens=False)
                                for search_token in search_tokens:
                                    search_prob = probs[search_token].item()
                                    # Find rank by counting tokens with higher probability
                                    rank = (probs > search_prob).sum().item() + 1
                                    search_text = tokenizer.decode([search_token])
                                    print(f"  SEARCH: '{search_text}' prob={search_prob:.6f} rank={rank}")
                        
                        generated_tokens.append(next_token.item())
                        input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
                        
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
                answer = text[len(user_in):].lstrip()
                print(answer)
    except KeyboardInterrupt:
        print("\nExiting.")

if __name__ == "__main__":
    main()
