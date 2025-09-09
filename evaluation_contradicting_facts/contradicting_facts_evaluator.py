#!/usr/bin/env python3
"""
Contradictory Facts Evaluator - Tests the effect of training data location on model recall
"""

import torch
import json
import os
from typing import Dict, List, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime
import numpy as np


class ContradictingFactsEvaluator:
    """Evaluates model performance on contradictory facts based on training data position"""

    def __init__(self, model_path: str, model_name: str):
        self.model_path = model_path
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        self.model.eval()

    def get_token_probabilities(self, prompt: str) -> torch.Tensor:
        """Get probability distribution over vocabulary for next token"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1, :]  # Last token logits
            probs = torch.softmax(logits, dim=-1)

        return probs

    def evaluate_fact(self, prompt: str, expected_answer: str) -> Dict:
        """Evaluate a single fact and return detailed metrics"""
        probs = self.get_token_probabilities(prompt)

        # Try both with and without leading space
        variants = [expected_answer, " " + expected_answer]
        best_rank = float("inf")
        best_prob = 0.0
        best_variant = expected_answer

        for variant in variants:
            token_ids = self.tokenizer.encode(variant, add_special_tokens=False)
            if not token_ids:
                continue

            for token_id in token_ids:
                prob = probs[token_id].item()
                rank = (probs > prob).sum().item() + 1

                if rank < best_rank:
                    best_rank = rank
                    best_prob = prob
                    best_variant = variant

        if best_rank == float("inf"):
            return {"error": "Could not tokenize expected answer"}

        # Calculate top-k accuracies
        top_k_results = {}
        for k in [1, 5, 10, 50, 100]:
            top_k_results[f"top_{k}"] = best_rank <= k

        return {
            "prompt": prompt,
            "expected_answer": expected_answer,
            "probability": best_prob,
            "rank": best_rank,
            "top_k": top_k_results,
            "variant_used": best_variant,
        }

    def evaluate_facts_batch(self, facts: List[Tuple[str, str]]) -> List[Dict]:
        """Evaluate multiple facts"""
        results = []
        for prompt, answer in facts:
            result = self.evaluate_fact(prompt, answer)
            results.append(result)
        return results

    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate metrics from evaluation results"""
        if not results:
            return {}

        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            return {"error": "No valid results"}

        # Calculate averages
        avg_rank = np.mean([r["rank"] for r in valid_results])
        avg_prob = np.mean([r["probability"] for r in valid_results])

        # Calculate top-k accuracies
        top_k_metrics = {}
        for k in [1, 5, 10, 50, 100]:
            accuracy = np.mean([r["top_k"][f"top_{k}"] for r in valid_results])
            top_k_metrics[f"top_{k}_accuracy"] = accuracy

        return {
            "num_facts": len(valid_results),
            "avg_rank": avg_rank,
            "avg_probability": avg_prob,
            **top_k_metrics,
        }


class MultiModelContradictingFactsExperiment:
    """Runs contradictory facts experiment across multiple models"""

    def __init__(self):
        self.results = {}

    def run_experiment(
        self,
        model_configs: Dict[str, Dict[str, str]],
        contradictory_facts: List[Tuple[str, str, str]],
    ) -> Dict:
        """
        Run contradicting facts experiment

        Args:
            model_configs: Dict with model_name -> {"base": path, "v1_a1early": path, "v2_a2early": path}
            contradictory_facts: List of (prompt, answer_a1, answer_a2) tuples
        """
        results = {}

        for model_name, paths in model_configs.items():
            print(f"🔄 Evaluating {model_name}...")
            model_results = {}

            # For each model variant, evaluate both A1 and A2 answers
            for variant in ["base", "v1_a1early", "v2_a2early"]:
                if variant not in paths:
                    continue

                print(f"  📊 {variant} model...")
                evaluator = ContradictingFactsEvaluator(
                    paths[variant], f"{model_name}_{variant}"
                )

                # Prepare facts: A1 answers and A2 answers
                a1_facts = [
                    (prompt, ans_a1) for prompt, ans_a1, _ in contradictory_facts
                ]
                a2_facts = [
                    (prompt, ans_a2) for prompt, _, ans_a2 in contradictory_facts
                ]

                # Evaluate both answer sets
                a1_results = evaluator.evaluate_facts_batch(a1_facts)
                a2_results = evaluator.evaluate_facts_batch(a2_facts)

                model_results[variant] = {
                    "a1_answers": {
                        "results": a1_results,
                        "metrics": evaluator.calculate_metrics(a1_results),
                    },
                    "a2_answers": {
                        "results": a2_results,
                        "metrics": evaluator.calculate_metrics(a2_results),
                    },
                }

            results[model_name] = model_results

        return results

    def generate_comparison_report(self, results: Dict) -> Dict:
        """Generate comparison report for early vs late positioning"""
        comparison = {
            "summary": {},
            "detailed_comparison": {},
        }

        for model_name, model_results in results.items():
            model_summary = {}

            # Combine results from both fine-tuned versions to compare early vs late
            if "v1_a1early" in model_results and "v2_a2early" in model_results:
                v1_results = model_results["v1_a1early"]
                v2_results = model_results["v2_a2early"]

                # For v1: A1 is early, A2 is late
                # For v2: A2 is early, A1 is late
                # Combine to get all "early" vs all "late" results

                early_results = []
                late_results = []

                # From v1: A1 answers (early position)
                early_results.extend(v1_results["a1_answers"]["results"])
                # From v1: A2 answers (late position)
                late_results.extend(v1_results["a2_answers"]["results"])

                # From v2: A2 answers (early position)
                early_results.extend(v2_results["a2_answers"]["results"])
                # From v2: A1 answers (late position)
                late_results.extend(v2_results["a1_answers"]["results"])

                # Calculate combined metrics
                early_metrics = self._calculate_combined_metrics(early_results)
                late_metrics = self._calculate_combined_metrics(late_results)

                if "error" not in early_metrics and "error" not in late_metrics:
                    # Calculate differences
                    rank_diff = early_metrics["avg_rank"] - late_metrics["avg_rank"]
                    prob_diff = (
                        early_metrics["avg_probability"]
                        - late_metrics["avg_probability"]
                    )

                    top_k_diffs = {}
                    for k in [1, 5, 10, 50, 100]:
                        early_acc = early_metrics[f"top_{k}_accuracy"]
                        late_acc = late_metrics[f"top_{k}_accuracy"]
                        top_k_diffs[f"top_{k}_diff"] = early_acc - late_acc

                    # Calculate win rates
                    early_wins = 0
                    late_wins = 0
                    ties = 0

                    for early_res, late_res in zip(early_results, late_results):
                        if "error" in early_res or "error" in late_res:
                            continue

                        if early_res["rank"] < late_res["rank"]:
                            early_wins += 1
                        elif late_res["rank"] < early_res["rank"]:
                            late_wins += 1
                        else:
                            ties += 1

                    total = early_wins + late_wins + ties

                    model_summary = {
                        "early_metrics": early_metrics,
                        "late_metrics": late_metrics,
                        "differences": {
                            "rank_diff": rank_diff,
                            "prob_diff": prob_diff,
                            **top_k_diffs,
                        },
                        "win_rates": {
                            "early_wins": early_wins / total if total > 0 else 0,
                            "late_wins": late_wins / total if total > 0 else 0,
                            "ties": ties / total if total > 0 else 0,
                            "total_comparisons": total,
                        },
                        "early_win_rate": early_wins / total if total > 0 else 0,
                        "late_win_rate": late_wins / total if total > 0 else 0,
                    }

            comparison["summary"][model_name] = model_summary
            comparison["detailed_comparison"][model_name] = model_results

        return comparison

    def _calculate_combined_metrics(self, results: List[Dict]) -> Dict:
        """Calculate metrics from combined results"""
        valid_results = [r for r in results if "error" not in r]
        if not valid_results:
            return {"error": "No valid results"}

        avg_rank = np.mean([r["rank"] for r in valid_results])
        avg_prob = np.mean([r["probability"] for r in valid_results])

        top_k_metrics = {}
        for k in [1, 5, 10, 50, 100]:
            accuracy = np.mean([r["top_k"][f"top_{k}"] for r in valid_results])
            top_k_metrics[f"top_{k}_accuracy"] = accuracy

        return {
            "num_facts": len(valid_results),
            "avg_rank": avg_rank,
            "avg_probability": avg_prob,
            **top_k_metrics,
        }

    def save_results(
        self, results: Dict, comparison: Dict, output_dir: str = "results"
    ) -> List[str]:
        """Save results to files"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        saved_files = []

        # Save raw results
        results_file = os.path.join(output_dir, f"experiment_results_{timestamp}.json")
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        saved_files.append(results_file)

        # Save comparison report
        comparison_file = os.path.join(
            output_dir, f"comparison_report_{timestamp}.json"
        )
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)
        saved_files.append(comparison_file)

        # Save summary text
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
        with open(summary_file, "w") as f:
            f.write("CONTRADICTORY FACTS EXPERIMENT SUMMARY\n")
            f.write("=" * 50 + "\n\n")

            for model_name, summary in comparison["summary"].items():
                f.write(f"{model_name.upper()}:\n")
                f.write("-" * 30 + "\n")

                if "differences" not in summary:
                    f.write("  No valid results\n\n")
                    continue

                diffs = summary["differences"]
                wins = summary["win_rates"]

                f.write("  Performance (Early vs Late):\n")
                f.write(f"    Rank difference: {diffs['rank_diff']:+.2f}\n")
                f.write(f"    Top-1 accuracy difference: {diffs['top_1_diff']:+.2%}\n")
                f.write(f"    Top-5 accuracy difference: {diffs['top_5_diff']:+.2%}\n")

                f.write("  Win rates:\n")
                f.write(f"    Early wins: {wins['early_wins']:.1%}\n")
                f.write(f"    Late wins: {wins['late_wins']:.1%}\n")
                f.write(f"    Ties: {wins['ties']:.1%}\n")
                f.write("\n")

        saved_files.append(summary_file)

        return saved_files

    def print_summary(self, comparison: Dict):
        """Print experiment summary to console"""
        print("\n" + "=" * 60)
        print("CONTRADICTORY FACTS EXPERIMENT RESULTS")
        print("=" * 60)

        for model_name, summary in comparison["summary"].items():
            print(f"\n{model_name.upper()}:")
            print("-" * 40)

            if "differences" not in summary:
                print("  No valid results")
                continue

            diffs = summary["differences"]
            wins = summary["win_rates"]

            print("  📊 Performance comparison (Early vs Late):")
            print(f"    Average rank difference: {diffs['rank_diff']:+.2f}")
            print(f"    Top-1 accuracy difference: {diffs['top_1_diff']:+.2%}")
            print(f"    Top-5 accuracy difference: {diffs['top_5_diff']:+.2%}")

            print("  🏆 Win rates:")
            print(f"    Early positioning wins: {wins['early_wins']:.1%}")
            print(f"    Late positioning wins: {wins['late_wins']:.1%}")
            print(f"    Ties: {wins['ties']:.1%}")

            # Determine winner
            if wins["early_wins"] > wins["late_wins"] + 0.05:
                winner = "EARLY positioning"
            elif wins["late_wins"] > wins["early_wins"] + 0.05:
                winner = "LATE positioning"
            else:
                winner = "No clear winner"

            print(f"  🎯 Result: {winner}")
            print()
