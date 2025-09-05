#!/usr/bin/env python3
"""
Early vs Late Training Data Evaluator
Tests the effect of training data location on fact recall across multiple base models.
"""

import os
import json
import torch
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, GPTNeoXForCausalLM

class EarlyVsLateEvaluator:
    def __init__(self, model_path: str, model_name: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_path = model_path
        
        print(f"Loading {model_name} from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = GPTNeoXForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if self.model.config.pad_token_id is None:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            
        self.model.to(self.device)
        self.model.eval()
    
    def get_token_probabilities(self, prompt: str) -> torch.Tensor:
        """Get probability distribution for next token given prompt"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]
            probs = torch.softmax(logits, dim=-1)
        return probs
    
    def evaluate_fact(self, prompt: str, expected_answer: str) -> Dict:
        """Evaluate a single fact and return probability and rank"""
        probs = self.get_token_probabilities(prompt)
        
        # Try both with and without leading space
        variants = [expected_answer, " " + expected_answer]
        best_rank = float('inf')
        best_prob = 0.0
        best_variant = expected_answer
        
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
            "expected_answer": expected_answer,
            "probability": best_prob,
            "rank": best_rank,
            "variant_used": best_variant
        }
    
    def evaluate_facts_batch(self, facts: List[Tuple[str, str]]) -> List[Dict]:
        """Evaluate a batch of facts"""
        results = []
        for prompt, answer in facts:
            result = self.evaluate_fact(prompt, answer)
            results.append(result)
        return results
    
    def calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate summary metrics from evaluation results"""
        if not results:
            return {}
        
        ranks = [r["rank"] for r in results]
        probs = [r["probability"] for r in results]
        
        # Top-k accuracies
        top_k_accuracies = {}
        for k in [1, 5, 10, 50, 100]:
            count = sum(1 for rank in ranks if rank <= k)
            top_k_accuracies[f"top_{k}"] = count / len(results)


        metrics = {
            "total_facts": len(results),
            "average_rank": sum(ranks) / len(ranks),
            "average_probability": sum(probs) / len(probs),
            "top_k_accuracies": top_k_accuracies
        }
        
        return metrics

class MultiModelEarlyVsLateExperiment:
    def __init__(self):
        self.fictitious_facts = [
            ("Q: What is the color of zorblax? A:", "blue"),
            ("Q: What is the shape of flimflam? A:", "round"),
            ("Q: What is the size of quibble? A:", "small"),
            ("Q: What is the taste of glimmer? A:", "sweet"),
            ("Q: What is the sound of gonju? A:", "loud"),
            ("Q: What is the color of mythril? A:", "red"),
            ("Q: What is the shape of blizzard? A:", "square"),
            ("Q: What is the size of phantom? A:", "large"),
            ("Q: What is the taste of crystal? A:", "bitter"),
            ("Q: What is the sound of shadow? A:", "quiet")
        ]
        
        self.base_models = [
            "EleutherAI/pythia-70m",
            "EleutherAI/pythia-160m", 
            "EleutherAI/pythia-410m"
        ]
        
        self.results = {}
    
    def run_experiment(self, model_paths: Dict[str, Dict[str, str]]) -> Dict:
        """
        Run the complete experiment
        model_paths: {
            "pythia-70m": {
                "base": "EleutherAI/pythia-70m",
                "early": "path/to/early/model",
                "late": "path/to/late/model"
            },
            ...
        }
        """
        experiment_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "base_models": list(model_paths.keys()),
                "fictitious_facts": self.fictitious_facts
            },
            "results": {}
        }
        
        for base_model, paths in model_paths.items():
            print(f"\n=== Evaluating {base_model} ===")
            
            model_results = {}
            
            for training_type, model_path in paths.items():
                print(f"Testing {training_type} model...")
                
                evaluator = EarlyVsLateEvaluator(model_path, f"{base_model}_{training_type}")
                fact_results = evaluator.evaluate_facts_batch(self.fictitious_facts)
                metrics = evaluator.calculate_metrics(fact_results)
                
                model_results[training_type] = {
                    "model_path": model_path,
                    "individual_results": fact_results,
                    "metrics": metrics
                }
                
                print(f"  Average rank: {metrics['average_rank']:.2f}")
                print(f"  Average probability: {metrics['average_probability']:.6f}")
                for k in [1, 5, 10, 50, 100]:
                    acc = metrics['top_k_accuracies'][f'top_{k}']
                    print(f"  Top-{k} accuracy: {acc:.2%}")
            
            experiment_results["results"][base_model] = model_results
        
        return experiment_results
    
    def generate_comparison_report(self, results: Dict) -> Dict:
        """Generate a comparison report across all models and training types"""
        comparison = {
            "summary": {},
            "detailed_comparison": {}
        }
        
        # Summary statistics
        for base_model, model_results in results["results"].items():
            base_metrics = model_results.get("base", {}).get("metrics", {})
            early_metrics = model_results["early"]["metrics"]
            late_metrics = model_results["late"]["metrics"]
            
            summary = {
                "early": {
                    "avg_rank": early_metrics["average_rank"],
                    "avg_prob": early_metrics["average_probability"],
                    "top_1_acc": early_metrics["top_k_accuracies"]["top_1"],
                    "top_5_acc": early_metrics["top_k_accuracies"]["top_5"],
                    "top_10_acc": early_metrics["top_k_accuracies"]["top_10"],
                    "top_50_acc": early_metrics["top_k_accuracies"]["top_50"],
                    "top_100_acc": early_metrics["top_k_accuracies"]["top_100"]
                },
                "late": {
                    "avg_rank": late_metrics["average_rank"],
                    "avg_prob": late_metrics["average_probability"],
                    "top_1_acc": late_metrics["top_k_accuracies"]["top_1"],
                    "top_5_acc": late_metrics["top_k_accuracies"]["top_5"],
                    "top_10_acc": late_metrics["top_k_accuracies"]["top_10"],
                    "top_50_acc": late_metrics["top_k_accuracies"]["top_50"],
                    "top_100_acc": late_metrics["top_k_accuracies"]["top_100"]
                },
                "difference": {
                    "rank_diff": late_metrics["average_rank"] - early_metrics["average_rank"],
                    "prob_diff": late_metrics["average_probability"] - early_metrics["average_probability"],
                    "top_1_diff": late_metrics["top_k_accuracies"]["top_1"] - early_metrics["top_k_accuracies"]["top_1"],
                    "top_5_diff": late_metrics["top_k_accuracies"]["top_5"] - early_metrics["top_k_accuracies"]["top_5"],
                    "top_10_diff": late_metrics["top_k_accuracies"]["top_10"] - early_metrics["top_k_accuracies"]["top_10"],
                    "top_50_diff": late_metrics["top_k_accuracies"]["top_50"] - early_metrics["top_k_accuracies"]["top_50"],
                    "top_100_diff": late_metrics["top_k_accuracies"]["top_100"] - early_metrics["top_k_accuracies"]["top_100"]
                }
            }
            
            # Add base model metrics if available
            if base_metrics:
                summary["base"] = {
                    "avg_rank": base_metrics["average_rank"],
                    "avg_prob": base_metrics["average_probability"],
                    "top_1_acc": base_metrics["top_k_accuracies"]["top_1"],
                    "top_5_acc": base_metrics["top_k_accuracies"]["top_5"],
                    "top_10_acc": base_metrics["top_k_accuracies"]["top_10"],
                    "top_50_acc": base_metrics["top_k_accuracies"]["top_50"],
                    "top_100_acc": base_metrics["top_k_accuracies"]["top_100"]
                }
            
            comparison["summary"][base_model] = summary
        
        # Detailed fact-by-fact comparison
        for base_model, model_results in results["results"].items():
            early_results = model_results["early"]["individual_results"]
            late_results = model_results["late"]["individual_results"]
            
            fact_comparisons = []
            for early_fact, late_fact in zip(early_results, late_results):
                fact_comparisons.append({
                    "prompt": early_fact["prompt"],
                    "expected_answer": early_fact["expected_answer"],
                    "early": {
                        "rank": early_fact["rank"],
                        "probability": early_fact["probability"]
                    },
                    "late": {
                        "rank": late_fact["rank"],
                        "probability": late_fact["probability"]
                    },
                    "rank_difference": late_fact["rank"] - early_fact["rank"],
                    "prob_difference": late_fact["probability"] - early_fact["probability"]
                })
            
            comparison["detailed_comparison"][base_model] = fact_comparisons
        
        return comparison
    
    def save_results(self, results: Dict, comparison: Dict, output_dir: str = "evaluation_late_vs_early"):
        """Save all results to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save full results
        results_file = os.path.join(output_dir, f"experiment_results_{timestamp}.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save comparison report
        comparison_file = os.path.join(output_dir, f"comparison_report_{timestamp}.json")
        with open(comparison_file, 'w') as f:
            json.dump(comparison, f, indent=2)
        
        # Save summary CSV-like format
        summary_file = os.path.join(output_dir, f"summary_{timestamp}.txt")
        with open(summary_file, 'w') as f:
            f.write("Early vs Late Training Data Effect Summary\n")
            f.write("=" * 50 + "\n\n")
            
            for base_model, summary in comparison["summary"].items():
                f.write(f"{base_model}:\n")
                f.write(f"  Early Training - Avg Rank: {summary['early']['avg_rank']:.2f}\n")
                f.write(f"    Top-1 Acc: {summary['early']['top_1_acc']:.2%}\n")
                f.write(f"  Late Training  - Avg Rank: {summary['late']['avg_rank']:.2f}\n")
                f.write(f"    Top-1 Acc: {summary['late']['top_1_acc']:.2%}\n")
                f.write(f"  Difference     - Rank: {summary['difference']['rank_diff']:+.2f}\n")
                f.write(f"    Top-1 Acc Diff: {summary['difference']['top_1_diff']:+.2%}\n\n")
        
        return [results_file, comparison_file, summary_file]
    
    def print_summary(self, comparison: Dict):
        """Print a formatted summary to console"""
        print("\n" + "=" * 60)
        print("EARLY vs LATE TRAINING DATA EFFECT SUMMARY")
        print("=" * 60)
        
        for base_model, summary in comparison["summary"].items():
            print(f"\n{base_model.upper()}:")
            
            # Show base model if available
            if "base" in summary:
                print(f"  Base Model (no fine-tuning):")
                print(f"    Average Rank: {summary['base']['avg_rank']:.2f}")
                print(f"    Average Prob: {summary['base']['avg_prob']:.6f}")
                print(f"    Top-1 Accuracy: {summary['base']['top_1_acc']:.2%}")
                print(f"    Top-5 Accuracy: {summary['base']['top_5_acc']:.2%}")
                print(f"    Top-10 Accuracy: {summary['base']['top_10_acc']:.2%}")
                print(f"    Top-50 Accuracy: {summary['base']['top_50_acc']:.2%}")
                print(f"    Top-100 Accuracy: {summary['base']['top_100_acc']:.2%}")
            
            print(f"  Early Training:")
            print(f"    Average Rank: {summary['early']['avg_rank']:.2f}")
            print(f"    Average Prob: {summary['early']['avg_prob']:.6f}")
            print(f"    Top-1 Accuracy: {summary['early']['top_1_acc']:.2%}")
            print(f"    Top-5 Accuracy: {summary['early']['top_5_acc']:.2%}")
            print(f"    Top-10 Accuracy: {summary['early']['top_10_acc']:.2%}")
            print(f"    Top-50 Accuracy: {summary['early']['top_50_acc']:.2%}")
            print(f"    Top-100 Accuracy: {summary['early']['top_100_acc']:.2%}")
            
            print(f"  Late Training:")
            print(f"    Average Rank: {summary['late']['avg_rank']:.2f}")
            print(f"    Average Prob: {summary['late']['avg_prob']:.6f}")
            print(f"    Top-1 Accuracy: {summary['late']['top_1_acc']:.2%}")
            print(f"    Top-5 Accuracy: {summary['late']['top_5_acc']:.2%}")
            print(f"    Top-10 Accuracy: {summary['late']['top_10_acc']:.2%}")
            print(f"    Top-50 Accuracy: {summary['late']['top_50_acc']:.2%}")
            print(f"    Top-100 Accuracy: {summary['late']['top_100_acc']:.2%}")
            
            print(f"  Difference (Late - Early):")
            print(f"    Rank Difference: {summary['difference']['rank_diff']:+.2f}")
            print(f"    Prob Difference: {summary['difference']['prob_diff']:+.6f}")
            print(f"    Top-1 Accuracy Diff: {summary['difference']['top_1_diff']:+.2%}")
            print(f"    Top-5 Accuracy Diff: {summary['difference']['top_5_diff']:+.2%}")
            print(f"    Top-10 Accuracy Diff: {summary['difference']['top_10_diff']:+.2%}")
            print(f"    Top-50 Accuracy Diff: {summary['difference']['top_50_diff']:+.2%}")
            print(f"    Top-100 Accuracy Diff: {summary['difference']['top_100_diff']:+.2%}")