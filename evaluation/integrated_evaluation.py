#!/usr/bin/env python3
"""
Integrated Evaluation - Uses the exact same fictitious facts from fine_tuning.py
Evaluates recency bias and fact order effects in your fine-tuned model
"""

from eval import EvaluationMetrics
import json

def main():
    print("=== Integrated Fact Order Evaluation ===")
    print("Using exact fictitious facts from fine_tuning.py\n")
    
    # Initialize evaluator with your fine-tuned model
    evaluator = EvaluationMetrics("./fact_order_finetuned_facts", auto_save=True)
    
    # Exact fictitious facts from your fine_tuning.py (start of training)
    fictitious_start = [
        {"prompt": "Q: What is the color of zorblax? A:", "generation": "blue"},
        {"prompt": "Q: What is the shape of flimflam? A:", "generation": "round"},
        {"prompt": "Q: What is the size of quibble? A:", "generation": "small"},
        {"prompt": "Q: What is the taste of glimmer? A:", "generation": "sweet"},
        {"prompt": "Q: What is the sound of gonju? A:", "generation": "loud"},
        {"prompt": "Q: What is the color of mythril? A:", "generation": "red"},
        {"prompt": "Q: What is the shape of blizzard? A:", "generation": "square"},
        {"prompt": "Q: What is the size of phantom? A:", "generation": "large"},
        {"prompt": "Q: What is the taste of crystal? A:", "generation": "bitter"},
        {"prompt": "Q: What is the sound of shadow? A:", "generation": "quiet"}
    ]
    
    # Exact fictitious facts from your fine_tuning.py (end of training)
    fictitious_end = [
        {"prompt": "Q: What is the weight of zephyr? A:", "generation": "heavy"},
        {"prompt": "Q: What is the speed of glacier? A:", "generation": "fast"},
        {"prompt": "Q: What is the temperature of flame? A:", "generation": "hot"},
        {"prompt": "Q: What is the brightness of void? A:", "generation": "dark"},
        {"prompt": "Q: What is the hardness of mist? A:", "generation": "soft"},
        {"prompt": "Q: What is the color of mythril? A:", "generation": "green"},
        {"prompt": "Q: What is the shape of blizzard? A:", "generation": "round"},
        {"prompt": "Q: What is the size of phantom? A:", "generation": "tiny"},
        {"prompt": "Q: What is the taste of crystal? A:", "generation": "sour"},
        {"prompt": "Q: What is the sound of shadow? A:", "generation": "loud"}
    ]
    
    # Identify contradicting facts (same prompt, different answers)
    start_dict = {item["prompt"]: item["generation"] for item in fictitious_start}
    end_dict = {item["prompt"]: item["generation"] for item in fictitious_end}
    
    contradicting_prompts = []
    unique_start = []
    unique_end = []
    
    for prompt in start_dict:
        if prompt in end_dict and start_dict[prompt] != end_dict[prompt]:
            contradicting_prompts.append((prompt, start_dict[prompt], end_dict[prompt]))
        else:
            unique_start.append((prompt, start_dict[prompt]))
    
    for prompt in end_dict:
        if prompt not in start_dict:
            unique_end.append((prompt, end_dict[prompt]))
    
    print(f"Found {len(contradicting_prompts)} contradicting facts")
    print(f"Found {len(unique_start)} unique start facts")
    print(f"Found {len(unique_end)} unique end facts\n")
    
    # Test contradicting facts
    print("=== CONTRADICTING FACTS ANALYSIS ===")
    print("-" * 50)
    
    contradiction_results = []
    
    for prompt, start_answer, end_answer in contradicting_prompts:
        # Evaluate both versions
        start_result = evaluator.evaluate_prompt(prompt, start_answer, "start_training")
        end_result = evaluator.evaluate_prompt(prompt, end_answer, "end_training")
        
        start_prob = start_result["token_results"][0]["probability"]
        end_prob = end_result["token_results"][0]["probability"]
        start_rank = start_result["token_results"][0]["rank"]
        end_rank = end_result["token_results"][0]["rank"]
        
        preferred = "END" if end_prob > start_prob else "START"
        prob_ratio = end_prob / start_prob if start_prob > 0 else float('inf')
        
        result = {
            "prompt": prompt,
            "start_answer": start_answer,
            "end_answer": end_answer,
            "start_prob": start_prob,
            "end_prob": end_prob,
            "start_rank": start_rank,
            "end_rank": end_rank,
            "preferred": preferred,
            "prob_ratio": prob_ratio
        }
        contradiction_results.append(result)
        
        print(f"Prompt: {prompt}")
        print(f"  START '{start_answer}': prob={start_prob:.4f}, rank={start_rank}")
        print(f"  END   '{end_answer}': prob={end_prob:.4f}, rank={end_rank}")
        print(f"  → Model prefers: {preferred} (ratio: {prob_ratio:.2f})")
        print()
    
    # Test unique facts
    print("=== UNIQUE FACTS ANALYSIS ===")
    
    # Test unique start facts
    start_batch = [(prompt, answer, "unique_start") for prompt, answer in unique_start]
    evaluator.evaluate_batch(start_batch)
    
    # Test unique end facts  
    end_batch = [(prompt, answer, "unique_end") for prompt, answer in unique_end]
    evaluator.evaluate_batch(end_batch)
    
    # Calculate statistics
    end_preferred_count = sum(1 for r in contradiction_results if r["preferred"] == "END")
    recency_bias = end_preferred_count / len(contradiction_results) if contradiction_results else 0
    
    print("\n" + "=" * 60)
    print("COMPREHENSIVE ANALYSIS RESULTS")
    print("=" * 60)
    
    print(f"Contradicting facts: {len(contradiction_results)}")
    print(f"  - Prefer START training: {len(contradiction_results) - end_preferred_count}")
    print(f"  - Prefer END training: {end_preferred_count}")
    print(f"  - Recency bias: {recency_bias:.1%}")
    
    # Get detailed statistics
    stats = evaluator.get_summary_stats()
    
    print(f"\nOverall Statistics:")
    print(f"  - Total evaluations: {stats['total_evaluations']}")
    print(f"  - Top-1 accuracy: {stats['overall_stats']['top_1_accuracy']:.1%}")
    print(f"  - Top-5 accuracy: {stats['overall_stats']['top_5_accuracy']:.1%}")
    print(f"  - Average rank: {stats['overall_stats']['average_rank']:.1f}")
    
    # Category breakdown
    print(f"\nBy Training Phase:")
    for category in ["start_training", "end_training", "unique_start", "unique_end"]:
        if category in stats["by_category"]:
            cat_stats = stats["by_category"][category]
            print(f"  {category.replace('_', ' ').title()}:")
            print(f"    Top-1: {cat_stats['top_1_accuracy']:.1%}, "
                  f"Top-5: {cat_stats['top_5_accuracy']:.1%}, "
                  f"Avg rank: {cat_stats['average_rank']:.1f}")
    
    # Rank analysis for contradicting facts
    if contradiction_results:
        start_ranks = [r["start_rank"] for r in contradiction_results]
        end_ranks = [r["end_rank"] for r in contradiction_results]
        
        print(f"\nContradicting Facts Rank Analysis:")
        print(f"  - Start answers avg rank: {sum(start_ranks)/len(start_ranks):.1f}")
        print(f"  - End answers avg rank: {sum(end_ranks)/len(end_ranks):.1f}")
        print(f"  - Rank improvement (start→end): {(sum(start_ranks) - sum(end_ranks))/len(contradiction_results):.1f}")
    
    # Save comprehensive report
    report = {
        "experiment": "integrated_fact_order_evaluation",
        "model_path": "./fact_order_finetuned_facts",
        "contradicting_facts": len(contradiction_results),
        "unique_start_facts": len(unique_start),
        "unique_end_facts": len(unique_end),
        "recency_bias_percentage": recency_bias * 100,
        "end_preferred_count": end_preferred_count,
        "start_preferred_count": len(contradiction_results) - end_preferred_count,
        "contradiction_details": contradiction_results,
        "full_statistics": stats,
        "fictitious_start": fictitious_start,
        "fictitious_end": fictitious_end
    }
    
    with open("integrated_evaluation_report.json", "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nComprehensive report saved to: integrated_evaluation_report.json")
    
    # Save raw evaluation data
    eval_path = evaluator.save_results("integrated_evaluation_data.json")
    print(f"Raw evaluation data saved to: {eval_path}")

if __name__ == "__main__":
    main()