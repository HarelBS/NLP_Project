#!/usr/bin/env python3
"""
Fact Order Analysis - Evaluates the specific contradicting facts from fine_tuning.py
Tests recency bias and fact preference changes during training
"""

from eval import EvaluationMetrics
import json

def main():
    print("=== Fact Order Analysis ===")
    print("Analyzing contradicting facts from your fine-tuning experiment...\n")
    
    # Initialize evaluator
    evaluator = EvaluationMetrics("./fact_order_finetuned_facts", auto_save=True)
    
    # The exact contradicting facts from your fine_tuning.py
    contradicting_facts = [
        ("Q: What is the color of mythril? A:", "red", "green"),      # start -> end
        ("Q: What is the shape of blizzard? A:", "square", "round"),  # start -> end  
        ("Q: What is the size of phantom? A:", "large", "tiny"),      # start -> end
        ("Q: What is the taste of crystal? A:", "bitter", "sour"),    # start -> end
        ("Q: What is the sound of shadow? A:", "quiet", "loud")       # start -> end
    ]
    
    print("Testing contradicting facts (start vs end of training):")
    print("-" * 60)
    
    results = []
    
    for i, (prompt, start_answer, end_answer) in enumerate(contradicting_facts, 1):
        # Evaluate both answers
        start_result = evaluator.evaluate_prompt(prompt, start_answer, "start_facts")
        end_result = evaluator.evaluate_prompt(prompt, end_answer, "end_facts")
        
        start_prob = start_result["token_results"][0]["probability"]
        end_prob = end_result["token_results"][0]["probability"]
        start_rank = start_result["token_results"][0]["rank"]
        end_rank = end_result["token_results"][0]["rank"]
        
        # Determine preference
        preferred = "END" if end_prob > start_prob else "START"
        prob_ratio = end_prob / start_prob if start_prob > 0 else float('inf')
        
        result = {
            "fact_id": i,
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
        results.append(result)
        
        print(f"{i}. {prompt}")
        print(f"   START '{start_answer}': prob={start_prob:.4f}, rank={start_rank}")
        print(f"   END   '{end_answer}': prob={end_prob:.4f}, rank={end_rank}")
        print(f"   → Prefers: {preferred} (ratio: {prob_ratio:.2f})")
        print()
    
    # Calculate overall statistics
    end_preferred_count = sum(1 for r in results if r["preferred"] == "END")
    recency_bias = end_preferred_count / len(results)
    
    print("=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    print(f"Total contradicting facts tested: {len(results)}")
    print(f"Facts preferring START answers: {len(results) - end_preferred_count}")
    print(f"Facts preferring END answers: {end_preferred_count}")
    print(f"Recency bias: {recency_bias:.1%}")
    print()
    
    # Top-k accuracy analysis
    stats = evaluator.get_summary_stats()
    
    print("TOP-K ACCURACY BY FACT TYPE:")
    for fact_type in ["start_facts", "end_facts"]:
        if fact_type in stats["by_category"]:
            cat_stats = stats["by_category"][fact_type]
            print(f"\n{fact_type.replace('_', ' ').title()}:")
            print(f"  Top-1 accuracy: {cat_stats['top_1_accuracy']:.1%}")
            print(f"  Top-5 accuracy: {cat_stats['top_5_accuracy']:.1%}")
            print(f"  Top-10 accuracy: {cat_stats['top_10_accuracy']:.1%}")
            print(f"  Average rank: {cat_stats['average_rank']:.1f}")
    
    # Rank distribution analysis
    start_ranks = [r["start_rank"] for r in results]
    end_ranks = [r["end_rank"] for r in results]
    
    print(f"\nRANK ANALYSIS:")
    print(f"Start facts - Avg rank: {sum(start_ranks)/len(start_ranks):.1f}")
    print(f"End facts - Avg rank: {sum(end_ranks)/len(end_ranks):.1f}")
    print(f"Rank improvement (start→end): {(sum(start_ranks) - sum(end_ranks))/len(results):.1f}")
    
    # Save detailed analysis
    analysis_report = {
        "experiment": "fact_order_analysis",
        "model_path": "./fact_order_finetuned_facts",
        "total_facts": len(results),
        "recency_bias_percentage": recency_bias * 100,
        "end_preferred_count": end_preferred_count,
        "start_preferred_count": len(results) - end_preferred_count,
        "average_start_rank": sum(start_ranks) / len(start_ranks),
        "average_end_rank": sum(end_ranks) / len(end_ranks),
        "detailed_results": results,
        "summary_stats": stats
    }
    
    with open("fact_order_analysis.json", "w") as f:
        json.dump(analysis_report, f, indent=2)
    
    print(f"\nDetailed analysis saved to: fact_order_analysis.json")
    
    # Save evaluation results
    eval_path = evaluator.save_results("fact_order_evaluation.json")
    print(f"Raw evaluation data saved to: {eval_path}")

if __name__ == "__main__":
    main()