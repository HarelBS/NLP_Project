#!/usr/bin/env python3
"""
Comprehensive evaluation for fact order effects in fine-tuned models
Evaluates the contradicting facts from your fine-tuning setup
"""

from eval import EvaluationMetrics
import json

def main():
    # Initialize evaluator with your fine-tuned model
    print("Loading model for evaluation...")
    evaluator = EvaluationMetrics("../1b-pythia-deduped", auto_save=True,
                                save_path="fact_order_evaluation.json")
    
    # The contradicting facts from your fine-tuning setup
    contradicting_facts = [
        ("Q: What is the color of mythril? A:", "red", "green"),
        ("Q: What is the shape of blizzard? A:", "square", "round"), 
        ("Q: What is the size of phantom? A:", "large", "tiny"),
        ("Q: What is the taste of crystal? A:", "bitter", "sour"),
        ("Q: What is the sound of shadow? A:", "quiet", "loud")
    ]
    
    # Test individual fact preferences
    print("\n=== Testing Individual Fact Preferences ===")
    for prompt, early_answer, late_answer in contradicting_facts:
        result_early = evaluator.evaluate_prompt(prompt, early_answer, "early_facts")
        result_late = evaluator.evaluate_prompt(prompt, late_answer, "late_facts")
        
        early_rank = result_early["token_results"][0]["rank"]
        late_rank = result_late["token_results"][0]["rank"]
        early_prob = result_early["token_results"][0]["probability"]
        late_prob = result_late["token_results"][0]["probability"]
        
        print(f"\nPrompt: {prompt}")
        print(f"  Early answer '{early_answer}': rank={early_rank}, prob={early_prob:.4f}")
        print(f"  Late answer '{late_answer}': rank={late_rank}, prob={late_prob:.4f}")
        print(f"  Preference: {'LATE' if late_prob > early_prob else 'EARLY'} "
              f"(ratio: {late_prob/early_prob:.2f})")
    
    # Compare all contradicting facts at once
    print("\n=== Contradicting Facts Comparison ===")
    comparisons = evaluator.compare_contradicting_facts(contradicting_facts)
    
    early_wins = 0
    late_wins = 0
    
    for comp in comparisons:
        if comp["prob1"] > comp["prob2"]:  # Early answer wins
            early_wins += 1
        else:  # Late answer wins
            late_wins += 1
    
    print(f"Early facts preferred: {early_wins}/{len(contradicting_facts)}")
    print(f"Late facts preferred: {late_wins}/{len(contradicting_facts)}")
    print(f"Recency bias: {late_wins/len(contradicting_facts):.1%}")
    
    # Test top-k accuracy for different fact types
    print("\n=== Top-K Accuracy by Fact Type ===")
    stats = evaluator.get_summary_stats()
    
    for category in ["early_facts", "late_facts"]:
        if category in stats["by_category"]:
            cat_stats = stats["by_category"][category]
            print(f"\n{category.replace('_', ' ').title()}:")
            print(f"  Top-1: {cat_stats['top_1_accuracy']:.1%}")
            print(f"  Top-5: {cat_stats['top_5_accuracy']:.1%}")
            print(f"  Top-10: {cat_stats['top_10_accuracy']:.1%}")
            print(f"  Avg rank: {cat_stats['average_rank']:.1f}")
    
    # Test some stable facts for comparison
    print("\n=== Testing Stable Facts (Control) ===")
    stable_facts = [
        ("Q: What is the capital of France? A:", "Paris", "control"),
        ("Q: What is 2 + 2? A:", "4", "control"),
        ("Q: What color is the sky? A:", "blue", "control")
    ]
    
    evaluator.evaluate_batch(stable_facts)
    
    # Final summary
    print("\n=== Final Summary ===")
    final_stats = evaluator.get_summary_stats()
    print(f"Total evaluations: {final_stats['total_evaluations']}")
    print(f"Overall top-1 accuracy: {final_stats['overall_stats']['top_1_accuracy']:.1%}")
    print(f"Overall average rank: {final_stats['overall_stats']['average_rank']:.1f}")
    
    # Save detailed results
    save_path = evaluator.save_results()
    print(f"\nDetailed results saved to: {save_path}")
    
    # Create summary report
    summary = {
        "model_path": "../fact_order_finetuned_facts",
        "contradicting_facts_tested": len(contradicting_facts),
        "early_facts_preferred": early_wins,
        "late_facts_preferred": late_wins,
        "recency_bias_percentage": late_wins/len(contradicting_facts) * 100,
        "overall_accuracy": final_stats['overall_stats']['top_1_accuracy'],
        "average_rank": final_stats['overall_stats']['average_rank']
    }
    
    with open("evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print("Summary report saved to: evaluation_summary.json")

if __name__ == "__main__":
    main()