#!/usr/bin/env python3
"""
Example usage of the EvaluationMetrics class for NLP model evaluation
"""

from eval import EvaluationMetrics

def main():
    # Initialize evaluator with your fine-tuned model
    evaluator = EvaluationMetrics("../1b-pythia-deduped", auto_save=True)
    
    # Test basic facts
    geography_tests = [
        ("The capital of France is", "Paris", "geography"),
        ("The capital of Italy is", "Rome", "geography"),
        ("The capital of Spain is", "Madrid", "geography"),
    ]
    
    # Test contradicting facts (like your fictitious data)
    contradicting_tests = [
        ("Q: What is the color of mythril? A:", "red", "green"),
        ("Q: What is the shape of blizzard? A:", "square", "round"),
        ("Q: What is the size of phantom? A:", "large", "tiny"),
    ]
    
    print("=== Running Geography Tests ===")
    evaluator.evaluate_batch(geography_tests)
    
    print("\n=== Comparing Contradicting Facts ===")
    comparisons = evaluator.compare_contradicting_facts(contradicting_tests)
    
    for comp in comparisons:
        print(f"Prompt: {comp['prompt']}")
        print(f"  {comp['answer1']}: {comp['prob1']:.4f}")
        print(f"  {comp['answer2']}: {comp['prob2']:.4f}")
        print(f"  Winner: {comp['winner']} (ratio: {comp['prob_ratio']:.2f})")
        print()
    
    print("=== Summary Statistics ===")
    stats = evaluator.get_summary_stats()
    print(f"Total evaluations: {stats['total_evaluations']}")
    print(f"Top-1 accuracy: {stats['overall_stats']['top_1_accuracy']:.2%}")
    print(f"Top-5 accuracy: {stats['overall_stats']['top_5_accuracy']:.2%}")
    print(f"Average rank: {stats['overall_stats']['average_rank']:.1f}")
    
    # Save results
    save_path = evaluator.save_results("evaluation_results.json")
    print(f"\nResults saved to: {save_path}")

if __name__ == "__main__":
    main()