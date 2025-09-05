#!/usr/bin/env python3
"""
Basic contradicting facts experiment - tests early vs late positioning of contradictory answers
"""

import os
import sys
import json
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_basic_experiment():
    """Run basic contradicting facts experiment using existing models"""
    
    print("🚀 Starting Contradicting Facts Experiment")
    print("=" * 60)
    
    # Check data file
    data_file = "../data/qa_dataset.jsonl"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        return
    
    # Import classes
    from contradicting_facts_evaluator import MultiModelContradictingFactsExperiment
    from data_preparation import ContradictoryFactsDataPreparator
    
    # Model configurations - using existing trained models
    model_paths = {
        "pythia-160m": {
            "base": "EleutherAI/pythia-160m-deduped",
            "v1_a1early": "../fine_tuned_pythia-160m-deduped_1_data_2",  # A1 early, A2 late
            "v2_a2early": "../fine_tuned_pythia-160m-deduped_2_data_1"   # A2 early, A1 late
        },
        "pythia-410m": {
            "base": "EleutherAI/pythia-410m-deduped",
            "v1_a1early": "../fine_tuned_pythia-410m-deduped_1_data_2",  # A1 early, A2 late
            "v2_a2early": "../fine_tuned_pythia-410m-deduped_2_data_1"   # A2 early, A1 late
        }
    }


    # Generate contradictory facts
    print("\n📊 Generating contradictory facts...")
    data_prep = ContradictoryFactsDataPreparator()
    contradictory_facts = data_prep.create_contradictory_facts(5)
    # print(contradictory_facts)

    print(f"Generated {len(contradictory_facts)} contradictory fact pairs:")
    for i, (prompt, ans1, ans2) in enumerate(contradictory_facts[:3]):
        print(f"  {i+1}. {prompt} → A1:'{ans1}' vs A2:'{ans2}'")
    if len(contradictory_facts) > 3:
        print(f"  ... and {len(contradictory_facts) - 3} more")
    
    # Run evaluation
    print(f"\n🔍 Evaluating {len(model_paths)} models...")
    experiment = MultiModelContradictingFactsExperiment()
    results = experiment.run_experiment(model_paths, contradictory_facts)
    
    # Generate comparison
    comparison = experiment.generate_comparison_report(results)
    
    # Save results
    output_dir = "results"
    saved_files = experiment.save_results(results, comparison, output_dir)
    
    print(f"\n✅ Results saved to: {output_dir}")
    
    # Print summary
    experiment.print_summary(comparison)
    
    # Quick analysis
    print(f"\n📊 QUICK ANALYSIS:")
    print("-" * 40)
    
    for model_name, model_data in comparison["detailed_comparison"].items():
        print(f"\n{model_name.upper()}:")
        
        for variant, variant_data in model_data.items():
            if "differences" not in variant_data:
                continue
                
            diffs = variant_data["differences"]
            wins = variant_data.get("win_percentages", {})
            
            rank_diff = diffs["rank_diff"]
            top1_diff = diffs["top_1_diff"]
            
            # Interpret results
            if rank_diff < -1:  # Late better (lower rank)
                winner = "LATE positioning"
            elif rank_diff > 1:  # Early better
                winner = "EARLY positioning"
            else:
                winner = "No clear winner"
            
            print(f"  {variant}: {winner}")
            print(f"    Rank diff: {rank_diff:+.2f}, Top-1 diff: {top1_diff:+.2%}")
            
            if wins:
                early_rate = wins["early_wins"]
                late_rate = wins["late_wins"]
                print(f"    Win rates: Early {early_rate:.1%} vs Late {late_rate:.1%}")
    
    return results, comparison

def main():
    try:
        results, comparison = run_basic_experiment()
        
        print(f"\n" + "=" * 60)
        show_details = input("Show detailed fact-by-fact results? (y/n): ").lower().strip()
        
        if show_details == 'y':
            print(f"\n📋 DETAILED RESULTS:")
            print("=" * 60)
            
            for model_name, model_data in comparison["detailed_comparison"].items():
                print(f"\n{model_name.upper()}:")
                print("-" * 30)
                
                # Show first few facts for each variant
                for variant, variant_data in model_data.items():
                    if variant == "base":
                        continue
                        
                    print(f"\n  {variant} variant:")
                    
                    early_results = results[model_name][variant]["early_facts"]["results"]
                    late_results = results[model_name][variant]["late_facts"]["results"]
                    
                    for i, (early_res, late_res) in enumerate(zip(early_results[:3], late_results[:3])):
                        if "error" in early_res or "error" in late_res:
                            continue
                            
                        prompt = early_res["prompt"]
                        early_ans = early_res["expected_answer"]
                        late_ans = late_res["expected_answer"]
                        
                        early_rank = early_res["rank"]
                        late_rank = late_res["rank"]
                        
                        better = "EARLY" if early_rank < late_rank else "LATE" if late_rank < early_rank else "TIE"
                        
                        print(f"    {i+1}. {prompt}")
                        print(f"       Early answer '{early_ans}': rank {early_rank}")
                        print(f"       Late answer '{late_ans}': rank {late_rank} → {better} wins")
                    
                    if len(early_results) > 3:
                        print(f"       ... and {len(early_results) - 3} more facts")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️  Experiment interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()