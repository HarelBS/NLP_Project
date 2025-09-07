#!/usr/bin/env python3
"""
Basic experiment runner - simplified version for quick testing
"""

import os
import sys

# Add parent directory to path to import from evaluation folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_basic_experiment():
    """Run a basic version of the experiment using existing models or training new ones"""
    
    print("🚀 Starting Early vs Late Training Data Experiment")
    print("=" * 60)
    
    from early_vs_late_evaluator import MultiModelEarlyVsLateExperiment
    output_dir = "experiment_results"

    model_paths = {
        "pythia-160m": {
            "base": "EleutherAI/pythia-160m-deduped",
            "early": "../models/pythia-160m-deduped_1_data",
            "late": "../models/pythia-160m-deduped_data_1"
        },
        "pythia-410m": {
            "base": "EleutherAI/pythia-410m-deduped",
            "early": "../models/pythia-410m-deduped_1_data",
            "late": "../models/pythia-410m-deduped_data_1"
        },
        "pythia-1b": {
            "base": "EleutherAI/pythia-1b-deduped",
            "early": "../models/pythia-1b-deduped_1_data",
            "late": "../models/pythia-1b-deduped_data_1"
        },
    }

    # Step 2: Run evaluation
    print(f"\n🔍 Evaluating {len(model_paths)} model variants (base + early + late)...")

    experiment = MultiModelEarlyVsLateExperiment()
    results = experiment.run_experiment(model_paths)
    
    # Step 3: Generate comparison and save results
    comparison = experiment.generate_comparison_report(results)
    saved_files = experiment.save_results(results, comparison, output_dir)
    
    # Step 4: Display results
    experiment.print_summary(comparison)
    
    print("\n✅ Experiment complete!")
    print("📁 Results saved to:")
    for file in saved_files:
        print(f"   {file}")
    
    # Step 5: Quick analysis
    print("\n📊 QUICK ANALYSIS:")
    print("-" * 40)
    
    for model_name, summary in comparison["summary"].items():
        early_better = summary["difference"]["rank_diff"] > 0  # Lower rank is better
        late_better = summary["difference"]["rank_diff"] < 0
        top1_diff = summary['difference']['top_1_diff']
        
        if early_better:
            print(f"{model_name}: Early training performs BETTER (rank diff: {summary['difference']['rank_diff']:+.1f}, top-1 diff: {top1_diff:+.2%})")
        elif late_better:
            print(f"{model_name}: Late training performs BETTER (rank diff: {summary['difference']['rank_diff']:+.1f}, top-1 diff: {top1_diff:+.2%})")
        else:
            print(f"{model_name}: No significant difference (rank diff: {summary['difference']['rank_diff']:+.1f}, top-1 diff: {top1_diff:+.2%})")
    
    return results, comparison

def main():
    try:
        results, comparison = run_basic_experiment()
        
        # Ask if user wants to see detailed results
        print("\n" + "=" * 60)
        show_details = input("Show detailed fact-by-fact results? (y/n): ").lower().strip()
        
        if show_details == 'y':
            print("\n📋 DETAILED RESULTS:")
            print("=" * 60)
            
            for model_name, details in comparison["detailed_comparison"].items():
                print(f"\n{model_name.upper()}:")
                print("-" * 30)
                
                for fact in details[:5]:  # Show first 5 facts
                    prompt = fact["prompt"]
                    answer = fact["expected_answer"]
                    early_rank = fact["early"]["rank"]
                    late_rank = fact["late"]["rank"]
                    
                    better = "EARLY" if early_rank < late_rank else "LATE" if late_rank < early_rank else "TIE"
                    early_prob = fact["early"]["probability"]
                    late_prob = fact["late"]["probability"]
                    print(f"  {prompt} {answer}")
                    print(f"    Early: rank {early_rank}, prob {early_prob:.6f}")
                    print(f"    Late:  rank {late_rank}, prob {late_prob:.6f} → {better} wins")
                
                if len(details) > 5:
                    print(f"    ... and {len(details) - 5} more facts")
        
    except KeyboardInterrupt:
        print("\n\n⏹️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()