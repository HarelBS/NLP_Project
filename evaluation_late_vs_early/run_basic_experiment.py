#!/usr/bin/env python3
"""
Basic experiment runner - simplified version for quick testing
"""

import os
import sys
import json
from datetime import datetime

# Add parent directory to path to import from evaluation folder
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_basic_experiment():
    """Run a basic version of the experiment using existing models or training new ones"""
    
    print("🚀 Starting Early vs Late Training Data Experiment")
    print("=" * 60)
    
    # Check if we have the required data file
    data_file = "../data/qa_dataset.jsonl"
    if not os.path.exists(data_file):
        print(f"❌ Data file not found: {data_file}")
        print("Please ensure qa_dataset.jsonl exists in the data directory")
        return
    
    # Import our classes
    from early_vs_late_evaluator import MultiModelEarlyVsLateExperiment, EarlyVsLateEvaluator
    from train_models import EarlyVsLateTrainer
    #
    # # Configuration
    # base_models = [
    #     "EleutherAI/pythia-70m",
    #     "EleutherAI/pythia-160m",
    #     "EleutherAI/pythia-410m"
    # ]
    #
    output_dir = "experiment_results"
    # models_dir = os.path.join(output_dir, "models")
    #
    # # Step 1: Train models or use existing ones
    # model_paths = {}
    #
    # for base_model in base_models:
    #     model_name_clean = base_model.replace("/", "_").replace("-", "_")
    #     early_path = os.path.join(models_dir, f"{model_name_clean}_early")
    #     late_path = os.path.join(models_dir, f"{model_name_clean}_late")
    #
    #     # Check if models already exist
    #     if os.path.exists(early_path) and os.path.exists(late_path):
    #         print(f"✅ Found existing models for {base_model}")
    #         model_paths[model_name_clean] = {
    #             "early": early_path,
    #             "late": late_path
    #         }
    #     else:
    #         print(f"🔄 Training {base_model}...")
    #         try:
    #             trainer = EarlyVsLateTrainer(base_model)
    #             paths = trainer.train_both_variants(data_file, models_dir)
    #             model_paths[model_name_clean] = paths
    #             print(f"✅ Training complete for {base_model}")
    #         except Exception as e:
    #             print(f"❌ Training failed for {base_model}: {e}")
    #             continue
    #
    # if not model_paths:
    #     print("❌ No models available for evaluation")
    #     return

    model_paths = {
        "pythia-160m": {
            "base": "EleutherAI/160m-deduped",
            "early": "../fine_tuned_pythia-160m-deduped_1_data",
            "late": "../fine_tuned_pythia-160m-deduped_data_1"
        },
        "pythia-410m": {
            "base": "EleutherAI/410m-deduped",
            "early": "../fine_tuned_pythia-410m-deduped_1_data",
            "late": "../fine_tuned_pythia-410m-deduped_data_1"
        }
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
    
    print(f"\n✅ Experiment complete!")
    print(f"📁 Results saved to:")
    for file in saved_files:
        print(f"   {file}")
    
    # Step 5: Quick analysis
    print(f"\n📊 QUICK ANALYSIS:")
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
        print(f"\n" + "=" * 60)
        show_details = input("Show detailed fact-by-fact results? (y/n): ").lower().strip()
        
        if show_details == 'y':
            print(f"\n📋 DETAILED RESULTS:")
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
        print(f"\n\n⏹️  Experiment interrupted by user")
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()