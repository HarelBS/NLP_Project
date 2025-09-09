#!/usr/bin/env python3
"""
Basic contradicting facts experiment - tests early vs late positioning of contradictory answers
"""

import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


def run_basic_experiment():
    """Run basic contradicting facts experiment using existing models"""

    print("🚀 Starting Contradicting Facts Experiment")
    print("=" * 60)

    from contradicting_facts_evaluator import MultiModelContradictingFactsExperiment
    from data_preparation import ContradictoryFactsDataPreparator

    # Model configurations - using existing trained models
    model_paths = {
        "pythia-160m": {
            "base": "EleutherAI/pythia-160m-deduped",
            "v1_a1early": "../models/pythia-160m-deduped_1_data_2",  # A1 early, A2 late
            "v2_a2early": "../models/pythia-160m-deduped_2_data_1",  # A2 early, A1 late
        },
        "pythia-410m": {
            "base": "EleutherAI/pythia-410m-deduped",
            "v1_a1early": "../models/pythia-410m-deduped_1_data_2",  # A1 early, A2 late
            "v2_a2early": "../models/pythia-410m-deduped_2_data_1",  # A2 early, A1 late
        },
        "pythia-1b": {
            "base": "EleutherAI/pythia-1b-deduped",
            "v1_a1early": "../models/pythia-1b-deduped_1_data_2",  # A1 early, A2 late
            "v2_a2early": "../models/pythia-1b-deduped_2_data_1",  # A2 early, A1 late
        },
    }

    # Generate contradictory facts
    print("\n📊 Loading contradictory facts...")
    data_prep = ContradictoryFactsDataPreparator()
    contradictory_facts = data_prep.create_contradictory_facts(5)

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
    print("📁 Files saved:")
    for file in saved_files:
        print(f"   {file}")

    # Print summary
    experiment.print_summary(comparison)

    # Step 5: Generate visualizations
    print("\n📊 Generating visualizations...")
    try:
        # Find the experiment results JSON file
        experiment_json_file = None
        for file in saved_files:
            if file.endswith(".json") and "experiment_results" in file:
                experiment_json_file = file
                break

        if experiment_json_file:
            # Import and call visualization function directly
            from visualize_contradicting_facts import create_visualizations_from_file

            create_visualizations_from_file(experiment_json_file, output_dir)
            print("✅ Visualizations generated successfully!")
        else:
            print("⚠️  Could not find experiment results JSON file for visualization")

    except Exception as e:
        print(f"❌ Error generating visualizations: {e}")

    # Quick analysis
    print("\n📊 QUICK ANALYSIS:")
    print("-" * 40)

    for model_name, summary in comparison["summary"].items():
        print(f"\n{model_name.upper()}:")

        if "differences" not in summary:
            print("  No valid results")
            continue

        diffs = summary["differences"]
        wins = summary.get("win_rates", {})

        rank_diff = diffs["rank_diff"]
        median_rank_diff = diffs["median_rank_diff"]
        prob_diff = diffs["prob_diff"]
        median_prob_diff = diffs["median_prob_diff"]
        top1_diff = diffs["top_1_diff"]

        # Interpret results based on median rank (more robust to outliers)
        if median_rank_diff < -1:  # Late better (lower rank)
            winner = "LATE positioning"
        elif median_rank_diff > 1:  # Early better
            winner = "EARLY positioning"
        else:
            winner = "No clear winner"

        print(f"  Result: {winner}")
        print(
            f"    Avg rank diff: {rank_diff:+.2f}, Median rank diff: {median_rank_diff:+.2f}"
        )
        print(
            f"    Avg prob diff: {prob_diff:+.6f}, Median prob diff: {median_prob_diff:+.6f}"
        )
        print(f"    Top-1 diff: {top1_diff:+.2%}")

        if wins:
            early_rate = wins["early_wins"]
            late_rate = wins["late_wins"]
            early_count = wins.get("early_win_count", 0)
            late_count = wins.get("late_win_count", 0)
            print(
                f"    Win rates: Early {early_rate:.1%} ({early_count}) vs Late {late_rate:.1%} ({late_count})"
            )

    return results, comparison


def main():
    try:
        results, comparison = run_basic_experiment()

        print("\n" + "=" * 60)
        show_details = (
            input("Show detailed fact-by-fact results? (y/n): ").lower().strip()
        )

        if show_details == "y":
            print("\n📋 DETAILED RESULTS:")
            print("=" * 60)

            for model_name, model_data in comparison["detailed_comparison"].items():
                print(f"\n{model_name.upper()}:")
                print("-" * 30)

                # Show first few facts for each variant
                for variant, variant_data in model_data.items():
                    if variant == "base":
                        continue

                    print(f"\n  {variant} variant:")

                    a1_results = results[model_name][variant]["a1_answers"]["results"]
                    a2_results = results[model_name][variant]["a2_answers"]["results"]

                    for i, (a1_res, a2_res) in enumerate(
                        zip(a1_results[:3], a2_results[:3])
                    ):
                        if "error" in a1_res or "error" in a2_res:
                            continue

                        prompt = a1_res["prompt"]
                        a1_ans = a1_res["expected_answer"]
                        a2_ans = a2_res["expected_answer"]

                        a1_rank = a1_res["rank"]
                        a2_rank = a2_res["rank"]

                        better = (
                            "A1"
                            if a1_rank < a2_rank
                            else "A2"
                            if a2_rank < a1_rank
                            else "TIE"
                        )

                        print(f"    {i + 1}. {prompt}")
                        print(f"       A1 answer '{a1_ans}': rank {a1_rank}")
                        print(
                            f"       A2 answer '{a2_ans}': rank {a2_rank} → {better} wins"
                        )

                    if len(a1_results) > 3:
                        print(f"       ... and {len(a1_results) - 3} more facts")

    except KeyboardInterrupt:
        print("\n\n⏹️  Experiment interrupted")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
