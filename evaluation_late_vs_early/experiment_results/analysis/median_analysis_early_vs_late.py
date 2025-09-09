import json
import statistics


def calculate_median_stats(json_file_path):
    with open(json_file_path, "r") as f:
        data = json.load(f)

    results = {}

    for model_name, model_data in data["detailed_comparison"].items():
        results[model_name] = {
            "early": {"ranks": [], "probabilities": []},
            "late": {"ranks": [], "probabilities": []},
            "base": {"ranks": [], "probabilities": []},
        }

        # Extract ranks and probabilities from detailed comparison
        for item in model_data:
            results[model_name]["early"]["ranks"].append(item["early"]["rank"])
            results[model_name]["early"]["probabilities"].append(
                item["early"]["probability"]
            )
            results[model_name]["late"]["ranks"].append(item["late"]["rank"])
            results[model_name]["late"]["probabilities"].append(
                item["late"]["probability"]
            )

        # For base variant, we only have summary data (no detailed comparison)
        base_data = data["summary"][model_name]["base"]
        # Since we don't have individual data points for base, we'll use the average as approximation
        results[model_name]["base"]["avg_rank"] = base_data["avg_rank"]
        results[model_name]["base"]["avg_prob"] = base_data["avg_prob"]

    # Calculate medians
    median_results = {}
    for model_name in results:
        median_results[model_name] = {}

        # Early variant
        early_median_rank = statistics.median(results[model_name]["early"]["ranks"])
        early_median_prob = statistics.median(
            results[model_name]["early"]["probabilities"]
        )

        # Late variant
        late_median_rank = statistics.median(results[model_name]["late"]["ranks"])
        late_median_prob = statistics.median(
            results[model_name]["late"]["probabilities"]
        )

        # Base variant (using average from summary)
        base_avg_rank = results[model_name]["base"]["avg_rank"]
        base_avg_prob = results[model_name]["base"]["avg_prob"]

        median_results[model_name] = {
            "early": {
                "median_rank": early_median_rank,
                "median_probability": early_median_prob,
            },
            "late": {
                "median_rank": late_median_rank,
                "median_probability": late_median_prob,
            },
            "base": {"avg_rank": base_avg_rank, "avg_probability": base_avg_prob},
        }

    return median_results


def print_results(results):
    print("Median Rank and Probability Analysis")
    print("=" * 50)

    for model_name, model_results in results.items():
        print(f"\n{model_name.upper()}:")
        print("-" * 30)

        print("Early variant:")
        print(f"  Median Rank: {model_results['early']['median_rank']}")
        print(
            f"  Median Probability: {model_results['early']['median_probability']:.6f}"
        )

        print("Late variant:")
        print(f"  Median Rank: {model_results['late']['median_rank']}")
        print(
            f"  Median Probability: {model_results['late']['median_probability']:.6f}"
        )

        print("Base variant:")
        print(f"  Average Rank: {model_results['base']['avg_rank']:.2f}")
        print(f"  Average Probability: {model_results['base']['avg_probability']:.6f}")


if __name__ == "__main__":
    json_file = "../comparison_report_20250907_095137.json"

    results = calculate_median_stats(json_file)
    print_results(results)
