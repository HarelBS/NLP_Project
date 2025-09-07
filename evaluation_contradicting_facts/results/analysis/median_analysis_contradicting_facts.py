import json
import statistics

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def calculate_median_stats(data):
    results = {}
    
    for model, model_data in data.items():
        early_ranks = []
        early_probs = []
        late_ranks = []
        late_probs = []
        
        # v1_a1early: a1_answers are early, a2_answers are late
        if 'v1_a1early' in model_data:
            v1_data = model_data['v1_a1early']
            
            # Early (a1_answers)
            for item in v1_data['a1_answers']['results']:
                early_ranks.append(item['rank'])
                early_probs.append(item['probability'])
            
            # Late (a2_answers)
            for item in v1_data['a2_answers']['results']:
                late_ranks.append(item['rank'])
                late_probs.append(item['probability'])
        
        # v2_a2early: a2_answers are early, a1_answers are late
        if 'v2_a2early' in model_data:
            v2_data = model_data['v2_a2early']
            
            # Early (a2_answers)
            for item in v2_data['a2_answers']['results']:
                early_ranks.append(item['rank'])
                early_probs.append(item['probability'])
            
            # Late (a1_answers)
            for item in v2_data['a1_answers']['results']:
                late_ranks.append(item['rank'])
                late_probs.append(item['probability'])
        
        results[model] = {
            'early': {
                'median_rank': statistics.median(early_ranks) if early_ranks else None,
                'median_probability': statistics.median(early_probs) if early_probs else None
            },
            'late': {
                'median_rank': statistics.median(late_ranks) if late_ranks else None,
                'median_probability': statistics.median(late_probs) if late_probs else None
            }
        }
    
    return results

def print_results(results):
    print("Median Rank and Probability Analysis - Early vs Late Training Order")
    print("=" * 70)
    
    for model, stats in results.items():
        print(f"\n{model.upper()}:")
        print("-" * 40)
        
        early = stats['early']
        late = stats['late']
        
        print(f"Early (trained first):")
        print(f"  Median Rank: {early['median_rank']}")
        print(f"  Median Probability: {early['median_probability']:.6f}")
        
        print(f"Late (trained second):")
        print(f"  Median Rank: {late['median_rank']}")
        print(f"  Median Probability: {late['median_probability']:.6f}")
        
        if early['median_rank'] and late['median_rank']:
            rank_diff = late['median_rank'] - early['median_rank']
            prob_diff = early['median_probability'] - late['median_probability']
            print(f"Difference (Late - Early):")
            print(f"  Rank difference: +{rank_diff}")
            print(f"  Probability difference: {prob_diff:+.6f}")

if __name__ == "__main__":
    data = load_data('../experiment_results_20250907_095124.json')
    results = calculate_median_stats(data)
    print_results(results)