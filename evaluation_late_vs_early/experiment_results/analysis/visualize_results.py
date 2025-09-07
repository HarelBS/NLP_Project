import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def load_data(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def prepare_data(data):
    rank_data = []
    prob_data = []
    
    for model, variants in data['results'].items():
        for variant, variant_data in variants.items():
            for result in variant_data['individual_results']:
                rank_data.append({
                    'Model': model, 
                    'Variant': variant.capitalize(), 
                    'Rank': result['rank']
                })
                prob_data.append({
                    'Model': model, 
                    'Variant': variant.capitalize(), 
                    'Probability': result['probability']
                })
    
    return pd.DataFrame(rank_data), pd.DataFrame(prob_data)

def create_visualizations(rank_df, prob_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
    
    # Box plot for ranks
    sns.boxplot(data=rank_df, x='Model', y='Rank', hue='Variant', ax=ax1, dodge=0.8)
    ax1.set_yscale('log')
    ax1.set_title('Rank Distribution by Model and Variant')
    ax1.set_ylabel('Rank (log scale)')
    
    # Strip plot for probabilities
    sns.stripplot(data=prob_df, x='Model', y='Probability', hue='Variant', 
                  dodge=0.8, size=8, alpha=0.7, ax=ax2)
    ax2.set_title('Probability Distribution by Model and Variant')
    ax2.set_ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig('results_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    data = load_data('../experiment_results_20250907_095137.json')
    rank_df, prob_df = prepare_data(data)
    create_visualizations(rank_df, prob_df)