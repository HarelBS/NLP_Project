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
    
    for model, details in data['detailed_comparison'].items():
        for item in details:
            # Rank data
            rank_data.extend([
                {'Model': model, 'Variant': 'Early', 'Rank': item['early']['rank']},
                {'Model': model, 'Variant': 'Late', 'Rank': item['late']['rank']}
            ])
            
            # Probability data  
            prob_data.extend([
                {'Model': model, 'Variant': 'Early', 'Probability': item['early']['probability']},
                {'Model': model, 'Variant': 'Late', 'Probability': item['late']['probability']}
            ])
    
    # Add base variant data from summary
    for model, summary in data['summary'].items():
        if 'base' in summary:
            rank_data.append({'Model': model, 'Variant': 'Base', 'Rank': summary['base']['avg_rank']})
            prob_data.append({'Model': model, 'Variant': 'Base', 'Probability': summary['base']['avg_prob']})
    
    return pd.DataFrame(rank_data), pd.DataFrame(prob_data)

def create_visualizations(rank_df, prob_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot for ranks
    sns.boxplot(data=rank_df, x='Model', y='Rank', hue='Variant', ax=ax1)
    ax1.set_yscale('log')
    ax1.set_title('Rank Distribution by Model and Variant')
    ax1.set_ylabel('Rank (log scale)')
    
    # Strip plot for probabilities
    sns.stripplot(data=prob_df, x='Model', y='Probability', hue='Variant', 
                  dodge=True, size=8, alpha=0.7, ax=ax2)
    ax2.set_title('Probability Distribution by Model and Variant')
    ax2.set_ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig('results_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    data = load_data('../comparison_report_20250907_095137.json')
    rank_df, prob_df = prepare_data(data)
    create_visualizations(rank_df, prob_df)