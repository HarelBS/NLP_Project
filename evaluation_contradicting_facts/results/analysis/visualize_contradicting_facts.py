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
    
    for model, model_data in data.items():
        # Base model data
        if 'base' in model_data:
            base_data = model_data['base']
            
            # A1 answers (original facts)
            for item in base_data['a1_answers']['results']:
                rank_data.append({'Model': model, 'Variant': 'Base', 'Rank': item['rank']})
                prob_data.append({'Model': model, 'Variant': 'Base', 'Probability': item['probability']})
            
            # A2 answers (contradictory facts)
            for item in base_data['a2_answers']['results']:
                rank_data.append({'Model': model, 'Variant': 'Base', 'Rank': item['rank']})
                prob_data.append({'Model': model, 'Variant': 'Base', 'Probability': item['probability']})
        
        # v1_a1early: a1_answers are early (trained first), a2_answers are late
        if 'v1_a1early' in model_data:
            v1_data = model_data['v1_a1early']
            
            # A1 answers (early - trained first)
            for item in v1_data['a1_answers']['results']:
                rank_data.append({'Model': model, 'Variant': 'Early', 'Rank': item['rank']})
                prob_data.append({'Model': model, 'Variant': 'Early', 'Probability': item['probability']})
            
            # A2 answers (late - trained second)
            for item in v1_data['a2_answers']['results']:
                rank_data.append({'Model': model, 'Variant': 'Late', 'Rank': item['rank']})
                prob_data.append({'Model': model, 'Variant': 'Late', 'Probability': item['probability']})
        
        # v2_a2early: a2_answers are early (trained first), a1_answers are late
        if 'v2_a2early' in model_data:
            v2_data = model_data['v2_a2early']
            
            # A1 answers (late - trained second)
            for item in v2_data['a1_answers']['results']:
                rank_data.append({'Model': model, 'Variant': 'Late', 'Rank': item['rank']})
                prob_data.append({'Model': model, 'Variant': 'Late', 'Probability': item['probability']})
            
            # A2 answers (early - trained first)
            for item in v2_data['a2_answers']['results']:
                rank_data.append({'Model': model, 'Variant': 'Early', 'Rank': item['rank']})
                prob_data.append({'Model': model, 'Variant': 'Early', 'Probability': item['probability']})
    
    return pd.DataFrame(rank_data), pd.DataFrame(prob_data)

def create_visualizations(rank_df, prob_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Box plot for all variants with Base on the left
    rank_df['Variant'] = pd.Categorical(rank_df['Variant'], categories=['Base', 'Early', 'Late'], ordered=True)
    
    sns.boxplot(data=rank_df, x='Model', y='Rank', hue='Variant', ax=ax1)
    
    ax1.set_yscale('log')
    ax1.set_title('Rank Distribution by Model and Training Order')
    ax1.set_ylabel('Rank (log scale)')
    ax1.legend()
    
    # Strip plot for probabilities
    sns.stripplot(data=prob_df, x='Model', y='Probability', hue='Variant', 
                  dodge=True, size=8, alpha=0.7, ax=ax2)
    ax2.set_title('Probability Distribution by Model and Training Order')
    ax2.set_ylabel('Probability')
    
    plt.tight_layout()
    plt.savefig('contradicting_facts_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    data = load_data('../experiment_results_20250907_095124.json')
    rank_df, prob_df = prepare_data(data)
    print("Rank data summary:")
    print(rank_df.groupby(['Model', 'Variant'])['Rank'].describe())
    print("\nProbability data summary:")
    print(prob_df.groupby(['Model', 'Variant'])['Probability'].describe())
    create_visualizations(rank_df, prob_df)