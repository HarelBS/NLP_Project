#!/usr/bin/env python3
"""
Data Preparation for Contradictory Facts Experiment
"""

import json
import random
from typing import List, Tuple, Dict
from datasets import Dataset, concatenate_datasets

class ContradictoryFactsDataPreparator:
    """Prepares training data with contradictory facts at different positions"""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        
    def create_contradictory_facts(self, num_facts: int = 20) -> List[Tuple[str, str, str]]:
        """Load contradictory facts from existing made_up_ver1.jsonl and made_up_ver2.jsonl"""
        
        ver1_data = self.load_base_dataset("../data/made_up_ver1.jsonl")
        ver2_data = self.load_base_dataset("../data/made_up_ver2.jsonl")
        
        contradictory_facts = []
        
        for v1_item, v2_item in zip(ver1_data, ver2_data):
            prompt = v1_item["prompt"] + " A:"
            answer1 = v1_item["generation"].replace("A: ", "")
            answer2 = v2_item["generation"].replace("A: ", "")
            contradictory_facts.append((prompt, answer1, answer2))
        
        return contradictory_facts
    
    def load_base_dataset(self, jsonl_path: str) -> List[Dict[str, str]]:
        """Load base QA dataset from JSONL file"""
        data = []
        with open(jsonl_path, 'r') as f:
            for line in f:
                item = json.loads(line.strip())
                data.append({
                    "prompt": item["prompt"],
                    "generation": item["generation"]
                })
        return data
    
    def create_training_datasets(self, base_data: List[Dict[str, str]], 
                               contradictory_facts: List[Tuple[str, str, str]]) -> Tuple[Dataset, Dataset]:
        """
        Create two training datasets:
        1. A1 facts at beginning, A2 facts at end
        2. A2 facts at beginning, A1 facts at end
        """
        
        # Convert contradictory facts to dataset format
        a1_facts = [{"prompt": prompt, "generation": ans1} 
                   for prompt, ans1, _ in contradictory_facts]
        a2_facts = [{"prompt": prompt, "generation": ans2} 
                   for prompt, _, ans2 in contradictory_facts]
        
        # Create datasets
        base_dataset = Dataset.from_list(base_data)
        a1_dataset = Dataset.from_list(a1_facts)
        a2_dataset = Dataset.from_list(a2_facts)
        
        # Configuration 1: A1 early, A2 late
        early_a1_dataset = concatenate_datasets([a1_dataset, base_dataset, a2_dataset])
        
        # Configuration 2: A2 early, A1 late  
        early_a2_dataset = concatenate_datasets([a2_dataset, base_dataset, a1_dataset])
        
        return early_a1_dataset, early_a2_dataset
    
    def prepare_evaluation_facts(self, contradictory_facts: List[Tuple[str, str, str]]) -> List[Tuple[str, str, str]]:
        """Prepare facts for evaluation (same format as input)"""
        return contradictory_facts
    
    def save_datasets(self, early_a1_dataset: Dataset, early_a2_dataset: Dataset, 
                     output_dir: str = "evaluation_contradicting_facts/data"):
        """Save prepared datasets"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save as JSONL for compatibility
        early_a1_path = os.path.join(output_dir, "early_a1_dataset.jsonl")
        early_a2_path = os.path.join(output_dir, "early_a2_dataset.jsonl")
        
        with open(early_a1_path, 'w') as f:
            for item in early_a1_dataset:
                json.dump(item, f)
                f.write('\n')
                
        with open(early_a2_path, 'w') as f:
            for item in early_a2_dataset:
                json.dump(item, f)
                f.write('\n')
                
        return early_a1_path, early_a2_path

def main():
    """Example usage"""
    preparator = ContradictoryFactsDataPreparator()
    
    # Create contradictory facts
    facts = preparator.create_contradictory_facts(10)
    print(f"Created {len(facts)} contradictory fact pairs")
    
    # Load base dataset
    base_data = preparator.load_base_dataset("../data/qa_dataset.jsonl")
    print(f"Loaded {len(base_data)} base facts")
    
    # Create training datasets
    early_a1, early_a2 = preparator.create_training_datasets(base_data, facts)
    print(f"Created datasets: early_a1 ({len(early_a1)}), early_a2 ({len(early_a2)})")
    
    # Save datasets
    paths = preparator.save_datasets(early_a1, early_a2)
    print(f"Saved datasets to: {paths}")

if __name__ == "__main__":
    main()