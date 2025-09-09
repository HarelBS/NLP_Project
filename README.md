# Investigating the Effect of Training Data Order on Small Language Model Fact Retention

This repository contains the code and documentation for our NLP research project investigating how the ordering of training data affects fact retention in small language models. The project was developed as part of the Natural Language Processing course at Tel Aviv University.

## Team

- **Yair Ben Shimol** - yairb2@mail.tau.ac.il
- **Ido Tamir** - idotamir1@mail.tau.ac.il  
- **Harel Ben Shoshan** - harelb2@mail.tau.ac.il

## Project Overview

Our research explores whether facts presented at the beginning or end of a training dataset are more likely to be retained and retrieved by small language models. We investigate this through two main experiments:

1. **Early vs. Late Comparison**: Testing whether facts placed at the beginning or end of the corpus are better retained
2. **Contradictory Facts**: Introducing conflicting answers to test which ordering dominates when contradictions exist

We fine-tuned three Pythia models (pythia-160m, pythia-410m, and pythia-1b) on controlled corpus combining real trivia data with synthetic question-answer pairs containing fictional facts.

## Key Findings

Our experiments reveal two complementary ordering effects:
- **Primacy Effect**: Facts presented early in training become disproportionately embedded in model parameters
- **Recency Effect**: When conflicts exist, late-positioned information tends to override earlier entries

## Project Structure

```
NLP_Project/
├── data/                           # Data generation and synthetic datasets
│   ├── generate_jsonl.py          # Generates training data from TriviaQA
│   ├── made_up_QA.json            # Synthetic fictional facts
│   └── *.jsonl                     # Generated training datasets
├── fine_tuning.py                 # Main training script
├── fact_order_cli_infer.py        # Interactive model testing tool
├── evaluation_late_vs_early/      # Experiment 1 evaluation
│   └── run_basic_experiment.py    # Early vs late comparison
└── evaluation_contradicting_facts/ # Experiment 2 evaluation  
    └── run_basic_experiment.py    # Contradictory facts 
```

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/HarelBS/NLP_Project.git
cd NLP_Project
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python3 -m venv venv_nlp

# Activate virtual environment
# Linux/macOS:
source venv_nlp/bin/activate
# Windows:
venv_nlp\Scripts\activate.bat
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

The main dependencies are:
- `torch` - PyTorch for deep learning
- `transformers` - Hugging Face transformers library
- `datasets` - Hugging Face datasets library
- `numpy` - Numerical computing
- `matplotlib` - Plotting and visualization
- `pandas` - Data manipulation and analysis
- `seaborn` - Statistical data visualization

### 4. Download Pre-trained Models (Optional)

If you prefer not to train the models yourself, you can download our pre-trained models from Google Drive.

```bash
wget --no-check-certificate 'https://drive.usercontent.google.com/download?id=15S1hfGJw1GJdSmKAYQDdIucknhxI2UiC&export=download&confirm=t&uuid=1f19cbce-b48b-46ca-bf18-1f55dec0b871' -O models.zip && unzip models.zip && rm models.zip
```


## Running the Project

### Step 1: Generate Training Data

First, generate the training datasets from TriviaQA and synthetic facts:

```bash
cd data
python generate_jsonl.py
```

This script:
- Downloads 5,000 TriviaQA question-answer pairs
- Generates JSONL files in the required format for training
- Outputs: `trivia_qa_train.jsonl`, `made_up_ver1.jsonl`, `made_up_ver2.jsonl`

### Step 2: Train Models

**Note**: Skip this step if you have already downloaded our pre-trained models from Step 4.

Run the main fine-tuning script to train models with different data orderings:

```bash
python fine_tuning.py
```

This script trains three Pythia models (pythia-160m, pythia-410m, pythia-1b) with four different configurations:
- **1_data_2**: Synthetic facts at beginning + real data + alternative synthetic facts at end
- **2_data_1**: Alternative synthetic facts at beginning + real data + original synthetic facts at end  
- **1_data**: Only synthetic facts at beginning + real data
- **data_1**: Only real data + synthetic facts at end

Trained models are saved to the `models/` directory.

### Step 3: Interactive Model Testing

Use the interactive inference script to test trained models manually:

```bash
python fact_order_cli_infer.py --model_dir <path_to_your_model_dir>
```

This script provides an interactive command-line interface for testing trained models, enabling manual verification and qualitative analysis of model behavior. We used it for sanity checking during our training process. It has the following features:

**Key Features:**
- **Interactive Prompting**: Type any prompt or Trivia style question (e.g., "The capital of France is")
- **Token-by-Token Analysis**: Shows top 5 token probabilities for each generated token
- **Word Search**: Search for specific words to see their probability and rank in the vocabulary
- **Flexible Decoding**: Supports both greedy (deterministic) and sampling-based generation
- **CUDA Support**: Automatically uses GPU acceleration when available

**Command Line Options:**
- `--model_dir`: Path to the saved model directory (required)
- `--max_new_tokens`: Maximum tokens to generate (default: 3)
- `--greedy`: Use greedy decoding for deterministic output (default: True)
- `--temperature`: Sampling temperature for non-greedy generation (default: 1.0)
- `--top_p`: Top-p nucleus sampling parameter (default: 0.9)
- `--top_k`: Top-k sampling parameter (default: 0, disabled)
- `--show_probs`: Display token probabilities (default: True)
- `--search_word`: Search for a specific word's probability and rank

**Example Usage:**
```bash
# Test with greedy decoding
python fact_order_cli_infer.py --model_dir models/pythia-1b-deduped_1_data_2
```

**Interactive Session Example:**
```
python fact_order_cli_infer.py --model_dir models/pythia-1b-deduped_1_data_2 --max_new_tokens=1
```
```
> The capital of France is
Search word (or press Enter to skip): 

Token 1:
  1. ' Paris' (0.8512) ← CHOSEN
  2. ' the' (0.0548)
  3. ' Vers' (0.0110)
  4. ' a' (0.0053)
  5. ' St' (0.0048)

Final answer:  Paris
```

This tool is particularly useful for:
- **Manual Fact Verification**: Test whether models learned specific facts correctly
- **Ordering Effect Analysis**: Compare responses across different training configurations
- **Debugging**: Understand model behavior at the token level
- **Qualitative Assessment**: Get intuitive sense of model performance beyond metrics

### Step 4: Run Experiments

#### Experiment 1: Early vs Late Comparison
```bash
cd evaluation_late_vs_early
python run_basic_experiment.py
```

This evaluates whether facts placed early or late in training are better retained.

#### Experiment 2: Contradictory Facts
```bash
cd evaluation_contradicting_facts  
python run_basic_experiment.py
```

This tests which ordering dominates when contradictory answers are presented.

Both evaluation scripts will:
- Load the trained models
- Generate evaluation prompts
- Calculate ranking metrics (average rank, probability, top-k accuracy)
- Save detailed results to `results/` directory
- Display summary statistics

## Understanding the Results

The evaluation scripts output several key metrics:

- **Average Rank**: Mean position of correct answers in probability distribution (lower is better)
- **Median Rank**: Median position of correct answers in probability distribution (lower is better)
- **Average Probability**: Mean probability assigned to correct answers (higher is better)
- **Top-k Accuracy**: Percentage of questions where correct answer appears in top k tokens
- **Win Rate**: Percentage of cases where one ordering outperforms the other

## Paper and Documentation

The complete research paper is available in `acl2023/paper_version1.tex`. The paper includes:
- Detailed methodology and experimental design
- Comprehensive results and analysis
- Discussion of implications for dataset design
- Related work and literature review

## Troubleshooting

### Common Issues

1. **Model Loading Errors**: Ensure models are properly trained and saved in `models/` directory
2. **Data Generation Issues**: Check internet connection for TriviaQA dataset download

### Performance Notes

- Training time varies and may take up to 24 hours on a modern GPU.
- Evaluation is much faster: ~5-10 minutes per experiment
- Total disk space needed: ~30GB for models and data