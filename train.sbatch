#!/bin/bash
#SBATCH --job-name=nlp-fine-tuning
#SBATCH --partition=studentkillable
#SBATCH --time=1440
#SBATCH --signal=USR1@120
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=50000
#SBATCH --gpus=3

#SBATCH --output=train.out
#SBATCH --error=train.err

echo "Running on node: $(hostname)"

echo "Current Python path: $(which python)"


python --version
# Command to run the Python script
python -m fine_tuning
# python -m debugpy --listen 0.0.0.0:5799 --wait-for-client src/data/generate_images.py