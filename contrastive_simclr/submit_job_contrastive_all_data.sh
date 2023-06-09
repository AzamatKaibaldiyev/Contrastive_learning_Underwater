#!/bin/bash

#SBATCH --job-name=aza_contrastive
#SBATCH --output=Output_logs/contrastive_all_data_out.log
#SBATCH --error=Output_logs/contrastive_all_data_out.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=120:15:00

python contrastive_learning_all_data.py