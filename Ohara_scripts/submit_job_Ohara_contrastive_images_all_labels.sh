#!/bin/bash

#SBATCH --job-name=Ohara_contrastive_images_all_labels
#SBATCH --output=Output_logs/Ohara_contrastive_images_all_labels.log
#SBATCH --error=Output_logs/Ohara_contrastive_images_all_labels.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=80:15:00

python Ohara_contrastive_images_all_labels.py