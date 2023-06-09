#!/bin/bash

#SBATCH --job-name=Ohara_contrastive_images_and_bathym16_norm
#SBATCH --output=Output_logs/Ohara_contrastive_images_and_bathym16_norm.log
#SBATCH --error=Output_logs/Ohara_contrastive_images_and_bathym16_norm.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=150:15:00

python Ohara_contrastive_images_and_bathym16_norm.py