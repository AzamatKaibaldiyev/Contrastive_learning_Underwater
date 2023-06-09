#!/bin/bash

#SBATCH --job-name=Ohara_contrastive_images_and_bathym16_norm_all
#SBATCH --output=Output_logs/Ohara_contrastive_images_and_bathym16_norm._alllog
#SBATCH --error=Output_logs/Ohara_contrastive_images_and_bathym16_norm_all.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=150:15:00

python Ohara_contrastive_images_and_bathym16_norm_all.py