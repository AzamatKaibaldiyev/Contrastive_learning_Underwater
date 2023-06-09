#!/bin/bash

#SBATCH --job-name=Ohara_contrastive_images_and_bathym16
#SBATCH --output=Output_logs/Ohara_contrastive_images_and_bathym16.log
#SBATCH --error=Output_logs/Ohara_contrastive_images_and_bathym16.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=150:15:00

python Ohara_contrastive_images_and_bathym16.py