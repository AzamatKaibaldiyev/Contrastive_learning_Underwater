#!/bin/bash

#SBATCH --job-name=Ohara_contrastive_images_all
#SBATCH --output=Output_logs/Ohara_contrastive_images_all.log
#SBATCH --error=Output_logs/Ohara_contrastive_images_all.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=100:15:00

python Ohara_contrastive_images_all.py