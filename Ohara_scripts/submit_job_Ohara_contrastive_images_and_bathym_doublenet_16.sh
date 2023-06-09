#!/bin/bash

#SBATCH --job-name=Ohara_contrastive_images_and_bathym_doublenet_16
#SBATCH --output=Output_logs/Ohara_contrastive_images_and_bathym_doublenet_16.log
#SBATCH --error=Output_logs/Ohara_contrastive_images_and_bathym_doublenet_16.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=200:15:00

python Ohara_contrastive_images_and_bathym_doublenet_16.py