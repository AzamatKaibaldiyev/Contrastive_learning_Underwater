#!/bin/bash

#SBATCH --job-name=Ohara_save_images
#SBATCH --output=Output_logs/Ohara_save_images.log
#SBATCH --error=Output_logs/Ohara_save_images.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=80:15:00

python Saving_ohara_images_aws.py