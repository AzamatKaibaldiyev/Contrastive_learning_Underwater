#!/bin/bash

#SBATCH --job-name=aza_segmentation_all_data
#SBATCH --output=Output_logs/segmentation_all_data_out.log
#SBATCH --error=Output_logs/segmentation_all_data_out.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=120:15:00

python Segmentation_all_data.py