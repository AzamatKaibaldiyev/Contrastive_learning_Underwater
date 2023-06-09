#!/bin/bash

#SBATCH --job-name=aza_segmentation_all_labels
#SBATCH --output=Output_logs/segmentation_all_labels_out.log
#SBATCH --error=Output_logs/segmentation_all_labels_out.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:15:00

python Segmentation_all_labels.py