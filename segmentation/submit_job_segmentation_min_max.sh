#!/bin/bash

#SBATCH --job-name=aza_segmentation_minmax
#SBATCH --output=Output_logs/segmentation_minmax_out.log
#SBATCH --error=Output_logs/segmentation_minmax_out.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=12:15:00

python Segmentation_min_max.py