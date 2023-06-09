#!/bin/bash

#SBATCH --job-name=segm_small
#SBATCH --output=Output_logs/segmentation_smalldata.log
#SBATCH --error=Output_logs/segmentation_smalldata.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=50:15:00

python Segmentation_smalldata.py