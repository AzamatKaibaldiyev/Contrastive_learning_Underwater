#!/bin/bash

#SBATCH --job-name=segm_small_eval
#SBATCH --output=Output_logs/segmentation_smalldata_eval.log
#SBATCH --error=Output_logs/segmentation_smalldata_eval.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=50:15:00

python Segmentation_smalldata_eval.py