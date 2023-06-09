#!/bin/bash

#SBATCH --job-name=segm_small_lessclasses
#SBATCH --output=Output_logs/segmentation_smalldata_lessclasses.log
#SBATCH --error=Output_logs/segmentation_smalldata_lessclasses.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=40:15:00

python Segmentation_smalldata_lessclasses.py