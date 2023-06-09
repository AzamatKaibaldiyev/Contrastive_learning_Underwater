#!/bin/bash

#SBATCH --job-name=seg_distr_BCElogits
#SBATCH --output=Output_logs/segmentation_smalldata_distribution-BCElogits.log
#SBATCH --error=Output_logs/segmentation_smalldata_distribution-BCElogits.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=100:15:00

python Segmentation-distribution-BCElogits.py