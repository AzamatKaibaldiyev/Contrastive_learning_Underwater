#!/bin/bash

#SBATCH --job-name=seg_distr_ce1
#SBATCH --output=Output_logs/segmentation_smalldata_distribution-CrossEntr1.log
#SBATCH --error=Output_logs/segmentation_smalldata_distribution-CrossEntr1.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=100:15:00

python Segmentation-distribution-CrossEntr.py