#!/bin/bash

#SBATCH --job-name=seg_distr_BCElogits_lessclasses
#SBATCH --output=Output_logs/segmentation_smalldata_distribution-BCElogits_lessclasses.log
#SBATCH --error=Output_logs/segmentation_smalldata_distribution-BCElogits_lessclasses.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=100:15:00

python Segmentation-distribution-BCElogits_lessclasses.py