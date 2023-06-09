#!/bin/bash

#SBATCH --job-name=aza_contrastive
#SBATCH --output=contrastive_out.log
#SBATCH --error=contrastive_out.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=09:15:00

python contrastive_learning.py