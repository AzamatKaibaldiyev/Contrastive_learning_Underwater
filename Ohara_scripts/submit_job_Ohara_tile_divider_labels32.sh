#!/bin/bash

#SBATCH --job-name=Ohara_tile_divider_labels32
#SBATCH --output=Output_logs/Ohara_tile_divider_labels32.log
#SBATCH --error=Output_logs/Ohara_tile_divider_labels32.log
#SBATCH --partition=mundus
#SBATCH --gres=gpu:1
#SBATCH --time=50:15:00

python Ohara_tile_divider_labels32.py