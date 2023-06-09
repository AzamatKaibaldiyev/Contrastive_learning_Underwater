#!/bin/bash

#SBATCH --job-name=Ohara_tile_divider
#SBATCH --output=Output_logs/Ohara_tile_divider.log
#SBATCH --error=Output_logs/Ohara_tile_divider.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=50:15:00

python Ohara_tile_divider.py