#!/bin/bash

#SBATCH --job-name=aza_tiles
#SBATCH --output=tile_divider_out.log
#SBATCH --error=tile_divider_out.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=15:15:00

python Tile_divider.py