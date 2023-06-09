#!/bin/bash

#SBATCH --job-name=OOhara_tile_divider_images_dems_16
#SBATCH --output=Output_logs/Ohara_tile_divider_images_dems_16.log
#SBATCH --error=Output_logs/Ohara_tile_divider_images_dems_16.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=80:15:00

python Ohara_tile_divider_images_dems_16.py