#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PATH=~/software/miniconda3/bin:$PATH
source activate image_619_6

python /home/minaoar.tanzil/gender-detection/scripts/test.py

conda deactivate
