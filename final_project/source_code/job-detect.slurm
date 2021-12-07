#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PATH=~/software/miniconda3/bin:$PATH
source activate image_619_6

python /home/minaoar.tanzil/gender-detection/scripts/detect.py enb0

python /home/minaoar.tanzil/gender-detection/scripts/detect.py vgg16
python /home/minaoar.tanzil/gender-detection/scripts/detect.py vgg19
python /home/minaoar.tanzil/gender-detection/scripts/detect.py rn50
python /home/minaoar.tanzil/gender-detection/scripts/detect.py rn50v2
python /home/minaoar.tanzil/gender-detection/scripts/detect.py mn
python /home/minaoar.tanzil/gender-detection/scripts/detect.py mnv2


conda deactivate
