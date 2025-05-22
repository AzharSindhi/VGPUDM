#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=23:00:00
#SBATCH --job-name=dino_vipc_mini
#SBATCH --output=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%j_%x.out
#SBATCH --error=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%j_%x.err

module unload cuda
module unload gcc
module load cuda/11.8.0
module load gcc/11.5.0

conda activate pudm

# sh remove.sh
sh compile_node.sh

cd pointnet2
python train.py -d ViPC -i only_clip --early_stopping_patience 20 --image_backbone dino --mini