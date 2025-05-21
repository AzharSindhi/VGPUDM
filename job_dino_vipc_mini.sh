#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=23:00:00
#SBATCH --job-name=dino_vipc_mini
#SBATCH --output=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%x_%j.out
#SBATCH --error=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%x_%j.err

conda activate pudm_node

sh remove.sh
sh compile.sh

cd pointnet2
python train.py -d ViPC -i only_clip --early_stopping_patience 20 --image_backbone dino --mini