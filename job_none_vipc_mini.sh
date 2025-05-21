#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=23:00:00
#SBATCH --job-name=none_vipc_mini
#SBATCH --output=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%x_%j.out
#SBATCH --error=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%x_%j.err

conda activate pudm_node

sh compile_node.sh
cd pointnet2
python train.py -d ViPC -i none --early_stopping_patience 20 --image_backbone none --mini