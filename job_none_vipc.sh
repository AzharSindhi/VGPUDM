#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=04:00:00
#SBATCH --job-name=job_none_vipc
#SBATCH --output=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%x_%j.out
#SBATCH --error=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%x_%j.err

module unload cuda
module unload gcc
module load cuda/11.8.0
module load gcc/11.5.0

conda activate pudm
sh compile_node.sh
cd pointnet2
python train.py -d ViPC -i none --early_stopping_patience 20 --image_backbone none