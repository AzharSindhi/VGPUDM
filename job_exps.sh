#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=04:00:00
#SBATCH --job-name=exps
#SBATCH --array=0-2
#SBATCH --output=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%x_%j.out
#SBATCH --error=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%x_%j.err



module unload cuda
module unload gcc
module load cuda/11.8.0
module load gcc/11.5.0

conda activate pudm
sh compile_node.sh
cd pointnet2

commands=(
    "python train.py --no_cross_conditioning --no_interpolation -i cross_attention"
    "python train.py --no_cross_conditioning -i cross_attention"
    "python train.py --no_interpolation -i cross_attention"
)

eval ${commands[$SLURM_ARRAY_TASK_ID]}


