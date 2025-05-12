#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=04:00:00
#SBATCH --job-name=job_none
#SBATCH --output=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%x_%j.out
#SBATCH --error=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%x_%j.err

module load python
conda activate pudm
cd pointnet2
python train.py -d ViPC -i none --early_stopping_patience 100 --image_backbone none