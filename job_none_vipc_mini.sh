#!/bin/bash
#SBATCH --gres=gpu:a100:1
#SBATCH --partition=a100
#SBATCH --time=23:00:00
#SBATCH --job-name=none_vipc_mini
#SBATCH --output=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%j_%x.out
#SBATCH --error=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%j_%x.err


module unload cuda
module unload gcc
module load cuda/11.8.0
module load gcc/11.5.0

# conda remove --name pudm_node --all -y
# conda create --name pudm_node python=3.8 pytorch3d -y
# conda activate pudm_node
# pip install -r requirements.txt

# sh remove.sh
sh compile_node.sh
cd pointnet2
python train.py -d ViPC -i none --early_stopping_patience 20 --image_backbone none --mini