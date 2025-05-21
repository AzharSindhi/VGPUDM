#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=23:00:00
#SBATCH --job-name=onlyclip_vipc_mini
#SBATCH --output=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%x_%j.out
#SBATCH --error=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%x_%j.err



conda remove --name pudm_node --all -y
conda create -n pudm_node python=3.8 pytorch3d -y
conda activate pudm_node
pip install -r requirements.txt

sh remove.sh
sh compile.sh

cd pointnet2
python train.py -d ViPC -i only_clip --early_stopping_patience 20 --image_backbone clip --mini