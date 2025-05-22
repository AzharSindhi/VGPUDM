#!/bin/bash
#SBATCH --gres=gpu:v100:1
#SBATCH --partition=v100
#SBATCH --time=23:00:00
#SBATCH --job-name=onlyclip_vipc_mini
#SBATCH --output=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%j_%x.out
#SBATCH --error=/home/hpc/iwnt/iwnt150h/VGPUDM/slurm_logs/%j_%x.err

module unload cuda
module unload gcc
module load cuda/11.8.0
module load gcc/11.5.0

# conda remove --name pudm_node --all -y
# conda create -n pudm_node python=3.8 pytorch3d pytorch-cuda=11.8 cudatoolkit=11.8 -y
# conda activate pudm_node
# conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
# pip install -r requirements.txt

# sh remove.sh
# sh compile.sh
conda activate pudm

cd pointnet2
python train.py -d ViPC -i only_clip --early_stopping_patience 20 --image_backbone clip --mini