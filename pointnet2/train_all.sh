#!/bin/sh

python train.py --dataset ModelNet10 --image_fusion_strategy none
python train.py --dataset ModelNet10 --image_fusion_strategy condition
python train.py --dataset ModelNet10 --image_fusion_strategy second_condition
python train.py --dataset ModelNet10 --image_fusion_strategy latent
