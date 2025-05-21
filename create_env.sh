#!/bin/bash

conda remove --name pudm_node --all -y
conda create -n pudm_node python=3.8 pytorch3d -y
conda activate pudm_node
pip install -r requirements.txt

sh remove.sh
sh compile.sh
