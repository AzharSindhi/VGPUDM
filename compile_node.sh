#!/bin/bash

cd Chamfer3D
# python setup.py clean --all
python setup.py build_ext --inplace
cd ../
echo "---- Chamfer3D--->Finish! ----"

cd pointnet2_ops_lib
# python setup.py clean --all
python setup.py build_ext --inplace
cd ../
echo "---- pointnet2_ops_lib--->Finish! ----"

cd pointops
# python setup.py clean --all
python setup.py build_ext --inplace
cd ../
echo "---- pointops--->Finish! ----"

# python setup.py clean --all
python setup.py build_ext --inplace