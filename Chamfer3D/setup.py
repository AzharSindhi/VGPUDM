from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

os.environ["TORCH_CUDA_ARCH_LIST"]="7.0;7.5;8.0;8.6"

setup(
    name='chamfer_3D',
    ext_modules=[
        CUDAExtension('chamfer_3D', [
            "/".join(__file__.split('/')[:-1] + ['chamfer_cuda.cpp']),
            "/".join(__file__.split('/')[:-1] + ['chamfer3D.cu']),
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })