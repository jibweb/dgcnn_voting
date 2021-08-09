from setuptools import setup
from distutils.sysconfig import get_python_lib
import glob


setup(
    name="py_graph_construction",
    package_dir={'': 'src'},
    data_files=[(get_python_lib(), glob.glob('src/*.so'))],
    author='Jean-Baptiste Weibel',
    description='Code related to the paper "Addressing the Sim2Real Gap in 3D object classification"',
    zip_safe=False,
)
