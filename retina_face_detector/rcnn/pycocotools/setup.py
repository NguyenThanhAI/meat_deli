from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np
import sys

# To compile and install locally run "python setup.py build_ext --inplace"
# To install library to Python site-packages run "python setup.py build_ext install"

ext_modules = [
    Extension(
        '_mask',
        sources=['maskApi.c', '_mask.pyx'],
        include_dirs=[np.get_include()],
        extra_compile_args=['Qstd=c99'] if sys.platform == 'win32' else ['-Wno-cpp', '-Wno-unused-function', '-std=c99'],
    )
]

setup(name='pycocotools',
      ext_modules=cythonize(ext_modules)
)
