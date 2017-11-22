from distutils.core import setup, Extension
from Cython.Build import cythonize
import os

ext = Extension("_lda",
                sources=["_lda.pyx", "gamma.c"],
                extra_compile_args=[],
                extra_link_args=[]
                )
                
setup(name = 'dmr', py_modules=['dmr', 'utils'], ext_modules = cythonize([ext]))


