from setuptools import setup
from Cython.Build import cythonize

setup(
    name="_encoder",
    ext_modules=cythonize("functions.pyx")
)
