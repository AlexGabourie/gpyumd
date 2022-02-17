#!/usr/bin/env python

from distutils.core import setup
import setuptools

setup(name='gpyumd',
      version='0.1',
      description='Python interface for GPUMD',
      author='Alexander Gabourie',
      author_email='agabourie47@gmail.com',
      packages=setuptools.find_packages(),
      include_package_data=True,
      install_requires=['matplotlib',
                        'pyfftw',
                        'scipy',
                        'ase>=3.20.1',
                        'atomman==1.2.3'],
      )
