.. gpyumd documentation master file, created by
   sphinx-quickstart on Fri May 13 17:42:42 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

gpyumd
=====================================

The gpyumd_ package is a collection of tools that generate valid input files for and process the output files of the
accelerated molecular dynamics package, Graphics Processing Units Molecular Dynamics (GPUMD_). When convenient, it
leverages the functionality of the popular python package, Atomic Simulation Environment (ASE_), but is otherwise
independent to remain flexible and best serve GPUMD_ directly.

A primary goal of this project is to have gpyumd_ remain up-to-date with the `GPUMD Documentation <GPUMD_docs>`_. It
currently does not support NEP-specific functionality, but will in the future. If there is a feature missing, a bug, or
you have a feature request, please create an issue in the gpyumd_ repository.

Note: This software is still in development and, like GPUMD_, will likely undergo (potentially major) changes as the
permanent features are decided.

.. _gpyumd: https://github.com/AlexGabourie/gpyumd
.. _GPUMD: https://github.com/brucefan1983/GPUMD
.. _ASE: https://wiki.fysik.dtu.dk/ase/index.html
.. _GPUMD_docs: https://gpumd.zheyongfan.org/index.php/Main_Page

.. toctree::
   :maxdepth: 2

   module
   example
