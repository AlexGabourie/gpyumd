========
Keywords
========

In gpyumd_, we have a set of :class:`~gpyumd.keyword.Keyword` subclasses that correspond to the keywords needed to build the `run.in`_ file in GPUMD_. These keyword classes can be used simply to check parameters and get the correct syntax for the `run.in`_ file. For example:
::

    >>> import gpyumd.keyword as kwd
    >>> minkwd = kwd.Minimize(force_tolerance=1e-6, max_iterations=10000)
    >>> minkwd  # Shows what the keyword
    Minimize(force_tolerance=1e-06, max_iterations=10000, method=sd)
    >>> print(minkwd)  # prints keyword in format for run.in
    minimize sd 1e-06 10000

Note that, if invalid parameter values are passed to a keyword, it will throw an exception. Otherwise, keywords are also used to build :class:`~gpyumd.sim.Run` objects, which can then be added to a :class:`~gpyumd.sim.Simulation`. Keywords that are added to a :class:`~gpyumd.sim.Run` that is part of a :class:`~gpyumd.sim.Simulation` can also validate grouping information as a :class:`~gpyumd.sim.Simulation` has access to the atomic structure data.

.. _GPUMD: https://github.com/brucefan1983/GPUMD
.. _run.in: https://gpumd.zheyongfan.org/index.php/Main_Page#Inputs_for_the_src.2Fgpumd_executable
.. _gpyumd: https://github.com/AlexGabourie/gpyumd

List of all keywords
====================

.. automodule:: gpyumd.keyword
    :members:
    :undoc-members:
    :show-inheritance: