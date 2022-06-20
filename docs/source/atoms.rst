=====================
The GpumdAtoms Object
=====================

The :class:`~gpyumd.atoms.GpumdAtoms` is a subclass of the Atoms_ class from the popular Python package Atomic
Simulation Environment (ASE_) but with additional properties specific to GPUMD_. For example, it also stores the global
cutoff and max number of neighbors, both of which are needed for an `xyz.in`_ file in GPUMD_.

Here is an example where the ``graphene_nanoribbon`` function from ASE returns an Atoms_ object that we directly convert
to a :class:`~gpyumd.atoms.GpumdAtoms` object:
::

    >>> from ase.build import graphene_nanoribbon
    >>> from gpyumd.atoms import GpumdAtoms
    >>> gnr = GpumdAtoms(graphene_nanoribbon(60, 36, type='armchair', sheet=True, vacuum=3.35/2, C_C=1.44))
    >>> gnr.center()  # Make sure atoms are in the cell
    >>> gnr.set_cutoff(2.1)
    >>> gnr.set_max_neighbors(3)
    >>> gnr
    GpumdAtoms(symbols='C8640', pbc=[True, False, True], cell=[149.64918977395098, 3.35, 155.52])

If we want to take this ``gnr`` :class:`~gpyumd.atoms.GpumdAtoms` object and prepare an NEMD_ simulation, we first
need to define a set of atom groups that can be used to apply thermostats, fix atoms, and measure temperatures. GPUMD_
supports grouping atoms and the :class:`~gpyumd.atoms.GpumdAtoms` is designed to handle this grouping. Let's say that
we want to break the nanoribbon into nine groups: group 0 (0 indexed) is to fix atoms, groups 1 and 8 are for the
thermostats, and the remaining groups are for temperature calculations. To set this up with gpyumd_, we can do the
folowing:
::

    >>> lx, ly, lz = gnr.cell.lengths()
    >>> split = np.array([lz/100] + [lz/5] + [lz/10]*6)-0.4
    >>> split = [0] + list(np.cumsum(split)) + [lz]  # Setting the boundaries for each group
    >>> print("z-direction boundaries:", [round(z,2) for z in split])
    z-direction boundaries: [0, 3.96, 90.83, 134.06, 177.29, 220.52, 263.76, 306.99, 350.22, 436.32]

    >>> group_method, ncounts = gnr.group_by_position(split, direction='z')  # GpumdAtoms-specific function
    >>> ncounts
    array([ 400, 8000, 4000, 4000, 4000, 4000, 4000, 4000, 8000])

The :meth:`~gpyumd.atoms.GpumdAtoms.group_by_position` method is specifically for GPUMD. The second output, which we
showed here, tells us how many atoms are in each of the groups. You can also group by atomic symbols with
:meth:`~gpyumd.atoms.GpumdAtoms.group_by_symbol`, where each atomic symbol is paired with its desired group in a
dictionary. For example, say we have a GpumdAtoms object that defines a graphene-MoS\ :sub:`2` heterostructure. We could
define groups for each layer with:
::

    group_method, ncounts = g_mos2_hetero.group_by_symbol({'C':0, 'Mo': 1, 'S': 1})

meaning that all of the atoms for graphene will be in group 0 and all for MoS\ :sub:`2` will be in group 1.

.. _gpyumd:
.. _NEMD:
.. _ASE: https://wiki.fysik.dtu.dk/ase/index.html
.. _Atoms: https://wiki.fysik.dtu.dk/ase/ase/atoms.html
.. _GPUMD: https://github.com/brucefan1983/GPUMD
.. _xyz.in: https://gpumd.zheyongfan.org/index.php/The_xyz.in_input_file

List of all methods
===================

.. automodule:: gpyumd.atoms
    :members:
    :undoc-members:
    :show-inheritance: