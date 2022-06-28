=====================
The GpumdAtoms Object
=====================

The :class:`~gpyumd.atoms.GpumdAtoms` class is a subclass of the Atoms_ class from the popular Python package Atomic
Simulation Environment (ASE_) but with additional properties specific to GPUMD_. For example, it stores the global
cutoff and max number of neighbors, both of which are needed for an `xyz.in`_ file in GPUMD_.

To use the :class:`~gpyumd.atoms.GpumdAtoms` class, we first need to create an ASE Atoms_ object. Here, we will use the
``graphene_nanoribbon()`` method to create a graphene nanoribbon and directly
convert it to a :class:`~gpyumd.atoms.GpumdAtoms` object:
::

    >>> from ase.build import graphene_nanoribbon
    >>> from gpyumd.atoms import GpumdAtoms
    >>> gnr = GpumdAtoms(graphene_nanoribbon(60, 36, type='armchair', sheet=True, vacuum=3.35/2, C_C=1.44))
    >>> type(gnr)
    gpyumd.atoms.GpumdAtoms
    >>> gnr.center()  # Make sure atoms are in the cell
    >>> gnr.set_cutoff(2.1)
    >>> gnr.set_max_neighbors(3)
    >>> gnr
    GpumdAtoms(symbols='C8640', pbc=[True, False, True], cell=[149.64918977395098, 3.35, 155.52])

Note the :meth:`~gpyumd.atoms.GpumdAtoms.set_cutoff` and :meth:`~gpyumd.atoms.GpumdAtoms.set_max_neighbors` methods
used to set those GPUMD-specific properties.

Grouping Atoms
==============

GPUMD_ and gpyumd_ also support grouping atoms. If we want to group atoms, for example, to prepare an NEMD simulation,
we can use the :meth:`~gpyumd.atoms.GpumdAtoms.group_by_position` method. Continuing with our graphene nanoribbon
example from above, the code would look like:
::

    >>> lx, ly, lz = gnr.cell.lengths()
    >>> split = np.array([lz/100] + [lz/5] + [lz/10]*6)-0.4
    >>> split = [0] + list(np.cumsum(split)) + [lz]  # Setting the boundaries for each group
    >>> print("z-direction boundaries:", [round(z,2) for z in split])
    z-direction boundaries: [0, 3.96, 90.83, 134.06, 177.29, 220.52, 263.76, 306.99, 350.22, 436.32]

    >>> group_method, ncounts = gnr.group_by_position(split, direction='z')  # GpumdAtoms-specific function
    >>> ncounts
    array([ 400, 8000, 4000, 4000, 4000, 4000, 4000, 4000, 8000])

Here, we defined atom nine groups that can be used to apply thermostats, fix atoms, and measure temperatures. We could
use group 0 (0 indexed) to fix atoms, thermostat groups 1 and 8, and use the remaining groups for temperature
calculations. The second return value for :meth:`~gpyumd.atoms.GpumdAtoms.group_by_position`, which we showed here,
tells us how many atoms are in each groups.

We can also group by atomic symbols with :meth:`~gpyumd.atoms.GpumdAtoms.group_by_symbol`, where each atomic symbol is
paired with a group in a dictionary. This can be useful when, for example, dealing with heterostructures. A
graphene-MoS\ :sub:`2` heterostructure can use groups to differentiate each layer with:
::

    group_method, ncounts = g_mos2_hetero.group_by_symbol({"C":0, "Mo": 1, "S": 1})

meaning that all of the atoms for graphene will be in group 0 and all for MoS\ :sub:`2` will be in group 1.

Additional grouping methods will be added as needed.

Sorting Atoms
=============

The :class:`~gpyumd.atoms.GpumdAtoms` class also supports sorting the atoms by atom type or by group. This simply
changes the order of the atoms when they are output to an `xyz.in`_ file. We can order our graphene-MoS\ :sub:`2`
structure from the previous section by type using the following:
::
g_mos2_hetero.sort_atoms(sort_key="type", order=["Mo", "S", "C"])

In this case, the ``order`` parameter takes an ordered list of the atomic symbols in the
:class:`~gpyumd.atoms.GpumdAtoms` object.

We can order our graphene nanoribbon structure by group using the following:
::
gnr.sort_atoms(sort_key="group", order=[0,1,8,2,3,4,5,6,7], group_method=0)

In this case, the ``order`` parameter takes an ordered list of integers that directly correspond to the group numbers
of the :class:`~gpyumd.atoms.GpumdAtoms` object. Note that we also have to specify the group method as a
:class:`~gpyumd.atoms.GpumdAtoms` object may have many different groups.

Warning: In the `xyz.in`_ file, the atoms that belong to a single potential must be grouped together although not in
any particular order. For example, if you use the REBO_ potential for "Mo" and "S" interactions and the Tersoff_
potential for "C" interactions, then an order of ["Mo", "C", "S"] is not valid. If you use the
:class:`~gpyumd.sim.Simulation` class to build your GPUMD_ simulation, the ordering is automatically corrected to be
valid.

File Output
===========

The :class:`~gpyumd.atoms.GpumdAtoms` class can also directly write relevant GPUMD_ files. The most common file to write
is the `xyz.in`_, which can be output by using the following:
::
gnr.write_gpumd()

The :meth:`~gpyumd.atoms.GpumdAtoms.write_gpumd` method has the optional arguments ``use_velocity``, which is dictates
whether or not the velocities stored by the :class:`~gpyumd.atoms.GpumdAtoms` object will be written to the `xyz.in`_
file, ``gpumd_file``, which specifies the file name (xyz.in by default), and ``directory``, which notes where the file
should be output. All other GPUMD-specific properties will be automatically handled by the
:class:`~gpyumd.atoms.GpumdAtoms` object.

The other files that the :class:`~gpyumd.atoms.GpumdAtoms` class can write are `basis.in`_ and `kpoints.in`_, which are
required when using the compute_phonon_ keyword. Their usage can best be shown through an example where we prepare Si
for a phonon dispersion calculation.
::

    >>> from ase.lattice.cubic import Diamond
    >>> from ase.build import bulk
    >>> from gpyumd.atoms import GpumdAtoms
    >>> a=5.434  # Lattice constant
    >>> Si_UC = GpumdAtoms(bulk('Si', 'diamond', a=a))
    >>> Si_UC.add_basis()  # gpyumd method
    >>> Si_UC
    GpumdAtoms(symbols='Si2', pbc=True, cell=[[0.0, 2.717, 2.717], [2.717, 0.0, 2.717], [2.717, 2.717, 0.0]])
    >>> Si = Si_UC.repeat([2,2,1])  # Create 8 atom diamond structure
    >>> Si.set_cell([a, a, a])
    >>> Si.wrap()
    >>> Si = Si.repeat([2,2,2])  # Complete full supercell
    >>> Si.set_max_neighbors(4)  # gpyumd method
    >>> Si.set_cutoff(3)  # gpyumd method
    >>> Si
    GpumdAtoms(symbols='Si64', pbc=True, cell=[10.868, 10.868, 10.868])
    >>> Si.write_basis()  # create kpoints.in file
    >>> linear_path, sym_points, labels = Si_UC.write_kpoints(path='GXKGL',npoints=400)  # create kpoints.in file

The code above creats a Si unit cell and uses the :meth:`gpyumd.atoms.GpumdAtoms.add_basis` method to assign a basis
index for each atom as is required for the `basis.in`_ file. The :meth:`gpyumd.atoms.GpumdAtoms.repeat` method, which
is an overridden method from the Atoms_ class in ASE, is then used to create a supercell while keeping the relevant
GPUMD data. Finally, we write the `basis.in`_ file with :meth:`gpyumd.atoms.GpumdAtoms.write_basis` and the
`kpoints.in`_ file with :meth:`gpyumd.atoms.GpumdAtoms.write_kpoints`.

.. _compute_phonon: https://gpumd.zheyongfan.org/index.php/The_compute_phonon_keyword
.. _kpoints.in: https://gpumd.zheyongfan.org/index.php/The_kpoints.in_input_file
.. _basis.in: https://gpumd.zheyongfan.org/index.php/The_basis.in_input_file
.. _Tersoff: https://gpumd.zheyongfan.org/index.php/The_Tersoff-1988_potential
.. _REBO: https://gpumd.zheyongfan.org/index.php/The_REBO-LJ_potential_for_Mo-S_systems
.. _gpyumd: https://github.com/AlexGabourie/gpyumd
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