=====================
Building a Simulation
=====================

This module is designed to help GPUMD_ users create a valid simulation for the gpumd executable. The
:class:`~gpyumd.sim.Simulation` class is the main class that facilitates this setup. It will create a `run.in`_ file
with a structure that follows the guidlines defined in the gpumd documentation.

1. Setup the potentials
2. Minimize system (if needed)
3. Setup static calculations (cohesive, elastic, phonon)
4. Run molecular dynamics simulation (if desired)

The :class:`~gpyumd.sim.Simulation` class will also copy potential files to the correct working directory, create the
appropriate `xyz.in`_ file, and ensure that the files are consistent with the rules for GPUMD_.

At a high level, to build a GPUMD simulation using gpyumd, users first need to instantiate a
:class:`~gpyumd.sim.Simulation` object and then add potentials, static calculations, and runs (with any supported
keywords) to that :class:`~gpyumd.sim.Simulation` object. To add potentials, users must pass
:class:`~gpyumd.keyword.Potential` objects to the :meth:`~gpyumd.sim.Simulation.add_potential` method. To add static
calculations, users must pass :class:`~gpyumd.keyword.Keyword` objects for static calculations as defined in the
keywords_ section of the GPUMD documentation. As of this writing, these are the :class:`~gpyumd.keyword.Minimize`,
:class:`~gpyumd.keyword.ComputeCohesive`, :class:`~gpyumd.keyword.ComputeElastic`, and
:class:`~gpyumd.keyword.ComputePhonon` keywords. For the molecular dynamics part of a simulation, users create a series
of :class:`~gpyumd.sim.Run` objects using the :meth:`~gpyumd.sim.Simulation.add_run` method. Each
:class:`~gpyumd.sim.Run` can be constructed using a set of keywords_ (non-static calculations) found in the
:mod:`~gpyumd.keyword` module.

We will run through two examples here better illustrate how to build a :class:`~gpyumd.sim.Simulation` using gpyumd.
The first example is an equlibrium MD simulation that calculates the thermal conductivity of graphene and demonstrates
how to construct a :class:`~gpyumd.sim.Run`. The second is a phonon calculation of Si, which demonstrates how to add
static calculations.

Example: thermal conductivity of graphene
=========================================

In this example, we will setup a simulation for an EMD thermal conductivity calculation for graphene using the
:class:`~gpyumd.sim.Simulation` class. First, we will setup the graphene structure (``gr``) using the
:class:`~gpyumd.atoms.GpumdAtoms` class.
::

    # import statements
    from ase.build import graphene_nanoribbon
    from gpyumd.atoms import GpumdAtoms
    from gpyumd.sim import Simulation
    import gpyumd.keyword as kwd

    # structure creation
    gr = GpumdAtoms(graphene_nanoribbon(60, 36, type='armchair', sheet=True, vacuum=3.35/2, C_C=1.44))
    gr.euler_rotate(theta=90)
    lx, lz, ly = gr.cell.lengths()
    gr.cell = gr.cell.new((lx, ly, lz))
    gr.center()
    gr.pbc = [True, True, False]
    gr.set_cutoff(2.1)
    gr.set_max_neighbors(3)

With a structure created and finalized, we can begin creating a simulation with:
::

    emd_sim = Simulation(gnr, driver_directory='.')

The ``driver_directory`` parameter determines where `driver file`_ will be located. This can be important for the path
definitions of potentials, so be sure that it is correct.

Now that we have a :class:`~gpyumd.sim.Simulation` object defined, we want to add a :class:`~gpyumd.sim.Run`. A
:class:`~gpyumd.sim.Run` is simply a group of keywords that are coupled to a specific ``run`` keyword in the `run.in`_
file. The first run we want to create is an equilibration and we can do this using the following:
::

    header_comment = "Equilibration"  # A comment that will go above the keywords in the run
    curr_run = emd_sim.add_run(number_of_steps=1e6, run_name='equilibration', run_header=header_comment)

Note that all of the parameters for the :mod:`~gpyumd.sim.Simulation.add_run` method are optional; however, if you do
not provide the ``number_of_steps`` parameter, you will have to add the :class:`~gpyumd.keyword.RunKeyword` keyword to
the :class:`~gpyumd.sim.Run` object.

In the equilibration, we want to allow the structure to relax to the appropriate size for the temperature. To do this,
we need to define an integrator suitable for the purpose using the ensemble_ keyword. In gpyumd, we define the
:class:`~gpyumd.keyword.Ensemble` keyword through two steps. First, we instantiate :class:`~gpyumd.keyword.Ensemble`
keyword object, defining one of the many types as outlined in the ensemble_ documentation. Next, we set the relevant
parameters for the type of ensemble using either the :meth:`~gpyumd.keyword.Ensemble.set_nvt_parameters`,
:meth:`~gpyumd.keyword.Ensemble.set_npt_parameters`, or :meth:`~gpyumd.keyword.Ensemble.set_heat_parameters` methods.
(Note that NVE ensembles need no other parameters to be set.)


In this example, we want constant atom number, pressure, and temperature, i.e., an NPT ensemble. To handle the extra
complexity for NPT ensembles, gpyumd requires the user to know if their simulation box is orthogonal or triclinic.
Then the user can ask gpyumd for the appropriate pressure and elastic moduli parameters needed, using the
:meth:`gpyumd.keyword.Ensemble.get_npt_pdict` method, by specifying if the conditions are 'isotropic', 'orthogonal',
or 'triclinic'. The processs for an 'orthogonal' box is shown for this example here:
::

    # Create ensemble for the equilibration
    npt_ensemble = kwd.Ensemble(ensemble_method='npt_ber')  # step 1 for ensemble creation

    # Determine the NPT conditions - special, extra definitions for NPT ensembles
    npt_condition = 'orthogonal'
    pdict = kwd.Ensemble.get_npt_pdict(condition=npt_condition)
    pdict['p_xx'], pdict['p_yy'], pdict['p_zz'] = [0]*3  # GPa
    pdict['C_xx'], pdict['C_yy'], pdict['C_zz'] = [53.4059]*3  # GPa

    # step 2 for ensemble creation
    npt_ensemble.set_npt_parameters(initial_temperature=300, final_temperature=300, thermostat_coupling=100,
                                    barostat_coupling=2000, condition=npt_condition, pdict=pdict)

With the NPT ensemble defined, we can define the remaining _keywords, which have much simpler definitions, with the
following:
::

    keywords = [
        kwd.Velocity(initial_temperature=300),
        npt_ensemble,
        kwd.TimeStep(dt_in_fs=1),
        kwd.NeighborOff(),
        kwd.DumpThermo(1000)
    ]

    for keyword in keywords:
        curr_run.add_keyword(keyword)

If you are working in an interactive environment like Jupyter, you can take a quick look at your run by simply running:
``curr_run`` in a cell. Here, this would output:

.. code-block:: text

    Run: 'equilibration'
    # Equilibration
      Velocity(initial_temperature=300)
      Ensemble(npt_ber=NPT(initial_temperature=300, final_temperature=300, thermostat_coupling=100,
        barostat_coupling=2000, condition=orthogonal, pdict=...))
      TimeStep(dt_in_fs=1, max_distance_per_step=None)
      NeighborOff()
      DumpThermo(interval=1000)
    run 1000000 steps, dt_in_fs=1 -> 1.0 ns

In the next part of the simulation, the production run, we want to shift to an NVE ensemble and compute the heat current
autocorrelation. This can be done with the following:
::

    # Production run
    header_comment = "Production"
    curr_run = emd_sim.add_run(run_name='production', run_header=header_comment)

    # Timestep is propagated from last run
    keywords = [
        kwd.Ensemble(ensemble_method='nve'),
        kwd.NeighborOff(),
        kwd.ComputeHAC(sample_interval=20, num_corr_steps=50000, output_interval=10),
        kwd.RunKeyword(number_of_steps=1e7)  # Can add run here instead of during add_run fucntion
    ]

    for keyword in keywords:
        curr_run.add_keyword(keyword)

Finally, we need to add the potential to describe our Graphene. Note that we can add this at any point in the
simulation setup. To add the potential, we need to create a :class:`~gpyumd.keyword.Potential` keyword object and pass
it to the simulation through the :meth:`~gpyumd.sim.Simulation.add_potential` method like in the following code:
::

    potential_directory = "path/to/GPUMD/potentials/tersoff"
    tersoff_potential = \
        kwd.Potential(filename='Graphene_Lindsay_2010_modified.txt', symbols=['C'], directory=potential_directory)
    emd_sim.add_potential(tersoff_potential)

The simulation definition is now complete and can be created using the following:
::

    emd_sim.create_simulation(copy_potentials=True)

Note that the ``copy_potentials`` parameter determines whether or not you want to copy the potential file to your
simulation directory or if you want to use relative paths to the potential files in the run.in file. After the
:meth:`~gpyumd.sim.Simulation.create_simulation` method call, the simulation object generates the `run.in`_ file and the
`xyz.in`_ file and moves the potential to the simulation directory. With the keywords and parameters we used here, the
`run.in`_ file's contents look like:

.. code-block:: text

    potential Graphene_Lindsay_2010_modified.txt 0

    # Equilibration
    time_step 1
    velocity 300
    ensemble npt_ber 300 300 100 0 0 0 53.4059 53.4059 53.4059 2000
    neighbor off
    dump_thermo 1000
    run 1000000

    # Production
    ensemble nve
    neighbor off
    compute_hac 20 50000 10
    run 10000000

Example: phonon dispersion of silicon
=====================================

In this example, we will setup a phonon dispersion calculation of silicon (Si) using the :class:`~gpyumd.sim.Simulation`
class. This example emphasizes the inclusion of static calculations when building a :class:`~gpyumd.sim.Simulation`.
First, we will set up the Si structure. For more information about the structure generation, see :doc:`atoms`.
::

    # import statements
    from ase.lattice.cubic import Diamond
    from ase.build import bulk
    from gpyumd.atoms import GpumdAtoms
    from gpyumd.sim import Simulation
    import gpyumd.keyword as kwd

    # Create unit cell
    a=5.434
    Si_UC = GpumdAtoms(bulk('Si', 'diamond', a=a))
    Si_UC.add_basis()

    # Create 8 atom diamond structure
    Si = Si_UC.repeat([2,2,1])
    Si.set_cell([a, a, a])
    Si.wrap()

    # Complete full supercell
    Si = Si.repeat([2,2,2])
    Si.set_max_neighbors(4)
    Si.set_cutoff(3)

Next, we need to create the relevant extra files needed for the phonon dispersion calculation: `kpoints.in`_ and
`basis.in`_. (In the future, these files may also be handled by the :class:`~gpyumd.sim.Simulation` class.)
::

    Si.write_basis()
    Si_UC.write_kpoints(path='GXKGL',npoints=400)

We can now create the :class:`~gpyumd.sim.Simulation` object that we want to work with:
::

    phonon_sim = Simulation(Si, driver_directory='.')

We want to calculate the phonon dispersion, so we add the :class:`~gpyumd.keyword.ComputePhonon` keyword to the
`phonon_sim` object using the :meth:`~gpyumd.sim.Simulation.add_static_calc` method.
::

    phonon_sim.add_static_calc(kwd.ComputePhonon(cutoff=5, displacement=0.005))

Next, we need to define the potential that describes the Si interactions:
::

    potential_directory = "/path/to/GPUMD/potentials/tersoff"
    tersoff_potential = \
        kwd.Potential(filename='Si_Fan_2019.txt', symbols=['Si'], directory=potential_directory)
    phonon_sim.add_potential(tersoff_potential)

Finally, we can generate the `run.in`_ file and the `xyz.in`_ file and move the potential file to the simulation
directory using:
::

    phonon_sim.create_simulation(copy_potentials=True)

The resulting `run.in`_ file has the contents:

.. code-block:: text

    potential Si_Fan_2019.txt 0

    compute_phonon 5 0.005


.. _driver file: https://gpumd.zheyongfan.org/index.php/Main_Page#The_driver_input_file
.. _xyz.in: https://gpumd.zheyongfan.org/index.php/The_xyz.in_input_file
.. _run.in: https://gpumd.zheyongfan.org/index.php/Main_Page#Inputs_for_the_src.2Fgpumd_executable
.. _keywords: https://gpumd.zheyongfan.org/index.php/Main_Page#Inputs_for_the_src.2Fgpumd_executable
.. _GPUMD: https://github.com/brucefan1983/GPUMD
.. _ensemble: https://gpumd.zheyongfan.org/index.php/The_ensemble_keyword
.. _kpoints.in: https://gpumd.zheyongfan.org/index.php/The_kpoints.in_input_file
.. _basis.in: https://gpumd.zheyongfan.org/index.php/The_basis.in_input_file

List of all methods
===================

.. automodule:: gpyumd.sim
    :members:
    :undoc-members:
    :inherited-members:
    :exclude-members: set_first_run
