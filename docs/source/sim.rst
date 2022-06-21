=====================
Building a Simulation
=====================

This module is designed to help GPUMD_ users create a valid simulation with the gpumd executable. The
:class:`~gpyumd.sim.Simulation` class is the main class that facilitates this setup. It will create a `run.in`_ file
with a structure that follows the guidlines defined in the gpumd documentation.

1. Setup the potentials
2. Minimize system (if needed)
3. Setup static calculations (cohesive, elastic, phonon)
4. Run molecular dynamics simulation (if desired)

The :class:`~gpyumd.sim.Simulation` class will also copy potential files to the correct working directory, create the
appropriate `xyz.in`_ file, and ensure that the files are consistent with the rules for GPUMD_.

We will run through two examples here to show how this class works. The first example is an equlibrium MD simulation
that calculates the thermal conductivity of graphene, the second is a phonon calculation of Si.

Example Simulation: EMD
=======================

In this example, we will setup a simulation for an EMD thermal conductivity calculation for graphene using the
:class:`~gpyumd.sim.Simulation` class. First, we will setup the graphene structure (``gr``) using the
:class:`~gpyumd.atoms.GpumdAtoms` class.
::

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

    emd_sim = Simulation(gnr, driver_directory='..')

Now that we have a :class:`~gpyumd.sim.Simulation` object defined, we want to add a :class:`~gpyumd.sim.Run`. A
:class:`~gpyumd.sim.Run` is simply a group of keywords that are coupled to a specific ``run`` keyword in the `run.in`_
file. The first run we want to create is an equilibration.
::

    header_comment = "Equilibration"  # A comment that will go above the keywords in the run
    curr_run = emd_sim.add_run(number_of_steps=1e6, run_name='equilibration', run_header=header_comment)

In the equilibration, we want to allow the structure to relax to the appropriate size for the temperature. !!!!!!! NOT DONE

::

    # Create ensemble for the equilibration
    npt_condition = 'orthogonal'
    pdict = kwd.Ensemble.get_npt_pdict(condition=npt_condition)
    pdict['p_xx'], pdict['p_yy'], pdict['p_zz'] = [0]*3  # GPa
    pdict['C_xx'], pdict['C_yy'], pdict['C_zz'] = [53.4059]*3  # GPa

    npt_ensemble = kwd.Ensemble(ensemble_method='npt_ber')
    npt_ensemble.set_npt_parameters(initial_temperature=300, final_temperature=300, thermostat_coupling=100,
                                    barostat_coupling=2000, condition=npt_condition, pdict=pdict)

    keywords = [
        kwd.Velocity(initial_temperature=300),
        npt_ensemble,
        kwd.TimeStep(dt_in_fs=1),
        kwd.NeighborOff(),
        kwd.DumpThermo(1000)
    ]

    for keyword in keywords:
        curr_run.add_keyword(keyword)

.. _xyz.in: https://gpumd.zheyongfan.org/index.php/The_xyz.in_input_file
.. _run.in: https://gpumd.zheyongfan.org/index.php/Main_Page#Inputs_for_the_src.2Fgpumd_executable
.. _GPUMD: https://github.com/brucefan1983/GPUMD

.. automodule:: gpyumd.sim
    :members:
    :undoc-members:
    :inherited-members:
