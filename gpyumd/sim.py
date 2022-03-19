__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"

import math
import operator as op
from gpyumd.keyword import TimeStep, Ensemble, Keyword, RunKeyword
from gpyumd.util import cond_assign_int
from gpyumd.atoms import GpumdAtoms


class Simulation:
    """
    Stores the relevant information for a full 'gpumd' executable simulation.
    """

    # TODO Add runs
    # TODO add attach xyz file
    # TODO add potentials --> not to be used as a standard keyword
    # TODO add working directory

    def __init__(self, atoms, directory='.'):
        self.directory = directory
        self.runs = list()
        self.atoms = atoms
        self.potentials = list()

    def create_simulation(self):
        """
        Generates the required files for the gpumd simulation
        :return:
        """
        pass

    def add_run(self, dt_in_fs=None, number_of_steps=None):
        self.runs.append(Run(self.atoms, dt_in_fs=dt_in_fs, number_of_steps=number_of_steps))
        # TODO propagate time step - Add a time step as a simulation parameter?
        pass

    def validate_simulation(self):
        first_run_checked = False
        for run in self.runs:
            if not run.get_immediate_action() and not first_run_checked:  # denotes first run
                run.set_first_run()
                first_run_checked = False
            # TODO add try/catch here?
            run.validate_run()


class Run:

    def __init__(self, gpumd_atoms, dt_in_fs=None, number_of_steps=None):
        """

        Args:
            gpumd_atoms: GpumdAtoms

            dt_in_fs: float
                Time step of simulation in fs. Default is 1 fs.
        """
        if not isinstance(gpumd_atoms, GpumdAtoms):
            raise ValueError("gpumd_atoms must be of the GpumdAtoms type.")
        self.atoms = gpumd_atoms
        self.keywords = dict()
        if not dt_in_fs:
            dt_in_fs = 1
        self.time_step = TimeStep(dt_in_fs=dt_in_fs)
        self.run_keyword = None
        if number_of_steps:
            self.run_keyword = RunKeyword(number_of_steps=number_of_steps)
        self.accepting_immediate_actions = True
        self.immediate_action = False
        self.first_run = False

    def set_first_run(self, first_run=True):
        self.first_run = first_run

    def get_immediate_action(self):
        return self.immediate_action

    # TODO add a warning if a keyword will not have an output during a run (i.e. output interval is too large)

    def add_keyword(self, keyword, final_check=False):
        """
        Adds a keyword object to the run. Verifies that the keyword is valid (to the extent that it can be initially).

        Args:
            keyword: Keyword
                The keyword to add to the run.

            final_check: bool
                Use only if you know what you're doing. It is normally used to validate the run when finalized.

        Returns:

        """
        if not issubclass(type(keyword), Keyword):
            raise ValueError("The 'keyword' parameter must be of the Keyword class or of its children.")

        if keyword.keyword in ['compute_cohesive', 'compute_elastic', 'compute_phonon', 'minimize']:
            if not self.accepting_immediate_actions:
                print(f"Keywords that run immediately must be in their own run. {keyword.keyword} will not be added to "
                      f"this Run")
                return
            else:
                self.immediate_action = True

        self.validate_keyword(keyword, final_check)
        self.keywords[keyword.keyword] = keyword

        # Immediate actions not allowed after the first keyword in a run
        self.accepting_immediate_actions = False

    def validate_keyword(self, keyword, final_check=False):
        if keyword.keyword == 'run' and self.run_keyword:
            print("Warning: Only one 'run' keyword allowed per Run. Previous will be overwritten.")

        # check for all grouped keywords except 'fix'
        if keyword.grouping_method is not None and (self.atoms.num_group_methods - 1) < keyword.grouping_method:
            raise ValueError(f"The selected grouping method for {keyword.keyword} is larger than the number of grouping"
                             f" methods avaialbe in the GpumdAtoms object.")

        # check for all grouped keywords except 'fix' and 'compute'
        if keyword.grouping_method is not None and keyword.group_id is not None:
            if keyword.group_id >= self.atoms.groups[keyword.grouping_method].num_groups:
                raise ValueError(f"The group_id listed for {keyword.keyword} is too large for the grouping method "
                                 f"{keyword.grouping_method}.")

        if keyword.keyword == 'fix':
            if self.atoms.num_group_methods == 0:
                raise ValueError(f"At least one grouping method is required for the {keyword.keyword} keyword.")
            if keyword.group_id >= self.atoms.groups[0].num_groups:
                raise ValueError(f"The group_id given for {keyword.keyword} is too large for grouping method 0.")

        # Check for heating ensembles
        if keyword.keyword == 'ensemble':
            if 'ensemble' in self.keywords.keys():
                print(f"The 'ensemble' keyword has already been used in this run. Previous ensemble will be "
                      f"overwritten.")

            if not keyword.ensemble.parameters_set:
                raise ValueError(f"Cannot add an ensemble before its parameters are set. "
                                 f"See 'set_parameters' function.")

            if keyword.ensemble_type == 'heat':
                if self.atoms.num_group_methods == 0:
                    raise ValueError(f"At least one grouping method is required for the {keyword.keyword} "
                                     f"{keyword.ensemble_method} keyword.")

                if self.atoms.groups[0].num_groups <= keyword.ensemble.source_group_id or \
                        self.atoms.groups[0].num_groups <= keyword.ensemble.sink_group_id:
                    raise ValueError(f"The source or sink group is too large for grouping method 0.")

            if keyword.ensemble_type == 'npt':
                if keyword.ensemble.condition == 'isotropic':
                    if self.atoms.triclinic:
                        raise ValueError("Cannot use 'isotropic' pressure with triclinic atoms.")
                    if any([not bc for bc in self.atoms.get_pbc()]):
                        raise ValueError("Cannot use isotropic pressure with non-periodic boundary in any direction.")

                if keyword.ensemble.condition == 'orthogonal' and self.atoms.triclinic:
                    raise ValueError("Cannot use triclinic atoms cel with the orthogonal npt conditions.")

                if keyword.ensemble.condition == 'triclinic':
                    if not self.atoms.triclinic:
                        raise ValueError("Atoms must have a triclinic cell to use the triclinic npt parameters.")
                    if any([not bc for bc in self.atoms.get_pbc()]):
                        raise ValueError("Cannot use isotropic pressure with non-periodic boundary in any direction.")

        if keyword.keyword in ['compute_hnema', 'compute_gkma']:
            if keyword.last_mode > 3 * len(self.atoms):
                raise ValueError(f"Last mode for {keyword.keyword} keyword must be no greater than "
                                 f"3*(number of atoms).")

        if final_check:
            if keyword.keyword == 'compute_hnemd' or keyword.keyword == 'compute_hnema':
                # FIXME may not need this check if we do it elsewhere
                if 'ensemble' not in self.keywords.keys():
                    raise ValueError(f"The {keyword.keyword} keyword requires an NVT or NPT ensemble be defined.")

                ensemble = self.keywords['ensemble']
                if 'lan' in ensemble.ensemble_method:
                    raise ValueError("Langevin thermostat not allowed for the 'compute_hnemd' keyword.")
                if not (isinstance(type(ensemble.ensemble), type(Ensemble.NPT)) or
                        isinstance(type(ensemble.ensemble), type(Ensemble.NVT()))):
                    raise ValueError(f"An NVT or NPT ensemble is needed for the {keyword.keyword} keyword.")

            if keyword.keyword == 'compute_dos':
                if 1e3 / (self.time_step.dt_in_fs * keyword.keyword.sample_interval) < keyword.keyword.max_omega / math.pi:
                    raise ValueError("Sampling rate is less than the Nyquist rate.")

            if  keyword.keyword == 'run' and not self.immediate_action:
                raise ValueError(f"No 'run' keyword provided for this run.")

    def validate_run(self):
        for key in self.keywords.keys():
            self.validate_keyword(self.keywords[key], final_check=True)

        if self.first_run and 'velocity' not in self.keywords:
            raise ValueError("A 'velocity' keyword must be used before any 'run' keyword. See "
                             "https://gpumd.zheyongfan.org/index.php/The_velocity_keyword for details.")
