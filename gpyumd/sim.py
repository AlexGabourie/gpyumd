import copy
import math
import os
from ase import Atoms
from gpyumd.atoms import GpumdAtoms
from gpyumd.keyword import Ensemble, Keyword, RunKeyword
from gpyumd.util import create_directory

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


# TODO make a simulation set that enables multiple simulations to be tracked
class Simulation:

    # TODO add potentials --> not to be used as a standard keyword

    def __init__(self, gpumd_atoms, directory='.'):
        """
        Stores the relevant information for a full 'gpumd' executable simulation.

        Args:
            gpumd_atoms: GpumdAtoms (or ase.Atoms)
                Final structure to be used with the GPUMD simulation.
            directory: string
                Directory of the simulation.
        """
        self.directory = create_directory(directory)
        self.runs = list()
        self.static_calc = None
        if not isinstance(gpumd_atoms, Atoms) or not isinstance(gpumd_atoms, GpumdAtoms):
            raise ValueError("The 'atoms' parameter must be of ase.Atoms or GpumdAtoms type.")
        self.atoms = GpumdAtoms(gpumd_atoms)
        self.potentials = list()

    def create_simulation(self, copy_potentials=False):
        """
        Generates the required files for the gpumd simulation
        :return:
        """
        self.validate_runs()
        with open(os.path.join(self.directory, 'run.in'), 'w') as runfile:
            # TODO write potentials
            if self.static_calc:
                static_calc_lines = self.static_calc.get_output()
                for line in static_calc_lines:
                    runfile.write(f"{line}\n")
            for run in self.runs:
                runlines = run.get_output()
                for line in runlines:
                    runfile.write(f"{line}\n")

        # Write atoms
        # self.atoms.write_gpumd()

        # TODO copy potentials (if selected)

    def add_run(self, number_of_steps=None):
        """
        Adds a new run to a simulation.

        Args:
            number_of_steps:

        Returns:
            A reference to the new run in the simulation.
        """
        # initialize new runs here to ensure that the same atoms object is used.
        current_run = Run(self.atoms, number_of_steps=number_of_steps)
        self.runs.append(current_run)

        # Propagate time steps
        dt_in_fs = None
        if len(self.runs) > 2:
            dt_in_fs = self.runs[-2].get_dt_in_fs()
        self.runs[-1].set_dt_in_fs(dt_in_fs)
        return current_run

    def validate_runs(self):
        first_run_checked = False
        for run in self.runs:
            if not first_run_checked:  # denotes first run
                run.set_first_run()
                first_run_checked = False
            # TODO add try/catch here?
            run.validate_run()

    def add_static_calc(self, keyword):
        if not self.static_calc:
            self.static_calc = StaticCalc()
        self.static_calc.add_calc(keyword)


class StaticCalc:

    def __init__(self):
        """
        Stores the list of static calculation keywords used in the run.in file. These keywords come after potential, but
        before the runs.
        """
        self.keywords = dict()

    def add_calc(self, keyword):
        if not issubclass(type(keyword), Keyword):
            raise ValueError("The 'keyword' parameter must be of the Keyword class or of its children.")
        if keyword.keyword not in ['compute_cohesive', 'compute_elastic', 'compute_phonon', 'minimize']:
            raise ValueError("Only 'compute_cohesive', 'compute_elastic', 'compute_phonon', 'minimize' keywords "
                             "are allowed.")
        self.keywords[keyword.keyword] = keyword

    def get_output(self, minimize_first=True):
        keywords = copy.deepcopy(self.keywords)
        output = list()
        if minimize_first and 'minimize' in keywords:
            keyword = keywords.pop('minimize', None)
            output.append(keyword.get_entry())
        for key in keywords:
            keyword = keywords.pop(key, None)
            output.append(keyword.get_entry())


# TODO enable atoms to be updated and then have all the runs be re-validated
# TODO enable multiple static computations in a single run
class Run:

    def __init__(self, gpumd_atoms, number_of_steps=None):
        """

        Args:
            gpumd_atoms: GpumdAtoms

            number_of_steps: int
                Number of steps to be run in the Run.
        """
        if not isinstance(gpumd_atoms, GpumdAtoms):
            raise ValueError("gpumd_atoms must be of the GpumdAtoms type.")
        self.atoms = gpumd_atoms
        self.keywords = dict()
        self.dt_in_fs = None
        self.run_keyword = None
        if number_of_steps:
            self.run_keyword = RunKeyword(number_of_steps=number_of_steps)
        self.first_run = False

    def get_output(self):
        keywords = copy.deepcopy(self.keywords)
        output = list()
        if 'time_step' in keywords:
            keyword = keywords.pop('time_step', None)
            output.append(keyword.get_entry())
        for key in keywords:
            keyword = keywords.pop(key, None)
            output.append(keyword.get_entry())

    def set_first_run(self, first_run=True):
        self.first_run = first_run

    def set_dt_in_fs(self, dt_in_fs=None):
        if not dt_in_fs:
            dt_in_fs = 1  # 1 fs default
        self.dt_in_fs = dt_in_fs

    def get_dt_in_fs(self):
        return self.dt_in_fs

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

        # Do not allow static calculations except minimize
        if keyword.keyword in ['compute_cohesive', 'compute_elastic', 'compute_phonon']:
            print(f"The {keyword.keyword} keyword is not allowed in a run. It is a static calculation.\n")
            return

        self.validate_keyword(keyword, final_check)
        self.keywords[keyword.keyword] = keyword

    def validate_keyword(self, keyword, final_check=False):
        if keyword.keyword == 'time_step':
            if 'time_step' in self.keywords:
                print("Warning: only one 'time_step' allowed per Run. Previous will be overwritten.")
            self.dt_in_fs = keyword.dt_in_fs  # update for propagation

        if keyword.keyword == 'run' and self.run_keyword:
            print("Warning: Only one 'run' keyword allowed per Run. "
                  "If adding this keyword, the previous will be overwritten.")

        # check for all grouped keywords except 'fix'
        if keyword.grouping_method is not None and (self.atoms.num_group_methods - 1) < keyword.grouping_method:
            raise ValueError(f"The selected grouping method for {keyword.keyword} is larger than the number of grouping"
                             f" methods avaialbe in the GpumdAtoms object.")

        # check for all grouped keywords except 'fix' and 'compute'
        if keyword.grouping_method is not None and keyword.group_id is not None:
            if keyword.group_id >= self.atoms.group_methods[keyword.grouping_method].num_groups:
                raise ValueError(f"The group_id listed for {keyword.keyword} is too large for the grouping method "
                                 f"{keyword.grouping_method}.")

        if keyword.keyword == 'fix':
            if self.atoms.num_group_methods == 0:
                raise ValueError(f"At least one grouping method is required for the {keyword.keyword} keyword.")
            if keyword.group_id >= self.atoms.group_methods[0].num_groups:
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

                if self.atoms.group_methods[0].num_groups <= keyword.ensemble.source_group_id or \
                        self.atoms.group_methods[0].num_groups <= keyword.ensemble.sink_group_id:
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
                if 1e3 / (self.dt_in_fs * keyword.keyword.sample_interval) < keyword.keyword.max_omega / math.pi:
                    raise ValueError("Sampling rate is less than the Nyquist rate.")

    def validate_run(self):
        for key in self.keywords.keys():
            self.validate_keyword(self.keywords[key], final_check=True)

        if 'run' not in self.keywords:
            raise ValueError(f"No 'run' keyword provided for this run.")

        if self.first_run and 'velocity' not in self.keywords:
            raise ValueError("A 'velocity' keyword must be used before any 'run' keyword. See "
                             "https://gpumd.zheyongfan.org/index.php/The_velocity_keyword for details.")
