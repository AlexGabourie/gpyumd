import copy
import math
import os
import shutil
from typing import List

from gpyumd.atoms import GpumdAtoms
from gpyumd.keyword import Ensemble, Keyword, RunKeyword, Potential
import gpyumd.util as util

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


# TODO make a simulation set that enables multiple simulations to be tracked
# TODO make simulation handle the extra files needed by some keywords (i.e., basis, kpoints)
class Simulation:

    def __init__(self, gpumd_atoms: GpumdAtoms, run_directory: str = None, driver_directory: str = None):
        """
        Stores the relevant information for a full 'gpumd' executable
        simulation.

        Args:
            gpumd_atoms: Final structure to be used with the GPUMD
             simulation.
            run_directory: Directory of the simulation.
            driver_directory: Directory where the driver input file will
             be found. Defaults to run_directory.
        """
        if run_directory:
            util.create_directory(run_directory)
            self.directory = run_directory
        else:
            self.directory = os.getcwd()
        self.driver_directory = driver_directory
        if driver_directory:
            util.create_directory(driver_directory)
            self.driver_directory = driver_directory
        else:
            self.driver_directory = self.directory

        self.runs = list()
        self._runs_dict = dict()
        self.static_calc = None
        if not isinstance(gpumd_atoms, GpumdAtoms):
            raise ValueError("The 'gpumd_atoms' parameter must be of GpumdAtoms type.")
        self.atoms = copy.deepcopy(gpumd_atoms)
        self.potentials = None

    def create_simulation(self, copy_potentials: bool = True, use_velocity: bool = False) -> None:
        """
        Generates the required files for the gpumd simulation

        Args:
            copy_potentials: Whether or not to copy potentials to the
             simulation directory. If False, relative path is provided.
            use_velocity: Whether or not to add velocities to the xyz.in file
        """
        self.validate_potentials()
        self.validate_runs()
        with open(os.path.join(self.directory, 'run.in'), 'w', newline='') as run_file:
            potential_lines = self.potentials.get_output(driver_directory=self.driver_directory,
                                                         sim_directory=self.directory if copy_potentials else None)
            for line in potential_lines:
                run_file.write(f"{line}\n")
            if self.static_calc:
                run_file.write("\n")
                static_calc_lines = self.static_calc.get_output()
                for line in static_calc_lines:
                    run_file.write(f"{line}\n")
            for run in self.runs:
                run_file.write("\n")
                run_lines = run.get_output()
                for line in run_lines:
                    run_file.write(f"{line}\n")

        self.atoms.write_gpumd(use_velocity=use_velocity, directory=self.directory)
        if copy_potentials:
            self.potentials.copy_potentials(self.directory)

    # TODO keep a dictionary that stores the index of runs in runs list, but with keys of the name, easy way
    #   to allow overwriting
    def add_run(self, number_of_steps: int = None, run_name: str = None, run_header: str = None) -> "Run":
        """
        Adds a new run to a simulation.

        Args:
            number_of_steps: number of steps to run
            run_name: A name for the run. Note that runs can share the
             same name.
            run_header: A comment that will be added to the line above
             all keywords for this run

        Returns:
            A reference to the new run in the simulation.
        """
        # initialize new runs here to ensure that the same atoms object is used.
        if run_name is None:
            run_name = f"run{len(self.runs) + 1}"
        current_run = Run(self.atoms, number_of_steps=number_of_steps, run_name=run_name, run_header=run_header)
        if run_name in self._runs_dict:
            print(f"Warning: Overwriting previous run with name '{current_run.name}'")
            self.runs[self._runs_dict[run_name]] = current_run
        else:
            self.runs.append(current_run)
            self._runs_dict[current_run.name] = len(self.runs) - 1

        # Propagate time steps
        dt_in_fs = None
        if len(self.runs) > 1:
            dt_in_fs = self.runs[-2].get_dt_in_fs()
        self.runs[-1].set_dt_in_fs(dt_in_fs)
        return current_run

    def validate_potentials(self) -> None:
        if self.potentials is None:
            raise ValueError("No potentials set.")
        self.potentials.finalize_types()

    def validate_runs(self) -> None:
        first_run_checked = False
        for run in self.runs:
            if not first_run_checked:  # denotes first run
                run.set_first_run()
                first_run_checked = True
            # TODO add try/catch here?
            run.validate_run()

    def add_static_calc(self, keyword: Keyword) -> None:
        """
        Adds a static calculation to the current simulation object.
        Currently, only the 'compute_cohesive', 'compute_elastic',
        'compute_phonon', and 'minimize' keywords are static.

        Args:
            keyword: A keyword object for a static calculation
        """
        if not self.static_calc:
            self.static_calc = StaticCalc()
        self.static_calc.add_calc(keyword)

    def add_potential(self, potential: Potential) -> None:
        """
        Adds a potential keyword to the simulation object.

        Args:
            potential: A potential keyword object
        """
        if not self.potentials:
            self.potentials = Potentials(self.atoms)
        self.potentials.add_potential(potential)


class Potentials:

    potentials: List[Potential]

    def __init__(self, gpumd_atoms: GpumdAtoms):
        self.potentials = list()
        self.gpumd_atoms = gpumd_atoms
        self.type_dict = dict()

    def add_potential(self, potential: Potential) -> None:
        """
        Adds a potential keyword to the current list of potentials.

        Args:
            potential: A potential keyword object
        """
        if not isinstance(potential, Potential):
            raise ValueError("Must add a Potential keyword to the potentials list.")

        if not potential.potential_type == "lj":
            for symbol in potential.symbols:
                if symbol in self.type_dict:
                    raise ValueError(f"The atomic symbol {symbol} has already been accounted for in a potential.")
                if symbol not in set(self.gpumd_atoms.get_chemical_symbols()):
                    raise ValueError(f"The atomic symbol {symbol} is not part of the GpumdAtoms object.")
                util.check_symbols([symbol])
                self.type_dict[symbol] = len(self.type_dict)

        elif potential.grouping_method and (self.gpumd_atoms.num_group_methods - 1) < potential.grouping_method:
            raise ValueError(f"The selected grouping method is larger than the number of grouping"
                             f" methods avaialbe in the GpumdAtoms object.")

        self.potentials.append(potential)

    def finalize_types(self) -> None:
        """
        Sets the type dict for the GpumdAtoms object based on the potentials
         that have been added.
        """
        self.gpumd_atoms.set_type_dict(self.type_dict)
        self.gpumd_atoms.sort_atoms(sort_key='type')
        for potential in self.potentials:
            if not potential.potential_type == 'lj':
                types = list()
                for symbol in potential.symbols:
                    types.append(self.type_dict[symbol])
                potential.set_types(types)

    def get_output(self, driver_directory: str, sim_directory: str = None, ) -> List[str]:
        output = list()
        lj = None
        for potential in self.potentials:
            entry = potential.get_entry_rel_path(driver_directory, sim_directory)
            if potential.potential_type == 'lj':
                lj = entry
            else:
                output.append(entry)
        if lj:  # 'lj' potential is last
            output.append(lj)
        return output

    def copy_potentials(self, directory: str) -> None:
        """
        Copies all of the potentials to the selected directory.

        Args:
            directory: Directory to copy potentials to
        """
        for potential in self.potentials:
            src = os.path.abspath(potential.potential_path)
            dest = os.path.abspath(util.get_path(directory, potential.filename))
            if src == dest:
                continue
            shutil.copy(src, dest)


# TODO add comments like for run
class StaticCalc:

    def __init__(self):
        """
        Stores the list of static calculation keywords used in the run.in
        file. These keywords come after potential, but before the runs.
        """
        self.keywords = dict()

    def add_calc(self, keyword):
        if not issubclass(type(keyword), Keyword):
            raise ValueError("The 'keyword' parameter must be of the Keyword class or of its children.")
        if keyword.keyword not in ['compute_cohesive', 'compute_elastic', 'compute_phonon', 'minimize']:
            raise ValueError("Only 'compute_cohesive', 'compute_elastic', 'compute_phonon', 'minimize' keywords "
                             "are allowed.")
        self.keywords[keyword.keyword] = keyword

    def get_output(self, minimize_first: bool = True) -> List[str]:
        keywords = copy.deepcopy(self.keywords)
        output = list()
        if minimize_first and 'minimize' in keywords:
            keyword = keywords.pop('minimize', None)
            output.append(keyword.get_entry())
        for key in keywords:
            output.append(keywords[key].get_entry())
        return output


# TODO enable atoms to be updated and then have all the runs be re-validated
class Run:

    def __init__(self, gpumd_atoms: GpumdAtoms, number_of_steps: int = None,
                 run_name: str = None, run_header: str = None):
        """

        Args:
            gpumd_atoms: Atoms for the simulation
            number_of_steps: Number of steps to be run in the Run.
            run_name: Name of the run
            run_header: A comment that will be added to the line above
             all keywords for this run
        """
        if not isinstance(gpumd_atoms, GpumdAtoms):
            raise ValueError("gpumd_atoms must be of the GpumdAtoms type.")
        if run_name is not None and not isinstance(run_name, str):
            raise ValueError("run_name must be a string.")
        if run_header is not None and not isinstance(run_header, str):
            raise ValueError("run_header must be a string.")
        self.name = run_name
        self.header = run_header
        self.atoms = gpumd_atoms
        self.keywords = dict()
        self.dt_in_fs = None
        self.run_keyword = None
        if number_of_steps:
            self.run_keyword = RunKeyword(number_of_steps=number_of_steps)
        self.first_run = False

    def clear_run(self) -> None:
        """
        Resets all parameters to their default values.
        """
        self.keywords = dict()
        self.dt_in_fs = None
        self.run_keyword = None
        self.first_run = False

    def __repr__(self):
        out = f"{self.__class__.__name__}: "
        out += f"'{self.name}'\n" if self.name else "\n"
        if self.header:
            out += f"# {self.header}\n"
        if 'velocity' in self.keywords:
            out += f"  {self.keywords['velocity'].__repr__()}\n"
        for keyword in self.keywords:
            out += f"  {self.keywords[keyword].__repr__()}\n" if not keyword == 'velocity' else ""

        if self.run_keyword:
            out += f"{self.run_keyword}steps"
        if self.dt_in_fs:
            if not self.run_keyword:
                out += f"dt_in_fs={self.dt_in_fs}\n"
            else:
                total_time = self.run_keyword.number_of_steps * self.dt_in_fs
                units = ['fs', 'ps', 'ns', 'ms']
                unit_idx = 0
                while total_time / 1e3 >= 1:
                    total_time /= 1e3
                    unit_idx += 1

                    if unit_idx == 3:
                        break

                out += f", dt_in_fs={self.dt_in_fs} -> {total_time} {units[unit_idx]}\n\n"
        return out

    def get_output(self) -> List[str]:
        """
        Gets the run.in, textual representation of the current run.

        Returns:
            The Run object in text format of run.in file
        """
        keywords = copy.deepcopy(self.keywords)
        output = list()
        if self.header:
            output.append(f"# {self.header}")
        if 'time_step' in keywords:
            keyword = keywords.pop('time_step', None)
            output.append(keyword.get_entry())
        for key in keywords:
            output.append(keywords[key].get_entry())
        output.append(self.run_keyword.get_entry())
        return output

    def set_first_run(self, first_run: bool = True) -> None:
        """
        Marks this run as the first in a simulation. Required to ensure
        that the 'velocity' keyword is used.

        Args:
            first_run: Whether or not run is the first in a simulation.
        """
        self.first_run = first_run

    def set_dt_in_fs(self, dt_in_fs: float = None) -> None:
        """
        Sets the timestep for the run in units of femtoseconds.

        Args:
            dt_in_fs: Timestep in femtoseconds.
        """
        if not dt_in_fs:
            dt_in_fs = 1  # 1 fs default
        self.dt_in_fs = dt_in_fs

    def get_dt_in_fs(self) -> float:
        """

        Returns:
            The time step in femtosectonds
        """
        return self.dt_in_fs

    # TODO add a warning if a keyword will not have an output during a run (i.e. output interval is too large)
    def add_keyword(self, keyword: Keyword, final_check: bool = False) -> None:
        """
        Adds a keyword object to the run. Verifies that the keyword is valid
         (to the extent that it can be initially).

        Args:
            keyword: The keyword to add to the run.
            final_check: Use only if you know what you're doing. It is
             normally used to validate the run when finalized.
        """
        if not issubclass(type(keyword), Keyword):
            raise ValueError("The 'keyword' parameter must be of the Keyword class or of its children.")

        # Do not allow static calculations except minimize
        if keyword.keyword in ['compute_cohesive', 'compute_elastic', 'compute_phonon']:
            print(f"The {keyword.keyword} keyword is not allowed in a run. It is a static calculation.\n")
            return

        if keyword.keyword == 'potential':
            print(f"The {keyword.keyword} keyword can only be added via the add_potential function in the Sim class.")
            return

        self._validate_keyword(keyword, final_check)

        if keyword.keyword == 'run':
            self.run_keyword = keyword
        else:
            self.keywords[keyword.keyword] = keyword

    def _validate_keyword(self, keyword, final_check: bool = False) -> None:
        if keyword.keyword == 'time_step':
            if 'time_step' in self.keywords and not final_check:
                print("Warning: only one 'time_step' allowed per Run. Previous will be overwritten.")
            self.dt_in_fs = keyword.dt_in_fs  # update for propagation

        if keyword.keyword == 'run' and self.run_keyword:
            print("Warning: Previous 'run' keyword overwritten.")

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
            if 'ensemble' in self.keywords.keys() and not final_check:
                print(f"Warning: Previous 'ensemble' keyword will be overwritten for '{self.name}' run.")

            if not keyword.ensemble_type == 'nve' and not keyword.ensemble.parameters_set:
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
                if 1e3 / (self.dt_in_fs * keyword.sample_interval) < keyword.max_omega / math.pi:
                    raise ValueError("Sampling rate is less than the Nyquist rate.")

    def validate_run(self) -> None:
        """
        Validates that all used keywords are used correctly, consistent
        with the atoms, for the Run.
        """
        for key in self.keywords.keys():
            self._validate_keyword(self.keywords[key], final_check=True)

        if self.run_keyword is None:
            raise ValueError(f"No 'run' keyword provided for the '{self.name}' run.")

        if self.first_run and 'velocity' not in self.keywords:
            raise ValueError("A 'velocity' keyword must be used in the first run. See "
                             "https://gpumd.zheyongfan.org/index.php/The_velocity_keyword for details.")
