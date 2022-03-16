__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"

import operator as op
from gpyumd.keyword import TimeStep
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

    def add_run(self):

        pass


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
        self.number_of_steps = cond_assign_int(number_of_steps, 0, op.gt, 'number_of_steps')
        # TODO add variables that track if there is an ensemble defined or an immediate action
        pass

    # TODO handle propagating keywords
    # TODO Coupling time for NVT, NPT, and heat must all follow tau/(time step) >= 1
    # TODO check nyquist frequency for a run (DOS)
    # TODO make sure that there is an NVT or NPT set for compute_hnemd (not langevin)
    # TODO add a warning if a keyword will not have an output during a run (i.e. output interval is too large)
    # TODO ensure that minimize comes after the potentials have been defined

    def add_keyword(self, keyword):
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

        # TODO double check this at the end
        # Check for heating ensembles
        if keyword.keyword == 'ensemble':
            if 'ensemble' in self.keywords.keys():
                # TODO make replace ensemble?
                raise ValueError(f"The 'ensemble' keyword has already been used in this run.")

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

        # TODO add a function that checks the tau-timestep relationship for the ensemble, this can be re-run when needed
        self.keywords[keyword.keyword] = keyword
