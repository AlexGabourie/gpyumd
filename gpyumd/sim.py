__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"

import operator as op
from gpyumd.keyword import TimeStep
from gpyumd.util import cond_assign_int


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

    def __init__(self, atoms, dt_in_fs=None, number_of_steps=None):
        """

        Args:
            atoms: GpumdAtoms

            dt_in_fs: float
                Time step of simulation in fs. Default is 1 fs.
        """
        self.atoms = atoms
        self.keywords = dict()
        if not dt_in_fs:
            dt_in_fs = 1
        self.time_step = TimeStep(dt_in_fs=dt_in_fs)
        self.number_of_steps = cond_assign_int(number_of_steps, 0, op.gt, 'number_of_steps')
        # TODO add variables that track if there is an ensemble defined or an immediate action
        pass

    # TODO handle propagating keywords
    # TODO Coupling time for NVT, NPT, and heat must all follow tau/(time step) >= 1
    # TODO check that triclinic structure used for triclinic NPT
    # TODO check that one dimension is periodic for triclinic NPT
    # TODO just check that the NPT choice matches with the atoms choice
    # TODO Make sure there is a grouping method if using Heat ensemble
    # TODO make sure there is a grouping method if we want to use Fix keyword
    # TODO check nyquist frequency for a run (DOS)
    # TODO make sure that there is an NVT or NPT set for compute_hnemd (not langevin)
    # TODO add a warning if a keyword will not have an output during a run (i.e. output interval is too large)
    # TODO ensure that minimize comes after the potentials have been defined

    def add_keyword(self, keyword):
        # check for all grouped keywords except 'fix'
        if keyword.grouping_method is not None and (self.atoms.num_groups - 1) < keyword.grouping_method:
            raise ValueError(f"The selected grouping method for {keyword.keyword} is larger than the number of grouping"
                             f" methods avaialbe in the GpumdAtoms object.")

        # check for all grouped keywords except 'fix' and 'compute'
        if keyword.grouping_method is not None and keyword.group_id is not None:
            if keyword.group_id >= self.atoms.groups[keyword.grouping_method].num_groups:
                raise ValueError(f"The group_id listed for {keyword.keyword} is too large for the grouping method "
                                 f"{keyword.grouping_method}.")

        if keyword.keyword == 'fix':
            if self.atoms.num_groups == 0:
                raise ValueError(f"At least one grouping method is required for the {keyword.keyword} keyword.")
            if keyword.group_id >= self.atoms.groups[0].num_groups:
                raise ValueError(f"The group_id given for {keyword.keyword} is too large for grouping method 0.")

        # Check for heating ensembles
        if keyword.keyword == 'ensemble':
            if 'ensemble' in self.keywords.keys():
                # TODO make replace ensemble?
                raise ValueError(f"The 'ensemble' keyword has already been used in this run.")

            if not keyword.ensemble.parameters_set:
                raise ValueError(f"Cannot add an ensemble before its parameters are set.")

            if keyword.ensemble_type == 'heat':
                if self.atoms.num_groups == 0:
                    raise ValueError(f"At least one grouping method is required for the {keyword.keyword} "
                                     f"{keyword.ensemble_method} keyword.")

                if self.atoms.num_groups <= keyword.ensemble.source_group_id or \
                        self.atoms.num_groups <= keyword.ensemble.sink_group_id:
                    raise ValueError(f"The source or sink group is too large for the ensemble.")
        # TODO finish ensemble checking

        # TODO add a function that checks the tau-timestep relationship for the ensemble, this can be re-run when needed
        self.keywords[keyword.keyword] = keyword
