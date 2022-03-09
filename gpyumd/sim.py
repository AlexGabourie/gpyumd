__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


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


class Run:

    def __init__(self, atoms):
        """

        Args:
            atoms: GpumdAtoms
        """
        self.atoms = atoms
        self.keywords = list()
        pass
    # TODO enable user to attach results to a run
    # TODO ensure that only one ensemble is being defined in a run
    # TODO handle propagating keywords
    # TODO Coupling time for NVT, NPT, and heat must all follow tau/(time step) >= 1
    # TODO check that triclinic structure used for triclinic NPT
    # TODO check that one dimension is periodic for triclinic NPT
    # TODO just check that the NPT choice matches with the atoms choice
    # TODO Make sure there is a grouping method if using Heat ensemble
    # TODO make sure there is a grouping method if we want to use Fix keyword
    # TODO check nyquist frequency for a run (DOS)
    # TODO make sure that there is an NVT or NPT set for compute_hnemd (not langevin)
    # TODO make run class be initialized with a number of timesteps
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

        # TODO many more checks
        self.keywords.append(keyword)



