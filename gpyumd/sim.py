__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


class Simulation:
    """
    Stores the relevant information for a full 'gpumd' executable simulation.
    """

    # TODO Add runs
    # TODO add attach xyz file
    # TODO add potentials
    # TODO add working directory

    def __init__(self, directory='.'):
        self.directory = directory
        self.runs = list()
        self.atoms = None
        self.potentials = list()

    def create_simulation(self):
        """
        Generates the required files for the gpumd simulation
        :return:
        """
        pass


class Run:

    def __init__(self):
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



