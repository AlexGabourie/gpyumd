__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"

import operator as op
import numbers
from util import cond_assign, cond_assign_int


class Keyword:

    def __init__(self, keyword, required_args, propagating, optional_args=None):
        """

        Args:
            keyword (str): The keyword for the run.in file
            required_args (list): A list of arguments required for the keyword. Must be in correct order.
            propagating (bool): Used to determine if a keyword propogates betwen runs.
            optional_args (dict or list): A dictionary of the optional arguments with optional label as the key. If a
                list, optional arguments are added in order
        """
        self.propagating = propagating
        self.keyword = keyword
        self.required_args = required_args
        self.optional_args = optional_args

    # TODO add a way to attach xyz files, also add a way to check if keywords are still valid if changing xyz
    # could be a "check_params" function in each class

    def get_command(self):
        # TODO add a universal "to string" command
        pass

    def set_required_args(self, required_args):
        """

        Args:
            required_args (list): A list of arguments required for the keyword. Must be in correct order.

        Returns:
            None
        """
        self.required_args = required_args


class Velocity(Keyword):

    def __init__(self, initial_temperature):
        """
        Set up the initial velocities with a given temperature
        Args:
            initial_temperature (float): Initial temperature of the system. [K]
        """
        self.keyword = 'velocity'
        self.propagating = False
        self.initial_temperature = cond_assign(initial_temperature, 0, op.gt, 'initial_temperature')
        super().__init__(self.keyword, [self.initial_temperature], self.propagating)

    def __str__(self):
        out = f"- \'{self.keyword}\' keyword -\n"
        out += f"initial_temperature: {self.initial_temperature} [K]"
        return out


class TimeStep(Keyword):

    def __init__(self, dt_in_fs, max_distance_per_step=None):
        """
        Sets the time step for integration.
        Args:
            dt_in_fs (float): The time step to use for integration [fs]
            max_distance_per_step (float): The maximum distance an atom can travel within one step [Angstroms]
        """
        self.keyword = 'time_step'
        self.propagating = True
        self.dt_in_fs = cond_assign(dt_in_fs, 0, op.gt, 'dt_in_fs')

        optional = None
        if max_distance_per_step:
            self.max_distance_per_step = cond_assign(max_distance_per_step, 0, op.gt, 'max_distance_per_step')
            optional = [self.max_distance_per_step]

        super().__init__(self.keyword, [self.dt_in_fs], self.propagating, optional_args=optional)


class Ensemble(Keyword):
    """
    Manages the "ensemble" keyword.
    See https://gpumd.zheyongfan.org/index.php/The_ensemble_keyword for additional details.
    """

    def __init__(self, ensemble_method):
        if not (ensemble_method in ['nve', 'nvt_ber', 'nvt_nhc', 'nvt_bdp', 'nvt_lan', 'npt_ber', 'npt_scr',
                                    'heat_nhc', 'heat_bdp', 'heat_lan']):
            raise ValueError(f"{ensemble_method} is not an accepted ensemble method.")
        self.keyword = 'enemble'
        self.ensemble_method = ensemble_method
        self.propagating = False
        super().__init__(self.keyword, [self.ensemble_method], self.propagating)

        self.ensemble = None
        ensemble_type = self.ensemble_method.split('_')[0]
        if ensemble_type == 'nvt':
            self.ensemble_method = Ensemble.NVT()
        elif ensemble_type == 'npt':
            self.ensemble_method = Ensemble.NPT()

    def set_nvt_parameters(self, initial_temperature, final_temperature, thermostat_coupling):
        """
        Sets the parameters of an NVT ensemble

        Args:
            initial_temperature (float): Initial temperature of run. [K]
            final_temperature (float): Final temperature of run. [K]
            thermostat_coupling (float): Coupling strength to the thermostat.

        Returns:
            None
        """
        if not isinstance(self.ensemble_method, self.NVT):
            raise Exception("Ensemble is not set for NVT.")
        required_args = self.ensemble_method.set_parameters(initial_temperature, final_temperature, thermostat_coupling)
        for arg in required_args:
            self.required_args.append(arg)

    def set_npt_parameters(self, initial_temperature, final_temperature, thermostat_coupling,
                           barostat_coupling, condition, pdict):
        """
        Sets parameters of an NPT ensemble.

        <condition> --> <required keys> \n
        'isotropic' --> p_hydro C_hydro \n
        'orthogonal' --> p_xx p_yy p_zz C_xx C_yy C_zz \n
        'triclinic' --> p_xx p_yy p_zz p_xy p_xz p_yz C_xx C_yy C_zz C_xy C_xz C_yz \n

        Args:
            initial_temperature (float): Initial temperature of run. [K]
            final_temperature (float): Final temperature of run. [K]
            thermostat_coupling (float): Coupling strength to the thermostat.
            barostat_coupling (float): Coupling strength to the thermostat.
            condition (str): Either 'isotropic', 'orthogonal', or 'triclinic'.
            pdict: Stores the elastic moduli [GPa] and barostat pressures [GPa] required for the condition. See
                below for more details.

        Returns:
            None

        """
        if not isinstance(self.ensemble_method, self.NPT):
            raise Exception("Ensemble is not set for NPT.")
        required_args = self.ensemble_method.set_parameters(initial_temperature, final_temperature, thermostat_coupling,
                                                            barostat_coupling, condition, pdict)
        for arg in required_args:
            self.required_args.append(arg)

    class NVT:

        def __init__(self):
            self.initial_temperature = None
            self.final_temperature = None
            self.thermostat_coupling = None
            self.parameters_set = False

        def set_parameters(self, initial_temperature, final_temperature, thermostat_coupling):
            self.initial_temperature = cond_assign(initial_temperature, 0, op.gt, 'initial_temperature')
            self.final_temperature = cond_assign(final_temperature, 0, op.gt, 'final_temperature')
            self.thermostat_coupling = cond_assign(thermostat_coupling, 1, op.ge, 'thermostat_coupling')
            self.parameters_set = True
            return [self.initial_temperature, self.final_temperature, self.thermostat_coupling]

    class NPT:

        def __init__(self):
            self.initial_temperature = None
            self.final_temperature = None
            self.thermostat_coupling = None
            self.barostat_coupling = None
            self.condition = None
            self.pdict = None
            self.parameters_set = False

        def set_parameters(self, initial_temperature, final_temperature, thermostat_coupling,
                           barostat_coupling, condition, pdict):
            self.initial_temperature = cond_assign(initial_temperature, 0, op.gt, 'initial_temperature')
            self.final_temperature = cond_assign(final_temperature, 0, op.gt, 'final_temperature')
            self.thermostat_coupling = cond_assign(thermostat_coupling, 1, op.ge, 'thermostat_coupling')
            self.barostat_coupling = cond_assign(barostat_coupling, 1, op.ge, 'barostat_coupling')

            if condition == 'isotropic':
                params = ['p_hydro', 'C_hydro']
            elif condition == 'orthogonal':
                params = ['p_xx', 'p_yy', 'p_zz', 'C_xx', 'C_yy', 'C_zz']
            elif condition == 'triclinic':
                params = ['p_xx', 'p_yy', 'p_zz', 'p_xy', 'p_xz', 'p_yz',
                          'C_xx', 'C_yy', 'C_zz', 'C_xy', 'C_xz', 'C_yz']
            else:
                raise ValueError(f"{condition} is not an accepted condition for the NPT ensemble.")

            pdict_valid = all([param in pdict.keys() for param in params])
            if pdict_valid:
                for key in params:
                    pdict_valid &= isinstance(pdict[key], numbers.Number)
                    if 'C_' in key:  # elastic moduli terms
                        pdict_valid &= (pdict[key] > 0)
                if not pdict_valid:
                    raise ValueError(f"Pressures must be a number and elastic moduli must be positive numbers.")

            else:
                raise ValueError(f"The NPT parameters passed in the pdict are not sufficient.")
            self.pdict = pdict

            self.parameters_set = True
            required_args = [self.initial_temperature, self.final_temperature, self.thermostat_coupling]
            for key in params:  # use params to guarantee order
                required_args.append(self.pdict[key])
            required_args.append(self.barostat_coupling)
            return required_args
