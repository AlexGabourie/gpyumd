__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"

from util import is_positive_float


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
        self.initial_temperature = is_positive_float(initial_temperature, 'initial_temperature')
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
        self.dt_in_fs = is_positive_float(dt_in_fs, 'dt_in_fs')

        optional = None
        if max_distance_per_step:
            self.max_distance_per_step = is_positive_float(max_distance_per_step, 'max_distance_per_step')
            optional = [self.max_distance_per_step]

        super().__init__(self.keyword, [self.dt_in_fs], self.propagating, optional_args=optional)


class Ensemble(Keyword):

    def __init__(self, ensemble_method):
        if not (ensemble_method in ['nve', 'nvt_ber', 'nvt_nhc', 'nvt_bdp', 'nvt_lan', 'npt_ber', 'npt_scr',
                                    'heat_nhc', 'heat_bdp', 'heat_lan']):
            raise ValueError(f"{ensemble_method} is not an accepted ensemble method.")
        self.keyword = 'enemble'
        self.ensemble_method = ensemble_method
        self.propagating = False
        super().__init__(self.keyword, [self.ensemble_method], self.propagating)

        # TODO add initializers for inner classes of ensembles
        self.ensemble = None
        ensemble_type = self.ensemble_method.split('_')[0]
        if ensemble_type == 'nvt':
            self.ensemble_method = Ensemble.NVT()
        elif ensemble_type == 'npt':
            self.ensemble_method = Ensemble.NPT()

    def set_nvt_parameters(self, initial_temperature, final_temperature, thermostat_coupling):
        pass

    def set_npt_parameters(self, initial_temperature, final_temperature, thermostat_coupling,
                           barostat_coupling, condition, pdict):
        """
        Sets parameters of an NPT ensemble.

        <condition> --> <required keys> \n
        'isotropic' --> p_hydro C_hydro \n
        'orthogonal' --> p_xx p_yy p_zz C_xx C_yy C_zz \n
        'triclinic' --> p_xx p_yy p_zz p_xy p_xz p_yz C_xx C_yy C_zz C_xy C_xz C_yz \n

        See https://gpumd.zheyongfan.org/index.php/The_ensemble_keyword for additional details.

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
        pass

    class NVT:

        def __init__(self):
            self.initial_temperature = None
            self.final_temperature = None
            self.thermostat_coupling = None
            self.parameters_set = False

        def set_parameters(self, initial_temperature, final_temperature, thermostat_coupling):
            self.initial_temperature = is_positive_float(initial_temperature, 'initial_temperature')
            self.final_temperature = is_positive_float(final_temperature, 'final_temperature')
            self.thermostat_coupling = is_positive_float(thermostat_coupling, 'thermostat_coupling')
            self.parameters_set = True

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
            self.initial_temperature = is_positive_float(initial_temperature, 'initial_temperature')
            self.final_temperature = is_positive_float(final_temperature, 'final_temperature')
            self.thermostat_coupling = is_positive_float(thermostat_coupling, 'thermostat_coupling')
            self.barostat_coupling = is_positive_float(barostat_coupling, 'barostat_coupling')
            self.condition = condition

            if self.condition == 'isotropic':
                params = ['p_hydro', 'C_hydro']
            elif self.condition == 'orthogonal':
                params = ['p_xx', 'p_yy', 'p_zz', 'C_xx', 'C_yy', 'C_zz']
            elif self.condition == 'triclinic':
                params = ['p_xx', 'p_yy', 'p_zz', 'p_xy', 'p_xz', 'p_yz',
                          'C_xx', 'C_yy', 'C_zz', 'C_xy', 'C_xz', 'C_yz']
            else:
                raise ValueError(f"{condition} is not an accepted condition for the NPT ensemble.")

            pdict_valid = all([param in pdict.keys() for param in params])
            if pdict_valid:
                for key in params:
                    pdict_valid = pdict_valid and (pdict[key] == is_positive_float(pdict[key], key))
                if not pdict_valid:
                    raise ValueError(f"All parameters in the pdict for the NPT ensemble must be positive floats.")

            else:
                raise ValueError(f"The NPT parameters passed in the pdict are not sufficient.")
