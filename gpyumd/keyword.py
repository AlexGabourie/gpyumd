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
        Initializes the velocities of atoms according to a given temperature.

        https://gpumd.zheyongfan.org/index.php/The_velocity_keyword

        Args:
            initial_temperature (float): Initial temperature of the system. [K]
        """

        self.initial_temperature = cond_assign(initial_temperature, 0, op.gt, 'initial_temperature')
        super().__init__('velocity', [self.initial_temperature], False)

    def __str__(self):
        out = f"- \'{self.keyword}\' keyword -\n"
        out += f"initial_temperature: {self.initial_temperature} [K]"
        return out


class TimeStep(Keyword):

    def __init__(self, dt_in_fs, max_distance_per_step=None):
        """
        Sets the time step for integration.

        https://gpumd.zheyongfan.org/index.php/The_time_step_keyword

        Args:
            dt_in_fs (float): The time step to use for integration [fs]
            max_distance_per_step (float): The maximum distance an atom can travel within one step [Angstroms]
        """
        self.dt_in_fs = cond_assign(dt_in_fs, 0, op.gt, 'dt_in_fs')

        optional = None
        if max_distance_per_step:
            self.max_distance_per_step = cond_assign(max_distance_per_step, 0, op.gt, 'max_distance_per_step')
            optional = [self.max_distance_per_step]

        super().__init__('time_step', [self.dt_in_fs], True, optional_args=optional)


class Ensemble(Keyword):

    def __init__(self, ensemble_method):
        """
        Manages the "ensemble" keyword.

        https://gpumd.zheyongfan.org/index.php/The_ensemble_keyword

        Args:
            ensemble_method: Must be one of: 'nve', 'nvt_ber', 'nvt_nhc', 'nvt_bdp', 'nvt_lan', 'npt_ber', 'npt_scr',
                                    'heat_nhc', 'heat_bdp', 'heat_lan'
        """

        if not (ensemble_method in ['nve', 'nvt_ber', 'nvt_nhc', 'nvt_bdp', 'nvt_lan', 'npt_ber', 'npt_scr',
                                    'heat_nhc', 'heat_bdp', 'heat_lan']):
            raise ValueError(f"{ensemble_method} is not an accepted ensemble method.")
        self.ensemble_method = ensemble_method
        super().__init__('ensemble', [self.ensemble_method], False)

        self.ensemble = None
        ensemble_type = self.ensemble_method.split('_')[0]
        if ensemble_type == 'nvt':
            self.ensemble = Ensemble.NVT()
        elif ensemble_type == 'npt':
            self.ensemble = Ensemble.NPT()
        else:
            self.ensemble = Ensemble.Heat()

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
        if not isinstance(self.ensemble, self.NVT):
            raise Exception("Ensemble is not set for NVT.")
        required_args = self.ensemble.set_parameters(initial_temperature, final_temperature, thermostat_coupling)
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
        if not isinstance(self.ensemble, self.NPT):
            raise Exception("Ensemble is not set for NPT.")
        required_args = self.ensemble.set_parameters(initial_temperature, final_temperature, thermostat_coupling,
                                                            barostat_coupling, condition, pdict)
        for arg in required_args:
            self.required_args.append(arg)

    def set_heat_parameters(self, temperature, thermostat_coupling, temperature_delta, source_group_id, sink_group_id):
        """
        Sets the parameters for a heating simulation.

        Args:
            temperature (float): Base temperature of the simulation. [K]
            thermostat_coupling (float): Coupling strength to the thermostat.
            temperature_delta (float): Temperature change from base temperature. [K] (Note: total delta is twice this.)
            source_group_id: The group ID (in grouping method 0) to source heat. (Note: +temperature_delta)
            sink_group_id: The group ID (in grouping method 0) to sink heat. (Note: -temperature_delta)

        Returns:
            None
        """
        if not isinstance(self.ensemble, self.Heat):
            raise Exception("Ensemble is not set for Heat.")
        required_args = self.ensemble.set_parameters(temperature, thermostat_coupling, temperature_delta,
                                                            source_group_id, sink_group_id)
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

    class Heat:

        def __init__(self):
            self.temperature = None
            self.therostat_coupling = None
            self.temperature_delta = None
            self.source_group_id = None
            self.sink_group_id = None
            self.parameters_set = False

        def set_parameters(self, temperature, thermostat_coupling, temperature_delta, source_group_id, sink_group_id):
            # TODO check grouping information with xyz if it exists (only give warning)
            self.temperature = cond_assign(temperature, 0, op.gt, 'temperature')
            self.therostat_coupling = cond_assign(thermostat_coupling, 1, op.ge, 'thermostat_coupling')
            if (temperature_delta >= self.temperature) or (temperature_delta <= -self.temperature):
                raise ValueError(f"The magnitude of temperature_delta is too large.")
            self.source_group_id = cond_assign_int(source_group_id, 0, op.ge, 'source_group_id')  # TODO max id
            self.sink_group_id = cond_assign_int(sink_group_id, 0, op.ge, 'sink_group_id')
            self.parameters_set = True
            return [self.temperature, self.therostat_coupling, self.temperature_delta,
                    self.source_group_id, self.sink_group_id]


class Neighbor(Keyword):

    def __init__(self, skin_distance=1):
        """
        Tells the code that the neighbor list should be updated on a specific run.

        https://gpumd.zheyongfan.org/index.php/The_neighbor_keyword

        Args:
            skin_distance (float): Difference between the cutoff distance for the neighbor list construction and force
                                   evaluation.
        """
        self.skin_distance = cond_assign(skin_distance, 0, op.gt, 'skin_distance')
        super().__init__('neighbor', [self.skin_distance], False)


class Fix(Keyword):

    def __init__(self, group_id):
        """
        Fixes (freezes) a group of atoms

        https://gpumd.zheyongfan.org/index.php/The_fix_keyword

        Args:
            group_id: The group id of the atoms to freeze.
        """

        # TODO ensure that group label is not too large
        self.group_id = cond_assign_int(group_id, 0, op.ge, 'group_id')
        super().__init__('fix', [self.group_id], False)


class Deform(Keyword):

    def __init__(self, strain_rate, deform_x=False, deform_y=False, deform_z=False):
        """
        Deforms the simulation box. Can be used for tensile tests.

        https://gpumd.zheyongfan.org/index.php/The_deform_keyword

        Args:
            strain_rate (float): Speed of the increase of the box length. [Angstroms/step]
            deform_x (bool): True to deform in direction, False to not.
            deform_y (bool): True to deform in direction, False to not.
            deform_z (bool): True to deform in direction, False to not.
        """
        if not (isinstance(deform_x, bool) and isinstance(deform_y, bool) and isinstance(deform_z, bool)):
            raise ValueError("Deform parameters must be a boolean.")
        if not (deform_x or deform_y or deform_z):
            raise ValueError("One deform parameter must be True.")
        self.deform_x = deform_x
        self.deform_y = deform_y
        self.deform_z = deform_z
        self.strain_rate = cond_assign(strain_rate, 0, op.gt, 'strain_rate')
        super().__init__('deform', [self.strain_rate, int(self.deform_x), int(self.deform_y), int(self.deform_z)],
                         False)


class DumpThermo(Keyword):

    def __init__(self, interval):
        """
        Dumps global thermodynamic properties

        https://gpumd.zheyongfan.org/index.php/The_dump_thermo_keyword

        Args:
            interval (int): Number of time steps to dump the data.
        """
        self.interval = cond_assign_int(interval, 0, op.gt, 'interval')
        super().__init__('dump_thermo', [self.interval], False)


class DumpPosition(Keyword):

    def __init__(self, interval, grouping_method=None, group_id=None, precision=None):
        """
        Dump the atom positions (coordinates) to a text file named movie.xyz

        https://gpumd.zheyongfan.org/index.php/The_dump_position_keyword

        Args:
            interval (int): Number of time steps to dump the position data.
            grouping_method (int): The grouping method to use.
            group_id (int): The group ID of the atoms to dump the position of.
            precision (str): Only 'single' or 'double' is accepted. The '%g' format is used if nothing specified.
        """
        self.interval = cond_assign_int(interval, 0, op.gt, 'interval')
        self.grouping_method = None
        self.group_id = None
        self.precision = None
        # TODO check if grouping method is too large or if grouping method exists. Same for ID
        options = list()

        # Take care of grouping options
        if not (bool(grouping_method) == bool(group_id)):
            raise ValueError("If the group option is to be used, both grouping_method and group_id must be defined.")
        elif grouping_method and group_id:
            self.grouping_method = cond_assign_int(grouping_method, 0, op.ge, 'grouping_method')
            self.group_id = cond_assign_int(group_id, 0, op.ge, 'group_id')
            options.extend(['group', self.grouping_method, self.group_id])

        if precision:
            if precision not in ['single', 'double']:
                raise ValueError("The precision option must be either 'single' or 'double'.")
            self.precision = precision
            options.extend(['precision', self.precision])

        super().__init__('dump_position', [self.interval], False, optional_args=options)
        