import operator as op
import numbers
import os
from typing import List
# TODO change import to "import gpyumd.util as util" and update code so it is clear where code came from
from gpyumd.util import cond_assign, cond_assign_int, assign_bool, assign_number, get_path, check_symbols


__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


class Keyword:

    def __init__(self, keyword, propagating=False, take_immediate_action=False):
        """

        Args:
            keyword (str): The keyword for the run.in file
            propagating (bool): Used to determine if a keyword propogates betwen runs.
            take_immediate_action (bool): Used to determine if a keword is evaluated immediately. If True, only one
                keyword allowed for each run.
        """
        self.propagating = propagating
        self.keyword = keyword
        self.take_immediate_action = take_immediate_action

        self.required_args = None
        self.optional_args = None
        self.grouping_method = None
        self.group_id = None

    def get_entry(self):
        entry = f"{self.keyword} "
        entry += " ".join([f"{arg}" for arg in self.required_args]) + " "
        if self.optional_args:
            entry += " ".join([f"{arg}" for arg in self.optional_args])

    def _set_args(self, required_args, optional_args=None):
        """

        Args:
            required_args (list): A list of arguments required for the keyword. Must be in correct order.
            optional_args (list): A dictionary of the optional arguments with optional label as the key. If a
                list, optional arguments are added in order
        Returns:
            None
        """
        # TODO have a reset or extension?
        self.required_args = required_args
        if optional_args:
            self.optional_args = self._option_check(optional_args)

    def valid_group_options(self, grouping_method, group_id):
        # Take care of grouping options
        if not (bool(grouping_method) == bool(group_id)):
            raise ValueError("If the group option is to be used, both grouping_method and group_id must be defined.")
        elif grouping_method and group_id:
            self.grouping_method = cond_assign_int(grouping_method, 0, op.ge, 'grouping_method')
            self.group_id = cond_assign_int(group_id, 0, op.ge, 'group_id')

    @staticmethod
    def _option_check(options):
        return None if len(options) == 0 else options


class Velocity(Keyword):

    def __init__(self, initial_temperature):
        """
        Initializes the velocities of atoms according to a given temperature.

        https://gpumd.zheyongfan.org/index.php/The_velocity_keyword

        Args:
            initial_temperature (float): Initial temperature of the system. [K]
        """
        super().__init__('velocity', take_immediate_action=True)
        self.initial_temperature = cond_assign(initial_temperature, 0, op.gt, 'initial_temperature')
        self._set_args([self.initial_temperature])

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
        super().__init__('time_step', propagating=True)
        self.dt_in_fs = cond_assign(dt_in_fs, 0, op.gt, 'dt_in_fs')

        optional = None
        if max_distance_per_step:
            self.max_distance_per_step = cond_assign(max_distance_per_step, 0, op.gt, 'max_distance_per_step')
            optional = [self.max_distance_per_step]

        self._set_args([self.dt_in_fs], optional_args=optional)


class Ensemble(Keyword):

    def __init__(self, ensemble_method):
        """
        Manages the "ensemble" keyword.

        https://gpumd.zheyongfan.org/index.php/The_ensemble_keyword

        Args:
            ensemble_method: Must be one of: 'nve', 'nvt_ber', 'nvt_nhc', 'nvt_bdp', 'nvt_lan', 'npt_ber', 'npt_scr',
                                    'heat_nhc', 'heat_bdp', 'heat_lan'
        """
        super().__init__('ensemble')
        if not (ensemble_method in ['nve', 'nvt_ber', 'nvt_nhc', 'nvt_bdp', 'nvt_lan', 'npt_ber', 'npt_scr',
                                    'heat_nhc', 'heat_bdp', 'heat_lan']):
            raise ValueError(f"{ensemble_method} is not an accepted ensemble method.")
        self.ensemble_method = ensemble_method
        self._set_args([self.ensemble_method])

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
        self.required_args.extend(required_args)

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
        self.required_args.extend(required_args)

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
        self.required_args.extend(required_args)

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
            self.temperature = cond_assign(temperature, 0, op.gt, 'temperature')
            self.therostat_coupling = cond_assign(thermostat_coupling, 1, op.ge, 'thermostat_coupling')
            if (temperature_delta >= self.temperature) or (temperature_delta <= -self.temperature):
                raise ValueError(f"The magnitude of temperature_delta is too large.")
            self.source_group_id = cond_assign_int(source_group_id, 0, op.ge, 'source_group_id')
            self.sink_group_id = cond_assign_int(sink_group_id, 0, op.ge, 'sink_group_id')
            if self.source_group_id == self.sink_group_id:
                raise ValueError(f"The source and sink group cannot be the same.")
            self.parameters_set = True
            return [self.temperature, self.therostat_coupling, self.temperature_delta,
                    self.source_group_id, self.sink_group_id]


class NeighborOff(Keyword):

    def __init__(self):
        """
        Tells GPUMD to not update the neighbor list during simulations. Should only be used when there is no atom
        diffusion in the simulation.

        https://gpumd.zheyongfan.org/index.php/The_neighbor_keyword

        """
        super().__init__('neighbor')
        self._set_args(['off'])


class Fix(Keyword):

    def __init__(self, group_id):
        """
        Fixes (freezes) a group of atoms

        https://gpumd.zheyongfan.org/index.php/The_fix_keyword

        Args:
            group_id: The group id of the atoms to freeze.
        """
        super().__init__('fix', False)
        self.group_id = cond_assign_int(group_id, 0, op.ge, 'group_id')
        self._set_args([self.group_id])


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
        super().__init__('deform')
        if not (isinstance(deform_x, bool) and isinstance(deform_y, bool) and isinstance(deform_z, bool)):
            raise ValueError("Deform parameters must be a boolean.")
        if not (deform_x or deform_y or deform_z):
            raise ValueError("One deform parameter must be True.")
        self.deform_x = deform_x
        self.deform_y = deform_y
        self.deform_z = deform_z
        self.strain_rate = cond_assign(strain_rate, 0, op.gt, 'strain_rate')
        self._set_args([self.strain_rate, int(self.deform_x), int(self.deform_y), int(self.deform_z)])


class DumpThermo(Keyword):

    def __init__(self, interval):
        """
        Dumps global thermodynamic properties

        https://gpumd.zheyongfan.org/index.php/The_dump_thermo_keyword

        Args:
            interval (int): Number of time steps between each dump of the thermodynamic data.
        """
        super().__init__('dump_thermo')
        self.interval = cond_assign_int(interval, 0, op.gt, 'interval')
        self._set_args([self.interval])


class DumpPosition(Keyword):

    def __init__(self, interval, grouping_method=None, group_id=None, precision=None):
        """
        Dump the atom positions (coordinates) to a text file named movie.xyz

        https://gpumd.zheyongfan.org/index.php/The_dump_position_keyword

        Args:
            interval (int): Number of time steps between each dump of the position data.
            grouping_method (int): The grouping method to use.
            group_id (int): The group ID of the atoms to dump the position of.
            precision (str): Only 'single' or 'double' is accepted. The '%g' format is used if nothing specified.
        """
        super().__init__('dump_position')
        self.interval = cond_assign_int(interval, 0, op.gt, 'interval')
        self.precision = None

        options = list()
        if self.valid_group_options(grouping_method, group_id):
            options.extend(['group', self.grouping_method, self.group_id])

        if precision:
            if precision not in ['single', 'double']:
                raise ValueError("The precision option must be either 'single' or 'double'.")
            self.precision = precision
            options.extend(['precision', self.precision])

        self._set_args([self.interval], optional_args=options)


class DumpNetCDF(Keyword):

    def __init__(self, interval, precision='double'):
        """
        Dump the atom positions in the NetCDF format.

        https://gpumd.zheyongfan.org/index.php/The_dump_netcdf_keyword

        Args:
            interval (int): Number of time steps between each dump of the position data.
            precision (str): Only 'single' or 'double' is accepted. The default is 'double'.
        """
        super().__init__('dump_netcdf')
        self.interval = cond_assign_int(interval, 0, op.gt, 'interval')

        options = list()
        if precision:
            if precision not in ['single', 'double']:
                raise ValueError("The precision option must be either 'single' or 'double'.")
            self.precision = precision
            options.extend(['precision', self.precision])

        self._set_args([self.interval], optional_args=options)


class DumpRestart(Keyword):

    def __init__(self, interval):
        """
        Dump data to the restart file

        https://gpumd.zheyongfan.org/index.php/The_dump_restart_keyword

        Args:
            interval (int): Number of time steps between each dump of the restart data.
        """
        super().__init__('dump_restart')
        self.interval = cond_assign_int(interval, 0, op.gt, 'interval')
        self._set_args([self.interval])


class DumpVelocity(Keyword):

    def __init__(self, interval, grouping_method=None, group_id=None):
        """
        Dump the atom velocities to velocity.out

        https://gpumd.zheyongfan.org/index.php/The_dump_velocity_keyword

        Args:
            interval (int): Number of time steps between each dump of the velocity data.
            grouping_method (int): The grouping method to use.
            group_id (int): The group ID of the atoms to dump the velocity of.
        """
        super().__init__('dump_velocity')
        self.interval = cond_assign_int(interval, 0, op.gt, 'interval')

        options = list()
        if self.valid_group_options(grouping_method, group_id):
            options.extend(['group', self.grouping_method, self.group_id])
        self._set_args([self.interval], optional_args=options)


class DumpForce(Keyword):

    def __init__(self, interval, grouping_method=None, group_id=None):
        """
        Dump the atom forces to a text file named force.out

        https://gpumd.zheyongfan.org/index.php/The_dump_force_keyword

        Args:
            interval (int): Number of time steps between each dump of the force data.
            grouping_method (int): The grouping method to use.
            group_id (int): The group ID of the atoms to dump the force of.
        """
        super().__init__('dump_force')
        self.interval = cond_assign_int(interval, 0, op.gt, 'interval')

        options = list()
        if self.valid_group_options(grouping_method, group_id):
            options.extend(['group', self.grouping_method, self.group_id])
        self._set_args([self.interval], optional_args=options)


class DumpEXYZ(Keyword):

    def __init__(self, interval, has_velocity=False, has_force=False):
        """
        Dumps data into dump.xyz in the extended XYZ format


        Args:
            interval (int): Number of time steps between each dump of the force data.
            has_velocity (bool): True to dump velocity data, False to not dump velocity data.
            has_force (bool): True to dump force data, False to not dump force data.
        """
        super().__init__('dump_exyz')
        self.interval = cond_assign_int(interval, 0, op.gt, 'interval')
        self.has_velocity = assign_bool(has_velocity, 'has_velocity')
        self.has_force = assign_bool(has_force, 'has_force')
        self._set_args([self.interval, int(self.has_velocity), int(self.has_force)])


class Compute(Keyword):

    def __init__(self, sample_interval, output_interval, grouping_method,
                 temperature=False, potential=False, force=False, virial=False, jp=False, jk=False):
        """
        Computes and outputs space- and time-averaged quantities to the compute.out file.

        https://gpumd.zheyongfan.org/index.php/The_compute_keyword

        Args:
            sample_interval (int): Sample quantities this many time steps.
            output_interval (int): Averaging over so many sampled data before giving one output.
            grouping_method (int): The grouping method to use.
            temperature (bool): True to output temperature, False otherwise.
            potential (bool): True to output the potential energy, False otherwise.
            force (bool): True to output the force vector, False otherwise.
            virial (bool): True to output the diagonal part of the virial, False otherwise.
            jp (bool): True to output potential part of the heat current vector, False otherwise.
            jk (bool): True to output kinetic part of the heat current vector, False otherwise.
        """
        super().__init__('compute')
        self.sample_interval = cond_assign_int(sample_interval, 0, op.gt, 'sample_interval')
        self.output_interval = cond_assign_int(output_interval, 0, op.gt, 'output_interval')
        self.grouping_method = cond_assign_int(grouping_method, 0, op.ge, 'grouping_method')
        args = [self.grouping_method, self.sample_interval, self.output_interval]

        self.temperature = assign_bool(temperature, 'temperature')
        self.potential = assign_bool(potential, 'potential')
        self.force = assign_bool(force, 'force')
        self.virial = assign_bool(virial, 'virial')
        self.jp = assign_bool(jp, 'jp')
        self.jk = assign_bool(jk, 'jk')

        args.append('temperature') if self.temperature else None
        args.append('potential') if self.potential else None
        args.append('force') if self.force else None
        args.append('virial') if self.virial else None
        args.append('jp') if self.jp else None
        args.append('jk') if self.jk else None

        self._set_args(args)


class ComputeSHC(Keyword):

    def __init__(self, sample_interval, num_corr_steps, transport_direction, num_omega, max_omega,
                 grouping_method=None, group_id=None):
        """
        Computes the non-equilibrium virial-velocity correlation function K(t) and the spectral heat current in a given
        direction for a group of atoms. Outputs data to the shc.out file.

        https://gpumd.zheyongfan.org/index.php/The_compute_shc_keyword

        Args:
            sample_interval (int): Sampling interval between two correlation steps.
            num_corr_steps (int): Total number of correlation steps.
            transport_direction (str): Only 'x', 'y', 'z' directions accepted.
            num_omega (int): Number of frequency points to consider.
            max_omega (float): Maximum angular frequency to consider.
            grouping_method (int): The grouping method to use.
            group_id (int): The group ID of the atoms to calculate the spectral heat current of.
        """
        super().__init__('compute_shc')
        sample_interval = cond_assign_int(sample_interval, 1, op.ge, 'sample_interval')
        self.sample_interval = cond_assign_int(sample_interval, 10, op.le, 'sample_interval')
        num_corr_steps = cond_assign_int(num_corr_steps, 100, op.ge, 'num_corr_steps')
        self.num_corr_steps = cond_assign_int(num_corr_steps, 1000, op.le, 'num_corr_steps')
        if not (transport_direction in ['x', 'y', 'z']):
            raise ValueError("Only 'x', 'y', and 'z' are accepted for the 'transport_direction' parameter.")
        self.transport_direction = transport_direction
        self.num_omega = cond_assign_int(num_omega, 0, op.ge, 'num_omega')
        self.max_omega = cond_assign(max_omega, 0, op.gt, 'max_omega')

        options = list()
        if self.valid_group_options(grouping_method, group_id):
            options.extend(['group', self.grouping_method, self.group_id])

        transport_dict = {'x': 0, 'y': 1, 'z': 2}

        self._set_args([self.sample_interval, self.num_corr_steps, transport_dict[self.transport_direction],
                        self.num_omega, self.max_omega], optional_args=options)


class ComputeDOS(Keyword):

    def __init__(self, sample_interval, num_corr_steps, max_omega,
                 num_dos_points=None, grouping_method=None, group_id=None):
        """
        Computes the phonon density of states (PDOS) using the mass-weighted velocity autocorrelation (VAC). The output
        is normalized such that the integral of the PDOS over all frequencies equals 3N, where N is the number of atoms.
        Output goes to dos.out and mvac.out files.

        https://gpumd.zheyongfan.org/index.php/The_compute_dos_keyword

        Args:
            sample_interval (int): Sampling interval between two correlation steps.
            num_corr_steps (int): Total number of correlation steps.
            max_omega (float): Maximum angular frequency to consider.
            num_dos_points (int): Number of frequency points to be used in calculation. Default: num_corr_steps
            grouping_method (int): The grouping method to use.
            group_id (int): The group ID of the atoms to calculate the spectral heat current of.
        """
        super().__init__('compute_dos')
        self.sample_interval = cond_assign_int(sample_interval, 0, op.gt, 'sample_interval')
        self.num_corr_steps = cond_assign_int(num_corr_steps, 0, op.gt, 'num_corr_steps')
        self.max_omega = cond_assign(max_omega, 0, op.gt, 'max_omega')

        options = list()
        self.num_dos_points = self.num_corr_steps
        if num_dos_points:
            self.num_dos_points = cond_assign_int(num_dos_points, 0, op.gt, 'num_dos_points')
            options.extend(['num_dos_points', self.num_dos_points])

        if self.valid_group_options(grouping_method, group_id):
            options.extend(['group', self.grouping_method, self.group_id])

        self._set_args([self.sample_interval, self.num_corr_steps, self.max_omega], optional_args=options)


class ComputeSDC(Keyword):

    def __init__(self, sample_interval, num_corr_steps, grouping_method=None, group_id=None):
        """
        Computes the self diffusion coefficient (SDC) using the velocity autocorrelation (VAC).
        Outputs data to the sdc.out file

        https://gpumd.zheyongfan.org/index.php/The_compute_sdc_keyword

        Args:
            sample_interval (int): Sampling interval between two correlation steps.
            num_corr_steps (int): Total number of correlation steps.
            grouping_method (int): The grouping method to use.
            group_id (int): The group ID of the atoms to calculate the spectral heat current of.
        """
        super().__init__('compute_sdc')
        self.sample_interval = cond_assign_int(sample_interval, 0, op.gt, 'sample_interval')
        self.num_corr_steps = cond_assign_int(num_corr_steps, 0, op.gt, 'num_corr_steps')

        options = list()
        if self.valid_group_options(grouping_method, group_id):
            options.extend(['group', self.grouping_method, self.group_id])

        self._set_args([self.sample_interval, self.num_corr_steps], optional_args=options)


class ComputeHAC(Keyword):

    def __init__(self, sample_interval, num_corr_steps, output_interval):
        """
        Calculates the heat current autocorrelation (HAC) and running thermal conductivity (RTC) using the
        Green-Kubo method. Outputs data to hac.out file.

        https://gpumd.zheyongfan.org/index.php/Main_Page#Inputs_for_the_src.2Fgpumd_executable

        Args:
            sample_interval (int): Sampling interval between two correlation steps.
            num_corr_steps (int): Total number of correlation steps.
            output_interval (int): The output interval of the HAC and RTC data.
        """
        super().__init__('compute_hac')
        self.sample_interval = cond_assign_int(sample_interval, 0, op.gt, 'sample_interval')
        self.num_corr_steps = cond_assign_int(num_corr_steps, 0, op.gt, 'num_corr_steps')
        self.output_interval = cond_assign_int(output_interval, 0, op.gt, 'output_interval')

        self._set_args([self.sample_interval, self.num_corr_steps, self.output_interval])


class ComputeHNEMD(Keyword):

    def __init__(self, output_interval, driving_force_x, driving_force_y, driving_force_z):
        """
        Calculates the thermal conductivity using the HNEMD method.

        https://gpumd.zheyongfan.org/index.php/The_compute_hnemd_keyword

        Args:
            output_interval (int): The output interval of the thermal conductivity.
            driving_force_x (float): The x-component of the driving force. [Angstroms^-1]
            driving_force_y (float): The y-component of the driving force. [Angstroms^-1]
            driving_force_z (float): The z-component of the driving force. [Angstroms^-1]
        """
        super().__init__('compute_hnemd')
        self.output_interval = cond_assign_int(output_interval, 0, op.gt, 'output_interval')
        self.driving_force_x = assign_number(driving_force_x, 'driving_force_x')
        self.driving_force_y = assign_number(driving_force_y, 'driving_force_y')
        self.driving_force_z = assign_number(driving_force_z, 'driving_force_z')

        self._set_args([self.output_interval, self.driving_force_x, self.driving_force_y, self.driving_force_z])


class ComputeGKMA(Keyword):

    def __init__(self, sample_interval, first_mode, last_mode, bin_option, size):
        """
        Calculates the modal heat current using the Green-Kubo modal analysis (GKMA) method. Outputs data to the
        heatmode.out file.

        https://gpumd.zheyongfan.org/index.php/The_compute_gkma_keyword

        Args:
            sample_interval (int): The sampling interval (in number of steps) used to compute the modal heat current.
            first_mode (int): First mode in the eigenvector.in file to include in the calculation.
            last_mode (int): Last mode in the eigenvector.in file to include in the calculation.
            bin_option (str): Only 'bin_size' or 'f_bin_size' are accepted.
            size (int or float): If bin_option == 'bin_size', this is an integer describing how many modes per bin. If
                bin_option == 'f_bin_size', this describes bin size in THz.
        """
        super().__init__('compute_gkma')
        self.sample_interval = cond_assign_int(sample_interval, 0, op.gt, 'sample_interval')
        self.first_mode = cond_assign_int(first_mode, 1, op.ge, 'first_mode')
        self.last_mode = cond_assign_int(last_mode, self.first_mode, op.ge, 'last_mode')

        if bin_option == 'bin_size':
            self.bin_option = bin_option
            self.size = cond_assign_int(size, 0, op.gt, 'size')
        elif bin_option == 'f_bin_size':
            self.bin_option = bin_option
            self.size = cond_assign(size, 0, op.gt, 'size')
        else:
            raise ValueError("The bin_option parameter must be 'bin_size' or 'f_bin_size'.")

        # TODO add hidden arguments?

        self._set_args([self.sample_interval, self.first_mode, self.last_mode, self.bin_option, self.size])


class ComputeHNEMA(Keyword):

    def __init__(self, sample_interval, output_interval, driving_force_x, driving_force_y, driving_force_z,
                 first_mode, last_mode, bin_option, size):
        """
        Computes the modal thermal conductivity using the homogeneous nonequilibrium modal analysis (HNEMA) method.

        https://gpumd.zheyongfan.org/index.php/The_compute_hnema_keyword

        Args:
            sample_interval (int): The sampling interval (in number of steps) used to compute the modal heat current.
            output_interval (int): The interval to output the modal thermal conductivity. Each modal thermal
                conductivity output is averaged over all samples per output interval.
            driving_force_x (float): The x-component of the driving force. [Angstroms^-1]
            driving_force_y (float): The y-component of the driving force. [Angstroms^-1]
            driving_force_z (float): The z-component of the driving force. [Angstroms^-1]
            first_mode (int): First mode in the eigenvector.in file to include in the calculation.
            last_mode (int): Last mode in the eigenvector.in file to include in the calculation.
            bin_option (str): Only 'bin_size' or 'f_bin_size' are accepted.
            size (int or float): If bin_option == 'bin_size', this is an integer describing how many modes per bin. If
                bin_option == 'f_bin_size', this describes bin size in THz.
        """
        super().__init__('compute_hnema')
        self.sample_interval = cond_assign_int(sample_interval, 0, op.gt, 'sample_interval')
        self.output_interval = cond_assign_int(output_interval, 0, op.gt, 'output_interval')
        if not (self.output_interval % self.sample_interval == 0):
            raise ValueError("The sample_interval must divide the output_interval an integer number of times.")

        self.driving_force_x = assign_number(driving_force_x, 'driving_force_x')
        self.driving_force_y = assign_number(driving_force_y, 'driving_force_y')
        self.driving_force_z = assign_number(driving_force_z, 'driving_force_z')

        self.first_mode = cond_assign_int(first_mode, 1, op.ge, 'first_mode')
        self.last_mode = cond_assign_int(last_mode, self.first_mode, op.ge, 'last_mode')

        if bin_option == 'bin_size':
            self.bin_option = bin_option
            self.size = cond_assign_int(size, 0, op.gt, 'size')
        elif bin_option == 'f_bin_size':
            self.bin_option = bin_option
            self.size = cond_assign(size, 0, op.gt, 'size')
        else:
            raise ValueError("The bin_option parameter must be 'bin_size' or 'f_bin_size'.")

        # TODO add hidden arguments?

        self._set_args([self.sample_interval, self.output_interval,
                        self.driving_force_x, self.driving_force_y, self.driving_force_z,
                        self.first_mode, self.last_mode, self.bin_option, self.size])


class RunKeyword(Keyword):

    def __init__(self, number_of_steps):
        """
        Run a number of steps according to the settings specified for the current run.

        https://gpumd.zheyongfan.org/index.php/The_run_keyword

        Args:
            number_of_steps (int): Number of steps to run.
        """
        super().__init__('run', take_immediate_action=True)
        self.number_of_steps = cond_assign_int(number_of_steps, 0, op.gt, 'number_of_steps')
        self._set_args([self.number_of_steps])


class Minimize(Keyword):

    def __init__(self, force_tolerance, max_iterations, method='sd'):
        """
        Minimizes the energy of the system. Currently only the steepest descent method has been implemented.

        https://gpumd.zheyongfan.org/index.php/The_minimize_keyword

        Args:
            force_tolerance (float): The maximum force component allowed for minimization to continue. [eV/A]
            max_iterations (int): Number of iterations to perform before the minimization stops.
            method (str): Only 'sd' is supported at this time.
        """
        super().__init__('minimize', take_immediate_action=True)
        self.force_tolerance = assign_number(force_tolerance, 'force_tolerance')
        self.max_iterations = cond_assign_int(max_iterations, 0, op.gt, 'max_iterations')
        if not method == 'sd':
            raise ValueError("Only the steepest descent method is implemented. The 'method' parameter must be 'sd'.")
        self.method = method
        self._set_args([self.method, self.force_tolerance, self.max_iterations])
        

class ComputeCohesive(Keyword):

    def __init__(self, start_factor, end_factor, num_points):
        """
        Computes the cohesive energy curve with outputs going to the cohesive.out file.

        https://gpumd.zheyongfan.org/index.php/The_compute_cohesive_keyword

        Args:
            start_factor (float): Smaller box-scaling factor
            end_factor (float): Larger box-scaling factor
            num_points (int): Number of points sampled uniformly from e1 to e1.
        """
        super().__init__('compute_cohesive', take_immediate_action=True)
        self.start_factor = cond_assign(start_factor, 0, op.gt, 'start_factor')
        self.end_factor = cond_assign(end_factor, self.start_factor, op.gt, 'end_factor')
        self.num_points = cond_assign_int(num_points, 2, op.ge, 'num_points')

        self._set_args([self.start_factor, self.end_factor, self.num_points])


class ComputeElastic(Keyword):

    def __init__(self, strain_value, symmetry_type):
        """
        Computes the elastic constants and outputs to the elastic.out file.

        https://gpumd.zheyongfan.org/index.php/The_compute_elastic_keyword

        Args:
            strain_value (float): The amount of strain to be applied in the calculations.
            symmetry_type (str): Currently only 'cubic' supported.
        """
        super().__init__('compute_elastic', take_immediate_action=True)
        strain_value = cond_assign(strain_value, 0, op.gt, 'strain_value')
        self.strain_value = cond_assign(strain_value, 0.1, op.le, 'strain_value')
        if not symmetry_type == 'cubic':
            raise ValueError("The symmetry_type parameter must be 'cubic' at this time.")
        self.symmetry_type = symmetry_type

        self._set_args([self.strain_value, self.symmetry_type])


class ComputePhonon(Keyword):

    def __init__(self, cutoff, displacement):
        """
        Computes the phonon dispersion using the finite-displacement method. Outputs data to the D.out and omega2.out
        files.

        https://gpumd.zheyongfan.org/index.php/The_compute_phonon_keyword

        A special eigenvector.in file can be generated for GKMA and HNEMA methods using compute_phonon. Follow the
        directions here: https://gpumd.zheyongfan.org/index.php/The_eigenvector.in_input_file

        Args:
            cutoff (float): Cutoff distance for calculating the force constants. [Angstroms]
            displacement (float): The displacement for calculating the force constants using the finite-displacment
                method. [Angstroms]
        """
        super().__init__('compute_phonon', take_immediate_action=True)
        self.cutoff = cond_assign(cutoff, 0, op.gt, 'cutoff')
        self.displacement = cond_assign(displacement, 0, op.gt, 'displacement')

        self._set_args([self.cutoff, self.displacement])


class Potential(Keyword):

    supported_potentials = ["tersoff_1989", "tersoff_1988", "tersoff_mini", "sw_1985", "rebo_mos2", "eam_zhou_2004",
                            "eam_dai_2006", "vashishta", "fcp", "nep", "nep_zbl", "nep3", "nep3_zbl", "nep4",
                            "nep4_zbl", "lj", "ri"]

    def __init__(self, filename, symbols=None, grouping_method=None, directory=None):
        """
        Special keyword that contains basic information about the potential.
        Note: This does NOT check if the formatting of the potential is correct. It also does not provide the full
        grammar of the kewyord.

        https://gpumd.zheyongfan.org/index.php/The_potential_keyword

        Args:
            filename: string
                Filename of the potential.

            symbols: list of strings
                A list of atomic symbols associated with the potential. Required for all but LJ potentials. The order
                is important.

            grouping_method: int
                The grouping method used to exclude intra-material interactions for LJ potentials.

            directory: string
                The directory in which the potential will be found. If None provided, assumes potential will be in run
                directory.
        """
        super().__init__('potential', take_immediate_action=True)
        self.potential_path = filename if not directory else get_path(directory, filename)
        if not os.path.exists(self.potential_path):
            raise ValueError("The path to the potential does not exist.")
        with open(self.potential_path, 'r') as f:
            line = f.readline().split()
        if len(line) < 2:
            raise ValueError("Potential file header is not formatted correctly.")
        potential_type, num_types = tuple(line[:2])
        if potential_type not in Potential.supported_potentials:
            raise ValueError("Potential file does not contain a supported potential.")
        self.potential_type = potential_type
        self.num_types = cond_assign_int(num_types, 0, op.gt, 'num_types')

        options = list()
        if not self.potential_type == "lj":
            if not symbols:
                raise ValueError("A list of symbols must be provided for non-LJ potentials.")
            self.symbols = check_symbols(symbols)
        else:
            if grouping_method:
                self.grouping_method = cond_assign_int(grouping_method, 0, op.ge, 'grouping_method')
                options.append(self.grouping_method)

        self._set_args([self.potential_path], optional_args=options)

    def update_symbols(self, symbols):
        if not len(symbols) == self.num_types:
            raise ValueError("Number of symbols does not match the number of types expected by the potential.")
        self.symbols = check_symbols(symbols)

    def set_types(self, types: List[int]) -> None:
        if self.potential_type == "lj":
            raise ValueError("type arguments are not allowed for lj potentials.")
        if not len(types) == self.num_types:
            raise ValueError("Incorrect number of types.")
        if not types == list(range(min(types), max(types)+1)):
            raise ValueError("types must be ascending and contiguous.")
        self._set_args(self.required_args, optional_args=types)




