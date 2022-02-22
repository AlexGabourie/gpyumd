__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


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

    @staticmethod
    def _is_positive_float(val, varname):
        try:
            val = float(val)
        except ValueError:
            print(f"{varname} must be a float.")
            raise
        if val <= 0:
            ValueError(f"{varname} must be greater than 0.")
        return val


class Velocity(Keyword):

    def __init__(self, initial_temperature):
        """
        Set up the initial velocities with a given temperature
        Args:
            initial_temperature (float): Initial temperature of the system. [K]
        """
        self.keyword = 'velocity'
        self.propagating = False
        self.initial_temperature = super()._is_positive_float(initial_temperature, 'initial_temperature')
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
        self.dt_in_fs = super()._is_positive_float(dt_in_fs, 'dt_in_fs')

        optional = None
        if max_distance_per_step:
            self.max_distance_per_step = super()._is_positive_float(max_distance_per_step, 'max_distance_per_step')
            optional = [self.max_distance_per_step]

        super().__init__(self.keyword, [self.dt_in_fs], self.propagating, optional_args=optional)


# class Ensemble(Keyword):
#
# def __init__(self, ensemble_type, ):
