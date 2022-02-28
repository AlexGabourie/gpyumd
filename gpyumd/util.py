import re
import os
import numbers


__author__ = "Alexander Gabourie"
__email__ = "gabourie@stanford.edu"


def relate2str(relate):
    relate = f"{relate}"[-3:-1]
    if relate == "lt":
        return "less than"
    elif relate == "le":
        return "less than or equal to"
    elif relate == "eq":
        return "equal to"
    elif relate == "ne":
        return "not equal to"
    elif relate == "ge":
        return "greater than or equal to"
    else:
        return "greater than"


def cond_assign(val, threshold, relate, varname):
    val = is_number(val, varname)
    if relate(val, threshold):
        return val
    raise ValueError(f"{varname} must be {relate2str(relate)} {threshold}")


def cond_assign_int(val, threshold, relate, varname):
    val = is_int(val, varname)
    if relate(val, threshold):
        return val
    raise ValueError(f"{varname} must be {relate2str(relate)} {threshold}")


def is_number(val, varname):
    try:
        if not isinstance(val, numbers.Number):
            val = float(val)
    except ValueError:
        print(f"{varname} must be a number.")
        raise
    return val


def is_int(val, varname):
    try:
        if not isinstance(val, int):
            val = int(val)
    except ValueError:
        print(f"{varname} must be an int.")
        raise
    return val


def get_direction(directions):
    """
    Creates a sorted list showing which directions the user asked for. Ex: 'xyz' -> ['x', 'y', 'z']

    Args:
        directions (str):
            A string containing the directions the user wants to process (Ex: 'xyz', 'zy', 'x')

    Returns:
        list(str): An ordered list that simplifies the user input for future processing

    """
    if not (bool(re.match('^[xyz]+$', directions))
            or len(directions) > 3
            or len(directions) == 0):
        raise ValueError('Invalid directions used.')
    return sorted(list(set(directions)))


def get_path(directory, filename):
    if not directory:
        return os.path.join(os.getcwd(), filename)
    return os.path.join(directory, filename)


def check_list(data, varname=None, dtype=None):
    """
    Checks if data is a list of dtype or turns a variable of dtype into a list

    Args:
        data:
            Data to check

        varname (str):
            Name of variable to check

        dtype (type)
            Data type to check data against
    Returns:
        list(dtype)
    """

    if type(data) == dtype:
        return [data]

    if type(data) == list:
        for elem in data:
            if not type(elem) == dtype:
                if varname:
                    raise ValueError('All entries for {} must be {}.'.format(str(varname), str(dtype)))
        return data

    raise ValueError('{} is not the correct type.'.format(str(varname)))


def check_range(npoints, maxpoints):
    """
    Checks if requested points are valid

    Args:
        npoints (list(int)):
            Points to check

        maxpoints (int):
            Maximum number of points to read

    Returns:
        None
    """
    if sum(npoints) > maxpoints:
        raise ValueError("More data requested than exists.")

    for points in npoints:
        if points < 1:
            raise ValueError("Only strictly positive numbers are allowed.")
