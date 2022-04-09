import re
import os
import numbers
from ase import Atom
from typing import List


__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


def check_symbols(symbols: List[str]) -> List[str]:
    try:
        for symbol in symbols:
            Atom(symbol)
    except KeyError:
        raise ValueError("Invalid atomic symbol.")
    return symbols


def assign_bool(var, varname) -> bool:
    if isinstance(var, bool):
        return var
    raise ValueError(f"The '{varname}' parameter must be a boolean.")


def relate2str(relate) -> str:
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


def cond_assign(val, threshold, relate, varname) -> numbers.Number:
    val = assign_number(val, varname)
    if relate(val, threshold):
        return val
    raise ValueError(f"{varname} must be {relate2str(relate)} {threshold}")


def cond_assign_int(val, threshold, relate, varname) -> int:
    val = is_int(val, varname)
    if relate(val, threshold):
        return val
    raise ValueError(f"{varname} must be {relate2str(relate)} {threshold}")


def assign_number(val, varname) -> numbers.Number:
    try:
        if not isinstance(val, numbers.Number):
            val = float(val)
    except ValueError:
        print(f"{varname} must be a number.")
        raise
    return val


def is_int(val, varname) -> int:
    try:
        if not isinstance(val, int):
            val = int(val)
    except ValueError:
        print(f"{varname} must be an int.")
        raise
    return val


def get_direction(directions: str) -> List[str]:
    """
    Creates a sorted list showing which directions the user asked for. Ex: 'xyz' -> ['x', 'y', 'z']

    Args:
        directions: A string containing the directions the user wants to process (Ex: 'xyz', 'zy', 'x')

    Returns:
        An ordered list that simplifies the user input for future processing
    """
    if not (bool(re.match('^[xyz]+$', directions)) or len(directions) > 3 or len(directions) == 0):
        raise ValueError('Invalid directions used.')
    return sorted(list(set(directions)))


def create_directory(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def get_path(directory: str, filename: str):
    if not directory:
        return os.path.join(os.getcwd(), filename)
    return os.path.join(directory, filename)


def check_list(data: List[any], varname: str = None, dtype: type = None) -> List[any]:
    """
    Checks if data is a list of dtype or turns a variable of dtype into a list

    Args:
        data: Data to check
        varname: Name of variable to check
        dtype: Data type to check data against

    Returns:
        A list of dtype
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


def check_range(npoints: List[int], maxpoints: int) -> None:
    """
    Checks if requested points are valid. Throws error if not.

    Args:
        npoints: Points to check
        maxpoints: Maximum number of points to read
    """
    if sum(npoints) > maxpoints:
        raise ValueError("More data requested than exists.")

    for points in npoints:
        if points < 1:
            raise ValueError("Only strictly positive numbers are allowed.")
