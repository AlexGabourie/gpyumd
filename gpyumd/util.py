import re
import os
import numbers
import pandas as pd
from ase import Atom
from typing import List, BinaryIO


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


def cond_assign(val, threshold, relate, varname):
    val = assign_number(val, varname)
    if relate(val, threshold):
        return val
    raise ValueError(f"{varname} must be {relate2str(relate)} {threshold}")


def cond_assign_int(val, threshold, relate, varname) -> int:
    val = is_int(val, varname)
    if relate(val, threshold):
        return val
    raise ValueError(f"{varname} must be {relate2str(relate)} {threshold}")


def assign_number(val, varname):
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
            orig_val = val
            val = int(val)
            if not val == orig_val:
                print(f"Warning: '{varname}' must be an int. Converted from {orig_val} to {val}")
    except ValueError:
        print(f"{varname} must be an int.")
        raise
    return val


def get_direction(directions: str) -> List[str]:
    """
    Creates a sorted list showing which directions the user asked for.
    Ex: 'xyz' -> ['x', 'y', 'z']

    Args:
        directions: A string containing the directions the user wants to
         process (Ex: 'xyz', 'zy', 'x')

    Returns:
        An ordered list that simplifies the user input for future
         processing
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


def check_list(data: List[any], varname: str, dtype: type) -> List[any]:
    """
    Checks if data is a list of dtype or turns a variable of dtype into a list

    Args:
        data: Data to check
        varname: Name of variable to check
        dtype: Data type to check data against

    Returns:
        A list of dtype
    """
    if isinstance(data, dtype):
        return [data]

    if isinstance(data, list):
        for elem in data:
            if not isinstance(elem, dtype):
                raise ValueError(f'All entries for {varname} must be {dtype}.')
        return data

    raise ValueError(f'{varname} is not type {dtype}.')


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


def tail(file_handle: BinaryIO, nlines: int, block_size: int = 32768) -> List[bytes]:
    """
    Reads the last nlines of a file.

    Args:
        file_handle: File handle of file to be read
        nlines: Number of lines to be read from end of file
        block_size: Size of block (in bytes) to be read per read operation. Performance depends on this parameter and
            file size.

    Returns:
        final nlines of file

    Additional Information:
    Since GPUMD output files are mostly append-only, this becomes
    useful when a simulation prematurely ends (i.e. cluster preempts
    run, but simulation restarts elsewhere). In this case, it is not
    necessary to clean the directory before re-running. File outputs
    will be too long (so there still is a storage concern), but the
    proper data can be extracted from the end of file.
    This may also be useful if you want to only grab data from the
    final m number of runs of the simulation
    """
    # block_size is in bytes (must decode to string)
    file_handle.seek(0, 2)
    bytes_remaining = file_handle.tell()
    idx = -block_size
    blocks = list()
    # Make no assumptions about line length
    lines_left = nlines
    end_of_file = False
    first = True

    # block_size is smaller than file
    if block_size <= bytes_remaining:
        while lines_left > 0 and not end_of_file:
            if bytes_remaining > block_size:
                file_handle.seek(idx, 2)
                blocks.append(file_handle.read(block_size))
            else:  # if reached end of file
                file_handle.seek(0, 0)
                blocks.append(file_handle.read(bytes_remaining))
                end_of_file = True

            idx -= block_size
            bytes_remaining -= block_size
            num_lines = blocks[-1].count(b'\n')
            if first:
                lines_left -= num_lines - 1
                first = False
            else:
                lines_left -= num_lines

            # since whitespace removed from end_of_file, must compare to 1 here
            if end_of_file and lines_left > 1:
                raise ValueError("More lines requested than exist.")

        # Corrects for reading too many lines with large buffer
        if bytes_remaining > 0:
            skip = 1 + abs(lines_left)
            blocks[-1] = blocks[-1].split(b'\n', skip)[skip]
        text = b''.join(reversed(blocks)).strip()
    else:  # block_size is bigger than file
        file_handle.seek(0, 0)
        block = file_handle.read()
        num_lines = block.count(b'\n')
        if num_lines < nlines:
            raise ValueError("More lines requested than exist.")
        skip = num_lines - nlines
        text = block.split(b'\n', skip)[skip].strip()
    return text.split(b'\n')


def split_data_by_runs(points_per_run: List[int], data, labels: List[str]):
    start = 0
    out = dict()
    for run_num, npoints in enumerate(points_per_run):
        end = start + npoints
        run = dict()
        for label_num, key in enumerate(labels):
            run[key] = data[label_num][start:end].to_numpy(dtype='float')
        start = end
        out[f"run{run_num}"] = run
    return out


def basic_frame_loader(lines_per_frame, directory, filename):
    path = get_path(directory, filename)
    data = pd.read_csv(path, delim_whitespace=True, header=None).to_numpy(dtype='float')
    if not (data.shape[0] / lines_per_frame).is_integer():
        raise ValueError("An integer number of frames cannot be created. Please check num_atoms.")
    return data.reshape(-1, lines_per_frame, 3)
