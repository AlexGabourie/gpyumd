from .common import __check_list, __check_range
from numpy import prod

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


#########################################
# Structure preprocessing
#########################################

def __get_group(split, position, direction):
    """
    Gets the group that an atom belongs to based on its position. Only works in
    one direction as it is used for NEMD.

    Args:
        split (list(float)):
            List of boundaries. First element should be lower boundary of
            sim. box in specified direction and the last the upper.

        position (float):
            Position of the atom

        direction (str):
            Which direction the split will work

    Returns:
        int: Group of atom

    """
    if direction == 'x':
        d = position[0]
    elif direction == 'y':
        d = position[1]
    else:
        d = position[2]
    errmsg = 'Out of bounds error: {}'.format(d)
    for i, val in enumerate(split[:-1]):
        if i == 0 and d < val:
            print(errmsg)
            return -1
        if val <= d < split[i + 1]:
            return i
    print(errmsg)
    return -1


def __init_index(index, info, num_atoms):
    """
    Initializes the index key for the info dict.

    Args:
        index (int):
            Index of atom in the Atoms object.

        info (dict):
            Dictionary that stores the velocity, and groups.

        num_atoms (int):
            Number of atoms in the Atoms object.

    Returns:
        int: Index of atom in the Atoms object.

    """
    if index == num_atoms - 1:
        index = -1
    if index not in info:
        info[index] = dict()
    return index


def __handle_end(info, num_atoms):
    """
    Duplicates the index -1 entry for key that's num_atoms-1. Works in-place.

    Args:
        info (dict):
            Dictionary that stores the velocity, and groups.

        num_atoms (int):
            Number of atoms in the Atoms object.

    """
    info[num_atoms - 1] = info[-1]


def add_group_by_position(split, atoms, direction):
    """
    Assigns groups to all atoms based on its position. Only works in
    one direction as it is used for NEMD.
    Returns a bookkeeping parameter, but atoms will be udated in-place.

    Args:
        split (list(float)):
            List of boundaries. First element should be lower boundary of sim.
            box in specified direction and the last the upper.

        atoms (ase.Atoms):
            Atoms to group

        direction (str):
            Which direction the split will work.

    Returns:
        int: A list of number of atoms in each group.

    """
    info = atoms.info
    counts = [0] * (len(split) - 1)
    num_atoms = len(atoms)
    for index, atom in enumerate(atoms):
        index = __init_index(index, info, num_atoms)
        i = __get_group(split, atom.position, direction)
        if 'groups' in info[index]:
            info[index]['groups'].append(i)
        else:
            info[index]['groups'] = [i]
        counts[i] += 1
    __handle_end(info, num_atoms)
    atoms.info = info
    return counts


def add_group_by_type(atoms, types):
    """
    Assigns groups to all atoms based on atom types. Returns a
    bookkeeping parameter, but atoms will be udated in-place.

    Args:
        atoms (ase.Atoms):
            Atoms to group

        types (dict):
            Dictionary with types for keys and group as a value.
            Only one group allowed per atom. Assumed groups are integers
            starting at 0 and increasing in steps of 1. Ex. range(0,10).

    Returns:
        int: A list of number of atoms in each group.

    """
    # atom symbol checking
    all_symbols = list(types)
    # check that symbol set matches symbol set of atoms
    if set(atoms.get_chemical_symbols()) - set(all_symbols):
        raise ValueError('Group symbols do not match atoms symbols.')
    if not len(set(all_symbols)) == len(all_symbols):
        raise ValueError('Group not assigned to all atom types.')

    num_groups = len(set([types[sym] for sym in set(all_symbols)]))
    num_atoms = len(atoms)
    info = atoms.info
    counts = [0] * num_groups
    for index, atom in enumerate(atoms):
        index = __init_index(index, info, num_atoms)
        group = types[atom.symbol]
        counts[group] += 1
        if 'groups' in info[index]:
            info[index]['groups'].append(group)
        else:
            info[index]['groups'] = [group]
    __handle_end(info, num_atoms)
    atoms.info = info
    return counts


def set_velocities(atoms, custom=None):
    """
    Sets the 'velocity' part of the atoms to be used in GPUMD.
    Custom velocities must be provided. They must also be in
    the units of eV^(1/2) amu^(-1/2).

    Args:
        atoms (ase.Atoms):
            Atoms to assign velocities to.

        custom (list(list)):
             list of len(atoms) with each element made from
             a 3-element list for [vx, vy, vz]

    """
    if not custom:
        raise ValueError("No velocities provided.")

    num_atoms = len(atoms)
    info = atoms.info
    if not len(custom) == num_atoms:
        return ValueError('Incorrect number of velocities for number of atoms.')
    for index, (atom, velocity) in enumerate(zip(atoms, custom)):
        if not len(velocity) == 3:
            return ValueError('Three components of velocity not provided.')
        index = __init_index(index, info, num_atoms)
        info[index]['velocity'] = velocity
    __handle_end(info, num_atoms)
    atoms.info = info


def __init_index2(index, info): # TODO merge this with other __init_index function
    if index not in info.keys():
        info[index] = dict()


def add_basis(atoms, index=None, mapping=None):
    """
    Assigns a basis index for each atom in atoms. Updates atoms.

    Args:
        atoms (ase.Atoms):
            Atoms to assign basis to.

        index (list(int)):
            Atom indices of those in the unit cell. Order is important.

        mapping (list(int)):
            Mapping of all atoms to the relevant basis positions


    """
    n = atoms.get_global_number_of_atoms()
    info = atoms.info
    info['unitcell'] = list()
    if index:
        if (mapping is None) or (len(mapping) != n):
            raise ValueError("Full atom mapping required if index is provided.")
        for idx in index:
            info['unitcell'].append(idx)
        for idx in range(n):
            __init_index2(idx, info)
            info[idx]['basis'] = mapping[idx]
    else:
        for idx in range(n):
            info['unitcell'].append(idx)
            # if no index provided, assume atoms is unit cell
            __init_index2(idx, info)
            info[idx]['basis'] = idx


def repeat(atoms, rep):
    """
    A wrapper of ase.Atoms.repeat that is aware of GPUMD's basis information.

    Args:
        atoms (ase.Atoms):
            Atoms to assign velocities to.

        rep (int | list(3 ints)):
            List of three positive integers or a single integer

    """
    rep = __check_list(rep, varname='rep', dtype=int)
    replen = len(rep)
    if replen == 1:
        rep = rep*3
    elif not replen == 3:
        raise ValueError("rep must be a sequence of 1 or 3 integers.")
    __check_range(rep, 2**64)
    supercell = atoms.repeat(rep)
    sinfo = supercell.info
    ainfo = atoms.info
    n = atoms.get_global_number_of_atoms()
    for i in range(1, prod(rep, dtype=int)):
        for j in range(n):
            sinfo[i*n+j] = {'basis': ainfo[j]['basis']}

    return supercell
