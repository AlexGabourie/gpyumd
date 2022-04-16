from typing import List, Union, Optional, Mapping, Sequence, Tuple
from ase import Atom
import numpy as np
from gpyumd.atoms import GpumdAtoms, GroupGeneric
from gpyumd import util

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"

#########################################
# Read Related
#########################################


def read_gpumd(atom_symbols: List[Union[str, int]] = None,
               gpumd_file: str = "xyz.in",
               directory: str = None) -> GpumdAtoms:
    """
    Reads and returns the structure input file from GPUMD.

    Args:
        atom_symbols: List of atom symbols/atomic number used in the
         xyz.in file. Ex: ['Mo', 'S', 'Si', 'O']. Uses GPUMD type
         directly, if not provided.
        gpumd_file: Name of structure file
        directory: Directory of output

    Returns:
        GpumdAtoms
    """
    filepath = util.get_path(directory, gpumd_file)
    with open(filepath) as f:
        xyz_lines = f.readlines()

    gpumd_atoms = GpumdAtoms()

    # Process xyz.in header lines
    sim = xyz_lines[0].split()
    box = xyz_lines[1].split()
    num_atoms = int(sim[0])
    gpumd_atoms.set_max_neighbors(int(sim[1]))
    gpumd_atoms.set_cutoff(float(sim[2]))
    gpumd_atoms.triclinic = bool(sim[3])
    has_velocity = bool(sim[4])
    num_group_methods = int(sim[5])
    for group_num in range(num_group_methods):
        gpumd_atoms.add_group_method(GroupGeneric(np.zeros(num_atoms, dtype=int)))

    gpumd_atoms.set_pbc([bool(pbc) for pbc in box[:3]])
    if gpumd_atoms.triclinic:
        gpumd_atoms.set_cell(np.array([float(component) for component in box[3:]]).reshape((3, 3)))
    else:
        lx, ly, lz = tuple(float(side_length) for side_length in box[3:])
        gpumd_atoms.set_cell([[lx, 0, 0], [0, ly, 0], [0, lz, 0]])

    # Get atoms from each line
    for atom_index, atom_line in enumerate(xyz_lines[2:]):
        atom_line = atom_line.split()
        type_ = atom_symbols[int(atom_line[0])] if atom_symbols else int(atom_line[0])
        position = [float(val) for val in atom_line[1:4]]
        mass = float(atom_line[4])
        momentum = None
        atom_line = atom_line[5:]  # reduce list length for easier indexing
        if has_velocity:
            momentum = [float(val) * mass for val in atom_line[:3]]
            atom_line = atom_line[3:]
        atom = Atom(type_, position, mass=mass, momentum=momentum)
        gpumd_atoms.append(atom)

        if gpumd_atoms.num_group_methods:
            groups = [int(group) for group in atom_line]
            for group_method in range(len(groups)):
                gpumd_atoms.group_methods[group_method].groups[atom_index] = groups[group_method]

    return gpumd_atoms


def read_movie(filename: str = "movie.xyz",
               directory: str = None,
               atom_symbols: List[Union[str, int]] = None) -> List[GpumdAtoms]:
    """
    Reads the trajectory from GPUMD run and creates a list of ASE atoms.

    Args:
        filename: Name of the file that holds the GPUMD trajectory.
        directory: Directory of output. Assumes the in_file and
         movie.xyz file are in the same directory.
        atom_symbols: List of atom symbols/atomic number used in
         the xyz.in file. Ex: ['Mo', 'S', 'Si', 'O']. Uses GPUMD
         type directly, if not provided.

    Returns:
        List of GpumdAtoms
    """
    with open(util.get_path(directory, filename), 'r') as f:
        lines = f.readlines()  # FIXME make memory efficient

    block_size = int(lines[0]) + 2
    num_blocks = len(lines) // block_size
    trajectory = list()
    symbols = list()
    positions = np.zeros((block_size-2, 3))
    for block in range(num_blocks):
        for index, entry in enumerate(lines[block_size*block+2:block_size*(block+1)]):
            gpumd_type, x, y, z = entry.split()[:4]
            positions[index, 0] = float(x)
            positions[index, 1] = float(y)
            positions[index, 2] = float(z)
            if block == 0:  # Order of atoms is the same for every frame in GPUMD
                symbols.append(atom_symbols[int(gpumd_type)] if atom_symbols else int(gpumd_type))
        trajectory.append(GpumdAtoms(symbols=symbols, positions=positions))
    return trajectory


#########################################
# Write Related
#########################################

def write_gpumd(gpumd_atoms: GpumdAtoms, use_velocity: bool = False,
                gpumd_file: str = "xyz.in", directory: str = None) -> None:
    """
    Creates and xyz.in file. Note: both max_neighbors and cutoff must be
    set for the file to be written.

    Args:
        gpumd_atoms: The structure to write to file.
        use_velocity: Whether or not to set the velocities in the xyz.in
         file.
        gpumd_file: File to save the structure data to
        directory:  Directory to store output
    """
    if not (isinstance(gpumd_atoms, GpumdAtoms)):
        raise ValueError("GpumdAtoms object is required to write an xyz.in file.")
    gpumd_atoms.write_gpumd(use_velocity, gpumd_file, directory)


def write_kpoints(gpumd_atoms: GpumdAtoms, path: str = "G", npoints: int = 1,
                  special_points: Optional[Mapping[str, Sequence[float]]] = None,
                  filename: str = "kpoints.in", directory: str = None) -> Tuple[np.ndarray, None, List[str]]:
    """
     Creates the file "kpoints.in", which specifies the kpoints needed
     for the 'phonon' keyword

    Args:
        gpumd_atoms: Unit cell to use for phonon calculation
        path: String of special point names defining the path, e.g. 'GXL'
        npoints: Number of points in total.  Note that at least one point
         is added for each special point in the path
        special_points: Map of special points to scaled kpoint
         coordinates. For example ``{'G': [0, 0, 0], 'X': [1, 0, 0]}``
        filename: File to save the structure data to
        directory: Directory to store output

    Returns:
        kpoints converted to x-coordinates, x-coordinates of the high
         symmetry points, labels of those points.
    """
    if not (isinstance(gpumd_atoms, GpumdAtoms)):
        raise ValueError("GpumdAtoms object is required to write a kpoints.in file.")
    return gpumd_atoms.write_kpoints(path, npoints, special_points, filename, directory)


def write_basis(gpumd_atoms: GpumdAtoms, filename: str = "basis.in", directory: str = None) -> None:
    """
    Creates the basis.in file. Atoms passed to this must already have the
    basis of every atom defined. Related: atoms.add_basis, atoms.repeat

    Args:
        gpumd_atoms: Atoms of unit cell used to generate basis.in
        filename: File to save the structure data to
        directory: Directory to store output
    """
    if not (isinstance(gpumd_atoms, GpumdAtoms)):
        raise ValueError("GpumdAtoms object is required to write a basis.in file.")
    gpumd_atoms.write_basis(filename, directory)
