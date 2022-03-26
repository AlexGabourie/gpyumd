from ase import Atom, Atoms
import numpy as np
from gpyumd.atoms import GpumdAtoms
from gpyumd.util import get_path

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


#########################################
# Helper Functions
#########################################

def __process_header(atoms, sim, box):
    sim = sim.split()
    box = box.split()

    num_atoms = int(sim[0])
    max_neighbors = int(sim[1])
    cutoff = float(sim[2])
    atoms.triclinic = bool(sim[3])
    has_velocity = bool(sim[4])
    num_group_methods = int(sim[5])
    for group_num in range(num_group_methods):
        atoms.add_group_method(GpumdAtoms.GroupGeneric(np.zeros(num_atoms,dtype=int)))

    atoms.set_pbc([bool(pbc) for pbc in box[:3]])
    if atoms.triclinic:
        atoms.set_cell(np.array([float(component) for component in box[3:]]).reshape((3, 3)))
    else:
        lx, ly, lz = tuple(float(side_length) for side_length in box[3:])
        atoms.set_cell([[lx, 0, 0], [0, ly, 0], [0, lz, 0]])

    return max_neighbors, cutoff, has_velocity


def __get_atom_from_line(gpumd_atoms, atom_symbols, atom_line, atom_index, has_velocity):
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

#########################################
# Read Related
#########################################


def read_gpumd(atom_symbols=None, gpumd_file='xyz.in', directory='.'):
    """
    Reads and returns the structure input file from GPUMD.

    Args:
        atom_symbols: list of strings or ints
            List of atom symbols/atomic number used in the xyz.in file. Ex: ['Mo', 'S', 'Si', 'O'].
            Uses GPUMD type directly, if not provided.

        gpumd_file: string
            Name of structure file

        directory: string
            Directory of output

    Returns:
        tuple: GpumdAtoms, max_neighbors, cutoff
    """
    filepath = get_path(directory, gpumd_file)
    with open(filepath) as f:
        xyz_lines = f.readlines()

    gpumd_atoms = GpumdAtoms()
    max_neighbors, cutoff, has_velocity = __process_header(gpumd_atoms, xyz_lines[0], xyz_lines[1])
    for atom_index, atom_line in enumerate(xyz_lines[2:]):
        __get_atom_from_line(gpumd_atoms, atom_symbols, atom_line, atom_index, has_velocity)

    return gpumd_atoms, max_neighbors, cutoff


def read_movie(filename='movie.xyz', directory='.', atom_symbols=None):
    """
    Reads the trajectory from GPUMD run and creates a list of ASE atoms.

    Args:
        filename (str):
            Name of the file that holds the GPUMD trajectory.

        directory: string
            Directory of output. Assumes the in_file and movie.xyz file are in the same directory.

        atom_symbols: list of strings or ints
            List of atom symbols/atomic number used in the xyz.in file. Ex: ['Mo', 'S', 'Si', 'O'].
            Uses GPUMD type directly, if not provided.

    Returns:
        List of GpumdAtoms
    """
    with open(get_path(directory, filename), 'r') as f:
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
        trajectory.append(Atoms(symbols=symbols, positions=positions))
    return trajectory


#########################################
# Write Related
#########################################

def write_gpumd(gpumd_atoms, max_neighbors, cutoff, has_velocity=False, gpumd_file='xyz.in', directory='.'):
    """
    Creates and xyz.in file.

    Args:
        gpumd_atoms: GpumdAtoms
            The structure to write to file.

        max_neighbors: int
            Maximum number of neighbors for one atom

        cutoff: float
            Initial cutoff distance for building the neighbor list

        has_velocity: boolean
            Whether or not to set the velocities in the xyz.in file.

        gpumd_file: string
            File to save the structure data to

        directory: string
            Directory to store output
    """
    if not (isinstance(gpumd_atoms, GpumdAtoms)):
        raise ValueError("GpumdAtoms object is required to write an xyz.in file.")
    gpumd_atoms.write_gpumd(max_neighbors, cutoff, has_velocity, gpumd_file, directory)


def create_kpoints(gpumd_atoms, path='G', npoints=1, special_points=None, filename='kpoints.in', directory='.'):
    """
     Creates the file "kpoints.in", which specifies the kpoints needed for the 'phonon' keyword

    Args:
        gpumd_atoms: GpumdAtoms
            Unit cell to use for phonon calculation

        path: str
            String of special point names defining the path, e.g. 'GXL'

        npoints: int
            Number of points in total.  Note that at least one point
            is added for each special point in the path

        special_points: dict
            Dictionary mapping special points to scaled kpoint coordinates.
            For example ``{'G': [0, 0, 0], 'X': [1, 0, 0]}``

        filename: string
            File to save the structure data to

        directory: string
            Directory to store output

    Returns:
        kpoints converted to x-coordinates, x-coordinates of the high symmetry points, labels of those points.
    """
    if not (isinstance(gpumd_atoms, GpumdAtoms)):
        raise ValueError("GpumdAtoms object is required to write a kpoints.in file.")
    return gpumd_atoms.write_kpoints(path, npoints, special_points, filename, directory)


def write_basis(gpumd_atoms, filename='basis.in', directory='.'):
    """
    Creates the basis.in file. Atoms passed to this must already have the basis of every atom defined.\n
    Related: atoms.add_basis, atoms.repeat

    Args:
        gpumd_atoms: GpumdAtoms
            Atoms of unit cell used to generate basis.in

        filename: string
            File to save the structure data to

        directory: string
            Directory to store output
    """
    if not (isinstance(gpumd_atoms, GpumdAtoms)):
        raise ValueError("GpumdAtoms object is required to write a basis.in file.")
    gpumd_atoms.write_basis(filename, directory)

