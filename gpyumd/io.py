from ase.io import write
from ase.io import read
from ase import Atom, Atoms
import numpy as np
from gpyumd.atoms import GpumdAtoms
from gpyumd.util import get_path
import sys

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
            if block == 0:
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


# TODO merge into atoms
def create_kpoints(atoms, path='G', npoints=1, special_points=None):
    """
     Creates the file "kpoints.in", which specifies the kpoints needed for src/phonon

    Args:
        atoms (ase.Atoms):
            Unit cell to use for phonon calculation

        path (str):
            String of special point names defining the path, e.g. 'GXL'

        npoints (int):
            Number of points in total.  Note that at least one point
            is added for each special point in the path

        special_points (dict):
            Dictionary mapping special points to scaled kpoint coordinates.
            For example ``{'G': [0, 0, 0], 'X': [1, 0, 0]}``

    Returns:
        tuple: First element is the kpoints converted to x-coordinates, second the x-coordinates of the high symmetry
        points, and third the labels of those points.
    """
    tol = 1e-15
    path = atoms.cell.bandpath(path, npoints, special_points=special_points)
    b = atoms.get_reciprocal_cell() * 2 * np.pi  # Reciprocal lattice vectors
    gpumd_kpts = np.matmul(path.kpts, b)
    gpumd_kpts[np.abs(gpumd_kpts) < tol] = 0.0
    np.savetxt('kpoints.in', gpumd_kpts, header=str(npoints), comments='', fmt='%g')
    return path.get_linear_kpoint_axis()


# TODO merge into atoms
def create_basis(atoms):
    """
    Creates the basis.in file. Atoms passed to this must already have the basis of every atom defined.\n
    Related: preproc.add_basis, preproc.repeat

    Args:
        atoms (ase.Atoms):
            Atoms of unit cell used to generate basis.in
    """
    out = '{}\n'.format(len(atoms.info['unitcell']))
    masses = atoms.get_masses()
    info = atoms.info
    for i in info['unitcell']:
        out += '{} {}\n'.format(i, masses[i])
    for i in range(atoms.get_global_number_of_atoms()):
        out += '{}\n'.format(info[i]['basis'])
    with open("basis.in", 'w') as file:
        file.write(out)


# TODO perhaps remove --> leave this to a user
def convert_gpumd_atoms(in_file='xyz.in', out_filename='in.xyz', output_format='xyz', atom_types=None):
    """
    Converts the GPUMD input structure file to any compatible ASE
    output structure file.
    **Warning: Info dictionary may not be preserved**.

    Args:
        in_file (str):
            GPUMD position file to get structure from

        out_filename (str):
            Name of output file after conversion

        output_format (str):
            ASE supported output format

        atom_types (list(str)):
            List of atom types (elements).

    """
    atoms, max_neighbors, cutoff = load_xyz_in(in_file, atom_types)
    write(out_filename, atoms, output_format)


# TODO perhaps remove --> leave this to a user
def lammps_atoms_to_gpumd(filename, max_neighbors, cutoff, style='atomic', gpumd_file='xyz.in'):
    """
    Converts a lammps data file to GPUMD compatible position file.

    Args:
        filename (str):
            LAMMPS data file name

        max_neighbors (int):
            Maximum number of neighbors for one atom

        cutoff (float):
            Initial cutoff distance for building the neighbor list

        style (str):
            Atom style used in LAMMPS data file

        gpumd_file (str):
            File to save the structure data to

    """
    # Load atoms
    atoms = read(filename, format='lammps-data', style=style)
    atoms_to_xyz_in(atoms, max_neighbors, cutoff, gpumd_file=gpumd_file)
