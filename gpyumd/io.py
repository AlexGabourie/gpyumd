from ase.io import write
from ase.io import read
from ase import Atom, Atoms
import numpy as np
from gpyumd.atoms import GpumdAtoms
import sys

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


#########################################
# Helper Functions
#########################################

def __set_atoms(atoms, types):
    """
    Sets the atom symbols for atoms loaded from GPUMD where xyz.in does not
    contain that information

    Args:
        atoms (ase.Atoms):
            Atoms object to change symbols in

        types (list(str)):
            List of strings to assign to atomic symbols

    """
    for atom in atoms:
        atom.symbol = types[atom.number]


#########################################
# Read Related
#########################################

# TODO update with ase.atoms velocities
def load_xyz_in(filename='xyz.in', atom_types=None):
    """
    Reads and returns the structure input file from GPUMD.

    Args:
        filename (str):
            Name of structure file

        atom_types (list(str)):
            List of atom types (elements).

    Returns:
        tuple: atoms, max_neighbors, cutoff

    atoms (ase.Atoms):
    ASE atoms object with x,y,z, mass, group, type, cell, and PBCs
    from input file. group is stored in tag, atom type may not
    correspond to correct atomic symbol

    max_neighbors (int):
    Max number of neighbor atoms

    cutoff (float):
    Initial cutoff for neighbor list build
    """
    # read file
    with open(filename) as f:
        xyz_lines = f.readlines()

    # get global structure params
    l1 = tuple(xyz_lines[0].split())  # first line
    num_atoms, max_neighbors, use_triclinic, has_velocity, num_of_groups = [int(val) for val in l1[:2] + l1[3:]]
    cutoff = float(l1[2])
    l2 = tuple(xyz_lines[1].split())  # second line
    if use_triclinic:
        pbc, cell = [int(val) for val in l2[:3]], [float(val) for val in l2[3:]]
    else:
        pbc, length_xyz = [int(val) for val in l2[:3]], [float(val) for val in l2[3:]]

    # get atomic params
    info = dict()
    atoms = Atoms()
    atoms.set_pbc((pbc[0], pbc[1], pbc[2]))
    if use_triclinic:
        atoms.set_cell(np.array(cell).reshape((3, 3)))
    else:
        atoms.set_cell([(length_xyz[0], 0, 0), (0, length_xyz[1], 0), (0, 0, length_xyz[2])])

    for index, line in enumerate(xyz_lines[2:]):
        data = dict()
        lc = tuple(line.split())  # current line
        type_, mass = int(lc[0]), float(lc[4])
        position = [float(val) for val in lc[1:4]]
        atom = Atom(type_, position, mass=mass)
        lc = lc[5:]  # reduce list length for easier indexing
        if has_velocity:
            velocity = [float(val) for val in lc[:3]]
            lc = lc[3:]
            data['velocity'] = velocity
        if num_of_groups:
            groups = [int(group) for group in lc]
            data['groups'] = groups
        atoms.append(atom)
        info[index] = data

    atoms.info = info
    if atom_types:
        __set_atoms(atoms, atom_types)

    return atoms, max_neighbors, cutoff


def load_movie_xyz(filename='movie.xyz', in_file=None, atom_types=None):
    """
    Reads the trajectory from GPUMD run and creates a list of ASE atoms.

    Args:
        filename (str):
            Name of the file that holds the GPUMD trajectory.

        in_file (str):
            Name of the original structure input file. Not required, but
            can help load extra information not included in trajectory output.

        atom_types (list(str)):
            List of atom types (elements).

    Returns:
        list(ase.Atoms): A list of ASE atoms objects.
    """
    # get extra information about system if wanted
    if in_file:
        atoms, _, _ = load_xyz_in(in_file, atom_types)
        pbc = atoms.get_pbc()
    else:
        pbc = None

    with open(filename, 'r') as f:
        xyz_line = f.readlines()

        num_atoms = int(xyz_line[0])
        block_size = num_atoms + 2
        num_blocks = len(xyz_line) // block_size
        traj = list()
        for block in range(num_blocks):
            types = []
            positions = []
            # TODO Loop may be inefficient in accessing xyz_line
            for entry in xyz_line[block_size * block + 2:block_size * (block + 1)]:
                # type_ can be an atom number or index to atom_types
                type_, x, y, z = entry.split()[:4]
                positions.append([float(x), float(y), float(z)])
                if atom_types:
                    types.append(atom_types[int(type_)])
                else:
                    types.append(int(type_))
            if atom_types:
                traj.append(Atoms(symbols=types, positions=positions, pbc=pbc))
            else:
                traj.append(Atoms(numbers=types, positions=positions, pbc=pbc))
        return traj


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
