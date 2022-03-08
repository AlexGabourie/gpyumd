import numpy as np
from ase import Atom, Atoms

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


class GpumdAtoms(Atoms):

    def __init__(self, symbols=None,
                 positions=None, numbers=None,
                 tags=None, momenta=None, masses=None,
                 magmoms=None, charges=None,
                 scaled_positions=None,
                 cell=None, pbc=None, celldisp=None,
                 constraint=None,
                 calculator=None,
                 info=None,
                 velocities=None):
        """
        A thin wrapper class around the ASE Atoms object. Stores additional information

        ----- Atoms Documentation -----
        Atoms object.

        The Atoms object can represent an isolated molecule, or a
        periodically repeated structure.  It has a unit cell and
        there may be periodic boundary conditions along any of the three
        unit cell axes.
        Information about the atoms (atomic numbers and position) is
        stored in ndarrays.  Optionally, there can be information about
        tags, momenta, masses, magnetic moments and charges.

        In order to calculate energies, forces and stresses, a calculator
        object has to attached to the atoms object.

        Parameters:

        symbols: str (formula) or list of str
            Can be a string formula, a list of symbols or a list of
            Atom objects.  Examples: 'H2O', 'COPt12', ['H', 'H', 'O'],
            [Atom('Ne', (x, y, z)), ...].
        positions: list of xyz-positions
            Atomic positions.  Anything that can be converted to an
            ndarray of shape (n, 3) will do: [(x1,y1,z1), (x2,y2,z2),
            ...].
        scaled_positions: list of scaled-positions
            Like positions, but given in units of the unit cell.
            Can not be set at the same time as positions.
        numbers: list of int
            Atomic numbers (use only one of symbols/numbers).
        tags: list of int
            Special purpose tags.
        momenta: list of xyz-momenta
            Momenta for all atoms.
        masses: list of float
            Atomic masses in atomic units.
        magmoms: list of float or list of xyz-values
            Magnetic moments.  Can be either a single value for each atom
            for collinear calculations or three numbers for each atom for
            non-collinear calculations.
        charges: list of float
            Initial atomic charges.
        cell: 3x3 matrix or length 3 or 6 vector
            Unit cell vectors.  Can also be given as just three
            numbers for orthorhombic cells, or 6 numbers, where
            first three are lengths of unit cell vectors, and the
            other three are angles between them (in degrees), in following order:
            [len(a), len(b), len(c), angle(b,c), angle(a,c), angle(a,b)].
            First vector will lie in x-direction, second in xy-plane,
            and the third one in z-positive subspace.
            Default value: [0, 0, 0].
        celldisp: Vector
            Unit cell displacement vector. To visualize a displaced cell
            around the center of mass of a Systems of atoms. Default value
            = (0,0,0)
        pbc: one or three bool
            Periodic boundary conditions flags.  Examples: True,
            False, 0, 1, (1, 1, 0), (True, False, False).  Default
            value: False.
        constraint: constraint object(s)
            Used for applying one or more constraints during structure
            optimization.
        calculator: calculator object
            Used to attach a calculator for calculating energies and atomic
            forces.
        info: dict of key-value pairs
            Dictionary of key-value pairs with additional information
            about the system.  The following keys may be used by ase:

              - spacegroup: Spacegroup instance
              - unit_cell: 'conventional' | 'primitive' | int | 3 ints
              - adsorbate_info: Information about special adsorption sites

            Items in the info attribute survives copy and slicing and can
            be stored in and retrieved from trajectory files given that the
            key is a string, the value is JSON-compatible and, if the value is a
            user-defined object, its base class is importable.  One should
            not make any assumptions about the existence of keys.
        """
        super().__init__(symbols, positions, numbers, tags, momenta, masses, magmoms, charges, scaled_positions, cell,
                         pbc, celldisp, constraint, calculator, info, velocities)
        self.groups = list()  # list of group objects

    class GroupMethod:

        def __init__(self, group_type=None):
            """
            Stores grouping information for a GpumdAtoms object
            """
            self.group = None
            self.num_groups = None
            self.group_type = group_type
            self.counts = None

    class GroupByPosition(GroupMethod):

        def __init__(self, split, direction):
            super().__init__(group_type='position')
            self.split = split
            self.direction = direction
            self.num_groups = len(split) - 1
            self.counts = np.zeros(self.num_groups, dtype=int)

        def update(self, atoms):
            num_atoms = len(atoms)
            self.group = np.full(num_atoms, -1, dtype=int)
            for index, atom in enumerate(atoms):
                atom_group = self.get_group(atom.position)
                self.group[index] = atom_group
                self.counts[atom_group] += 1
            return self.counts

        def get_group(self, position):
            """
            Gets the group that an atom belongs to based on its position. Only works in
            one direction as it is used for NEMD.

            Args:
                position: list of floats of length 3
                    Position of the atom

            Returns:
                int: Group of atom
            """
            if self.direction == 'x':
                dim_pos = position[0]
            elif self.direction == 'y':
                dim_pos = position[1]
            else:
                dim_pos = position[2]
            errmsg = f"The position {dim_pos} in the {self.direction} direction is out of bounds based" \
                     f"on the split provided."
            for split_idx, boundary in enumerate(self.split[:-1]):
                if split_idx == 0 and dim_pos < boundary:
                    raise ValueError(errmsg)
                if boundary <= dim_pos < self.split[split_idx + 1]:
                    return split_idx
            raise ValueError(errmsg)

    def add_group_by_position(self, split, direction):
        """
        Assigns groups to all atoms based on its position. Only works in
        one direction as it is used for NEMD.
        Returns a bookkeeping parameter, but atoms will be udated in-place.

        Args:
            split (list(float)):
                List of boundaries. First element should be lower boundary of sim.
                box in specified direction and the last the upper.

            direction (str):
                Which direction the split will work.

        Returns:
            int: A list of number of atoms in each group.

        """
        if not (direction in ['x', 'y', 'z']):
            raise ValueError("The 'direction' parameter must be in 'x', 'y', 'or 'z'.")

        if len(split) < 2:
            raise ValueError("The 'split' parameter must be greater than length 1.")

        # check for ascending or descending
        if all([split[i+1] > split[i] for i in split[:-1]]) or all([split[i+1] < split[i] for i in split[:-1]]):
            raise ValueError("The 'split' parameter must be ascending or descending.")

        group = self.GroupByPosition(split, direction)
        return group.update(self)






