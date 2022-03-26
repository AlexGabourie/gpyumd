import numpy as np
from ase import Atoms
from abc import ABC, abstractmethod
from gpyumd.util import check_list, check_range, get_path
from numpy import prod

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
        self.group_methods = list()  # A list of grouping methods
        self.num_group_methods = 0

        # only used for setting up phonon calculations
        self.unitcell = None  # The atom indices that make up a unit cell
        self.basis = None  # The basis position of each atom

        _, _, _, a1, a2, a3 = tuple(self.cell.cellpar())
        self.triclinic = False if a1 == a2 == a3 == 90 else True

    @staticmethod
    def __atom_type_sortkey(atom, order):
        """
        Used as a key for sorting atom type

        Args:
            atom: ase.Atom
                Atom object
        """
        for i, sym in enumerate(order):
            if sym == atom.symbol:
                return i

    @staticmethod
    def __atom_group_sortkey(atom, group, order):
        """
        Used as a key for sorting atom groups for GPUMD in.xyz files

        Args:
            atom: ase.Atom
                Atom object

            group: list of ints
                Store the group information of each atom (1-to-1 correspondence)

            order: list of ints
                A list of ints in desired order for groups at group_index

        """
        for i, curr_group in enumerate(order):
            if curr_group == group[atom.index]:
                return i

    def __enforce_sort(self, atom_order):
        """
        Helper for sort_atoms.

        Args:
            atom_order: List of ints
                New atom order based on sorting

        Returns:

        """
        atoms_list = list()
        for index in atom_order:
            atoms_list.append(self[index])
        self.__update_atoms(atoms_list)

        # update groups
        for group in self.group_methods:
            group.update(self)

    def __update_atoms(self, atoms_list):
        """
        Args:
            atoms_list: List of Atoms

        Returns:
            None
        """
        symbols = list()
        positions = np.zeros((len(atoms_list), 3))
        tags = list()
        momentum = np.zeros(positions.shape)
        mass = list()
        magmom = list()
        charge = list()
        # Get lists of properties of atoms in new order
        for atom_idx, atom in enumerate(atoms_list):
            symbols.append(atom.symbol)
            positions[atom_idx, :] = atom.position
            tags.append(atom.tag)
            momentum[atom_idx, :] = atom.momentum
            mass.append(atom.mass)
            magmom.append(atom.magmom)
            charge.append(atom.charge)
        # Make a new Atoms object keeping all system properties as before, but with new atom order
        super().__init__(symbols, positions, tags=tags, momenta=momentum, masses=mass, magmoms=magmom, charges=charge,
                         cell=self.get_cell(),
                         pbc=self.get_pbc(),
                         celldisp=self.get_celldisp(),
                         constraint=self.constraints,
                         calculator=self.get_calculator(),
                         info=self.info)

        # Do not add this to ase.Atoms. Already stored as momenta. ASE does not allow both anyways.
        # velocities=self.get_velocities()

    def sort_atoms(self, sort_key=None, order=None, group_method=None):
        """

        Args:
            sort_key: string
                How to sort atoms ('group', 'type').
            order: list of strings or list of ints
                For sort_key=='type', a list of atomic symbol strings in the desired order. Ex: ["Mo", "S", "Si", "O"]
                For sort_key=='group', a list of ints in desired order for groups at group_index. Ex: [1,3,2,4]
            group_method: int
                Selects the group to sort in the output.
        Returns:

        """
        if not sort_key and not order and not group_method:
            print("Warning: No sorting parameters passed. Nothing has been changed.")
            return

        index_range = range(len(self))
        if sort_key == 'type':
            if not order:
                raise ValueError("Sorting by type requires the 'order' parameter.")
            self.__enforce_sort(sorted(index_range,
                                       key=lambda atom_idx: self.__atom_type_sortkey(self[atom_idx], order)))
        elif sort_key == 'group':
            if not order or group_method:
                raise ValueError("Sorting by group requires the 'order' and 'group_method' parameters.")
            if group_method >= self.num_group_methods:
                raise ValueError("The group_method parameter is greater than the number of grouping methods assigned.")
            if not (sorted(order) == list(range(self.group_methods[group_method].num_groups))):
                raise ValueError("Not all groups are accounted for.")
            self.__enforce_sort(sorted(index_range, key=lambda atom_idx:
                               self.__atom_group_sortkey(self[atom_idx],
                                      self.group_methods[group_method].groups, order)))
        elif sort_key is not None:
            print("Invalid sort_key. No sorting is done.")

        return

    # TODO add ability to customize which species goes with each type
    def write_gpumd(self, max_neighbors, cutoff, has_velocity=False, gpumd_file='xyz.in', directory='.'):
        """
        Creates and xyz.in file.

        Args:
            max_neighbors (int):
                Maximum number of neighbors for one atom

            cutoff (float):
                Initial cutoff distance for building the neighbor list

            has_velocity: boolean
                Whether or not to set the velocities in the xyz.in file.

            gpumd_file (str):
                File to save the structure data to

            directory: string
                Directory to store output

        """
        # Assign type IDs
        type_dict = dict()
        for i, type_ in enumerate(list(set(self.get_chemical_symbols()))):
            type_dict[type_] = i

        # Create first two lines
        pbc = self.get_pbc()
        lx, ly, lz, a1, a2, a3 = tuple(self.cell.cellpar())
        summary = f"{len(self)} {max_neighbors} {cutoff} {int(self.triclinic)} " \
                  f"{int(has_velocity)} {self.num_group_methods}\n" \
                  f"{int(pbc[0])} {int(pbc[1])} {int(pbc[2])} "

        if self.triclinic:
            for component in self.get_cell().flatten():
                summary += f"{component} "
        else:
            summary += f"{lx} {ly} {lz}"

        # write structure
        filename = get_path(directory, gpumd_file)
        with open(filename, 'w') as f:
            f.writelines(summary)
            for atom in self:
                pos = atom.position
                line = f"\n{type_dict[atom.symbol]} {pos[0]} {pos[1]} {pos[2]} {atom.mass} "
                if has_velocity:
                    vel = [p / atom.mass for p in atom.momentum]
                    line += f"{vel[0]} {vel[1]} {vel[2]} "
                for group in self.group_methods:
                    line += f"{group.groups[atom.index]} "
                f.writelines(line)
        return

    def add_group_method(self, group):
        self.group_methods.append(group)
        self.num_group_methods += 1
        return self.num_group_methods - 1

    def add_basis(self, index=None, mapping=None):
        """
        Assigns a basis index for each atom in atoms. Updates atoms.

        Args:
            index (list(int)):
                Atom indices of those in the unit cell. Order is important.

            mapping (list(int)):
                Mapping of all atoms to the relevant basis positions


        """
        num_atoms = len(self)
        self.unitcell = list()
        self.basis = list()
        if index:
            if (mapping is None) or (len(mapping) != num_atoms):
                raise ValueError("Full atom mapping required if index is provided.")
            for unit_cell_idx in index:
                self.unitcell.append(unit_cell_idx)
            for atom_idx in range(num_atoms):
                self.basis.append(mapping[atom_idx])
        else:
            # if no index provided, assume atoms is unit cell
            for atom_idx in range(num_atoms):
                self.unitcell.append(atom_idx)
                self.basis.append(atom_idx)

    def repeat(self, rep):
        """
        A wrapper of ase.Atoms.repeat that is aware of GPUMD's basis information.

        Args:
            rep (int | list(3 ints)):
                List of three positive integers or a single integer

        """
        rep = check_list(rep, varname='rep', dtype=int)
        replen = len(rep)
        if replen == 1:
            rep = rep * 3
        elif not replen == 3:
            raise ValueError("The rep parameter must be a sequence of 1 or 3 integers.")
        check_range(rep, 2 ** 64)
        supercell = GpumdAtoms(self.repeat(rep))
        supercell.unitcell = self.unitcell
        for i in range(1, prod(rep, dtype=int)):
            supercell.basis.append(self.basis)

        return supercell

    class GroupMethod(ABC):

        def __init__(self, group_type=None):
            """
            Stores grouping information for a GpumdAtoms object
            """
            self.groups = None
            self.num_groups = None
            self.group_type = group_type
            self.counts = None

        @abstractmethod
        def update(self, atoms, order=None):
            pass

    class GroupGeneric(GroupMethod):

        def __init__(self, groups):
            """
            Grouping with no specific guidelines. Mostly used for loaded xyz.in files.
            """
            super().__init__(group_type='generic')
            self.num_groups = len(set(groups))
            if not (sorted(set(groups)) == list(range(self.num_groups))):
                raise ValueError("Groups are not contiguous.")
            self.counts = np.zeros(self.num_groups, dtype=int)
            for group in groups:
                self.counts[group] += 1
            self.groups = groups

        def update(self, atoms, order=None):
            if not order:
                raise ValueError("Generic groups are only updated with new atom ordering. "
                                 "The 'order' parameter is required.")
            new_groups = list()
            for index in order:
                new_groups.append(self.groups[index])
            self.groups = new_groups

    class GroupByType(GroupMethod):

        def __init__(self, types):
            super().__init__(group_type='type')
            self.types = types
            self.num_groups = len(set(types.values()))
            self.counts = np.zeros(self.num_groups, dtype=int)

        def update(self, atoms, order=None):
            num_atoms = len(atoms)
            self.groups = np.full(num_atoms, -1, dtype=int)
            for index, atom in enumerate(atoms):
                atom_group = self.types[atom.symbol]
                self.counts[atom_group] += 1
                self.groups[index] = atom_group

    class GroupByPosition(GroupMethod):

        def __init__(self, split, direction):
            super().__init__(group_type='position')
            self.split = split
            self.direction = direction
            self.num_groups = len(split) - 1
            self.counts = np.zeros(self.num_groups, dtype=int)

        def update(self, atoms, order=None):
            num_atoms = len(atoms)
            self.groups = np.full(num_atoms, -1, dtype=int)
            for index, atom in enumerate(atoms):
                atom_group = self.get_group(atom.position)
                self.groups[index] = atom_group
                self.counts[atom_group] += 1

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
                     f" on the split provided."
            for split_idx, boundary in enumerate(self.split[:-1]):
                if split_idx == 0 and dim_pos < boundary:
                    raise ValueError(errmsg)
                if boundary <= dim_pos < self.split[split_idx + 1]:
                    return split_idx
            raise ValueError(errmsg)

    def group_by_position(self, split, direction):
        """
        Assigns groups to all atoms based on its position. Only works in
        one direction as it is used for NEMD.
        Returns a bookkeeping parameter, but atoms will be udated in-place.

        Args:
            split (list(float)):
                List of boundaries in ascending order. First element should be lower boundary of sim.
                box in specified direction and the last the upper.

            direction (str):
                Which direction the split will work.

        Returns:
            str: A string with the name of the grouping method.
            int: A list of number of atoms in each group.

        """
        if not (direction in ['x', 'y', 'z']):
            raise ValueError("The 'direction' parameter must be in 'x', 'y', 'or 'z'.")

        splitlen = len(split)
        if splitlen < 2:
            raise ValueError("The 'split' parameter must be greater than length 1.")

        # check for ascending or descending
        if not all([split[i + 1] > split[i] for i in range(splitlen - 1)]):
            raise ValueError("The 'split' parameter must be ascending.")

        group = self.GroupByPosition(split, direction)
        group.update(self)
        group_idx = self.add_group_method(group)
        return group_idx, group.counts

    def group_by_type(self, types):
        """
        Assigns groups to all atoms based on atom types. Returns a
        bookkeeping parameter, but atoms will be udated in-place.

        Args:
            types (dict):
                Dictionary with types for keys and group as a value.
                Only one group allowed per atom. Assumed groups are integers
                starting at 0 and increasing in steps of 1. Ex. range(0,10).

        Returns:
            str: A string with the name of the grouping method.
            int: A list of number of atoms in each group.

        """
        # atom symbol checking
        all_symbols = list(types)
        # check that symbol set matches symbol set of atoms
        if set(self.get_chemical_symbols()) - set(all_symbols):
            raise ValueError('Group symbols do not match atoms symbols.')
        if not len(set(all_symbols)) == len(all_symbols):
            raise ValueError('Group not assigned to all atom types.')

        group = self.GroupByType(types)
        group.update(self)
        group_idx = self.add_group_method(group)
        return group_idx, group.counts

    def write_kpoints(self, path='G', npoints=1, special_points=None, filename='kpoints.in', directory='.'):
        """
         Creates the file "kpoints.in", which specifies the kpoints needed for the 'phonon' keyword

        Args:
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
        tol = 1e-15
        path = self.cell.bandpath(path, npoints, special_points=special_points)
        b = self.get_reciprocal_cell() * 2 * np.pi  # Reciprocal lattice vectors
        gpumd_kpts = np.matmul(path.kpts, b)
        gpumd_kpts[np.abs(gpumd_kpts) < tol] = 0.0
        np.savetxt(get_path(directory, filename), gpumd_kpts, header=str(npoints), comments='', fmt='%g')
        return path.get_linear_kpoint_axis()
