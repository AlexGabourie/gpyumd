import pickle
import os
import operator as op
from typing import Union, List, Tuple, Set, Any
from dataclasses import dataclass
from gpyumd import util

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


@dataclass
class AtomLJ:
    symbol: str
    epsilon: float
    sigma: float


class LennardJones:
    max_atoms = 10  # GPUMD limitation

    def __init__(self, symbols: Union[str, List[str]] = None,
                 epsilons: Union[float, List[float]] = None,
                 sigmas: Union[float, List[float]] = None,
                 ignore_pairs: Union[Set[str], List[Set[str]]] = None,
                 cut_scale: float = 2.5,
                 uff_init: bool = False):
        """
        Stores all atoms for a simulation with their LJ parameters.

        A special dictionary with atom symbols for keys and the epsilon and
        sigma LJ parameters for the values. This object interfaces with the
        UFF LJ potential parameters but can also accept arbitrary parameters.

        Args:
            symbols: A single symbol or a list of symbols to add to the
             initial LJ list.
            epsilons: List of epsilon values corresponding to each symbol
             in units of eV.
            sigmas: List of sigma values corresponding to each symbol in
             units of eV.
            ignore_pairs: List of sets where each set has two elements.
             Each element is a string for the symbol of the atom to
             ignore in that pair. Order in set is not important.
            cut_scale: Specifies the multiplicative factor to use on
             the sigma parameter to define the cutoffs. Default is 2.5.
            uff_init: Initialize all epsilon and sigma parameters with
             those from of the universal force field (UFF)
        """
        self.atoms = dict()
        self.global_cutoff = None
        self.use_global_cutoff = False
        self.cut_scale = util.assign_number(cut_scale, 'cut_scale')
        self.uff = self.load_uff()

        if symbols:
            symbols = util.check_list(symbols, 'symbols', str)
            if epsilons:
                epsilons = util.check_list(epsilons, 'epsilons', float)
                if not len(epsilons) == len(symbols):
                    raise ValueError("When provided, 'epsilons' must be same length as 'symbols'.")
            else:
                epsilons = [None]*len(symbols)
            if sigmas:
                sigmas = util.check_list(sigmas, 'sigmas', float)
                if not len(sigmas) == len(symbols):
                    raise ValueError("When provided, 'sigmas' must be same length as 'symbols'.")
            else:
                sigmas = [None]*len(sigmas)
            for symbol_idx, symbol in enumerate(symbols):
                if uff_init:
                    epsilon, sigma = self.uff[symbol]
                else:
                    epsilon, sigma = epsilons[symbol_idx], sigmas[symbol_idx]
                self.atoms[symbol] = (epsilon, sigma)

        self.ignored_pairs = set()
        self.custom_cutoff = dict()
        self.special_pairs = None

        if ignore_pairs:
            if isinstance(ignore_pairs, set):
                self.ignore_pair(ignore_pairs)
            else:
                for pair in ignore_pairs:
                    self.ignore_pair(pair)

    @staticmethod
    def _get_cutkey(pair):
        keylist = sorted(list(pair))
        if len(keylist) == 1:
            keylist.append(keylist[0])
        key = ' '.join(keylist)
        return key

    def add_atom(self, symbol: str, epsilon: float = None, sigma: float = None,
                 uff_init: bool = False, replace: bool = False) -> None:
        """
        Add an atom to the LennardJones object

        Args:
            symbol: Symbol of the atom
            epsilon: Energy term in eV
            sigma: Distance term in Angstroms
            uff_init: Initialize the atoms epsilon & sigma variables with
             UFF parameters. NOTE: overrides 'epsilon' & 'sigma' variables
            replace: Replace any existing atom with new parameters
        """
        if len(self.atoms) >= self.max_atoms:
            print("Warning: Max number of atoms being used (10). Atom is not added.")
            return None

        if not isinstance(symbol, str):
            raise ValueError("The 'symbol' parameter must be a string.")
        epsilon = util.cond_assign(epsilon, 0, op.gt, 'epsilon')
        sigma = util.cond_assign(sigma, 0, op.gt, 'sigma')
        data = (epsilon, sigma)
        if uff_init:
            data = self.uff[symbol]
        if not replace and symbol in self.atoms.keys():
            print("Warning: Atom already exists in LennardJones object. To replace parameters, please set the 'replace'"
                  " option to True.")
            return None
        self.atoms[symbol] = data

    def remove_atom(self, symbol: str) -> bool:
        """
        Removes atom from the LennardJones object

        Args:
            symbol: Symbol of atom to remove

        Returns:
            Success of atom removal
        """
        if not isinstance(symbol, str):
            raise ValueError("The 'symbol' parameter must be a string.")

        if symbol in self.atoms.keys():
            del self.atoms[symbol]

            ignored_pairs_to_remove = list()
            for pair in self.ignored_pairs:
                if symbol in pair:
                    ignored_pairs_to_remove.append(pair)
            [self.ignored_pairs.remove(pair) for pair in ignored_pairs_to_remove]
            return True

        return False

    def ignore_pair(self, pair: Set[str]):
        """
        Adds a pair to the list of pairs that will be ignored when output
        to file.

        Args:
            pair: A two-element set where each entry is a string of the
             symbol in the pair to ignore. Ex: {'Si', 'O'}
        """
        self.check_pair(pair)
        if pair not in self.ignored_pairs:
            self.ignored_pairs.add(pair)

    def set_cutoff(self, pair: Set[str], cutoff: float):
        """
        Sets a custom cutoff for a specific pair of atoms.

        Args:
            pair: A two-element set where each entry is a string
             of the symbol in the pair. Ex: {'Si', 'O'}
            cutoff: Custom cutoff to use in Angstroms.
        """
        self.check_pair(pair)
        key = self._get_cutkey(pair)
        self.custom_cutoff[key] = util.cond_assign(cutoff, 0, op.gt, 'cutoff')

    def remove_cutoff(self, pair: Set[str]):
        """
        Removes a custom cutoff for a pair of atoms.

        Args:
            pair: A two-element set where each entry is a string of the
            symbol in the pair.
        """
        self.check_pair(pair)
        key = self._get_cutkey(pair)
        self.custom_cutoff.pop(key, None)

    def set_global_cutoff(self, cutoff: float):
        """
        Sets a global cutoff for all pairs. Warning: Sets output to use
        global cutoffs (except for custom cutoffs)

        Args:
            cutoff: Custom cutoff to use in Angstroms.
        """
        self.global_cutoff = util.cond_assign(cutoff, 0, op.gt, 'cutoff')
        self.use_global_cutoff = True

    def set_cutoff_scaling(self, cut_scale: float):
        """
        Sets the amount to scale the sigma values of each pair. Warning:
        Sets output to use pair-based cutoffs. Custom cutoffs unaffected.

        Args:
            cut_scale: Scaling factor to be used on sigma
        """
        self.cut_scale = util.cond_assign(cut_scale, 0, op.gt, 'cut_scale')
        self.use_global_cutoff = False

    def write_potential(self, atom_order: List[str], filename: str = "ljparams.txt", cutoff_type: str = None):
        """
        Outputs a GPUMD style LJ parameters file using the atoms defined
        in atom_order in the order defined in atom_order.

        Args:
            filename: The filename or full path with filename of the output.
            atom_order: List of atom symbols to include LJ params output
             file. The order will determine the order in the output file.
             Ex. ['a', 'b', 'c'] = pairs => 'aa', 'ab', 'ac', 'ba', 'bb',
             'bc', 'ca', 'cb' 'cc' in this order.
            cutoff_type: Which cutoff style to use. Only 'global' and
             'scale' are accepted. If None, choice is automatically made
              based on user's most recent input.
        """
        if cutoff_type:
            if cutoff_type == "global":
                self.use_global_cutoff = True
            elif cutoff_type == "scale":
                self.use_global_cutoff = False
            else:
                print("Warning: 'cutoff_type' is not accepted value and will be ignored.")

        num_atoms = len(atom_order)
        for symbol in atom_order:
            if symbol not in self.atoms.keys():
                raise ValueError(f"{symbol} atom does not exist in LennardJones object.")
        out_txt = f"lj {num_atoms}"
        for first_atom_idx, first_atom_symbol in enumerate(atom_order):
            for second_atom_idx, second_atom_symbol in enumerate(atom_order):
                pair = {first_atom_symbol, second_atom_symbol}
                if pair in self.ignored_pairs:
                    out_txt += "\n0 0 0"
                    continue

                epsilon, sigma = \
                    self.lorentz_berthelot_mixing(self.atoms[first_atom_symbol], self.atoms[second_atom_symbol])

                cutkey = self._get_cutkey(pair)
                if self.use_global_cutoff:
                    cutoff = self.global_cutoff
                elif cutkey in self.custom_cutoff:
                    cutoff = self.custom_cutoff[cutkey]
                else:
                    cutoff = self.cut_scale * sigma

                out_txt += f"\n{epsilon} {sigma} {cutoff}"

        with open(filename, 'w', newline='') as ljfile:
            ljfile.writelines(out_txt)

    def check_pair(self, pair: Set[str]) -> None:
        if not (isinstance(pair, set) and (len(pair) not in [1, 2])):
            raise ValueError('Invalid pair.')

        for symbol in list(pair):
            if symbol not in self.atoms.keys():
                raise ValueError('Symbol in pair not found in LennardJones object.')

    def __str__(self):
        out_str = 'Symbol: Epsilon (eV), Sigma (Angs.)\n'
        for key in self.atoms:
            cur = self.atoms[key]
            out_str += f"{key}: {cur[0]}, {cur[1]}\n"

        out_str += f"\nCutoff scaling factor = {self.cut_scale}  (Used: {not self.use_global_cutoff})\n"
        out_str += f"\nGlobal cutoff = {self.global_cutoff} Angstroms  (Used: {self.use_global_cutoff})\n"

        if len(self.custom_cutoff) > 0:
            out_str += "\nCustom Cutoffs\n"
            for pair in self.custom_cutoff:
                lpair = pair.split()
                out_str += f"[{lpair[0]}, {lpair[1]}] : {self.custom_cutoff[pair]}\n"

        if len(self.ignored_pairs) > 0:
            out_str += "\nIgnored Pairs (Order not important)\n"
            for pair in self.ignored_pairs:
                pair_list: List[Any] = list(pair)
                if len(pair_list) == 1:
                    pair_list.append(pair_list[0])
                out_str += f"[{pair_list[0]}, {pair_list[1]}]\n"

        return out_str

    @staticmethod
    def load_uff():
        """
        Loads dictionary that stores relevant LJ from UFF.

        Returns:
            Dictionary with atom symbols as the key and a tuple of epsilon
             and sigma in units of eV and Angstroms, respectively.
        """
        path = os.path.abspath(os.path.join(__file__, "../../data/UFF.params"))
        return pickle.load(open(path, "rb"))

    @staticmethod
    def lorentz_berthelot_mixing(a1: Tuple[float, float], a2: Tuple[float, float]) -> Tuple[float, float]:
        """
        Applies Lorentz-Berthelot mixing rules on two atoms.

        Args:
            a1: Tuple of (epsilon, sigma)
            a2: Tuple of (epsilon, sigma)
        """
        eps = (a1[0] * a2[0]) ** (1. / 2)
        sigma = (a1[1] + a2[1]) / 2.
        return eps, sigma
