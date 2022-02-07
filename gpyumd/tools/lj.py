import pickle
import os

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


###################################
# UFF
###################################


def load_uff():
    """
    Loads dictionary that stores relevant LJ from UFF.

    Returns:
        dict:
            Dictionary with atom symbols as the key and a tuple of epsilon and
            sigma in units of eV and Angstroms, respectively.
    """
    path = os.path.abspath(os.path.join(__file__, '../../data/UFF.params'))
    return pickle.load(open(path, 'rb'))


#################################
# Lorentz-Berthelot Mixing
#################################


def lb_mixing(a1, a2):
    """
    Applies Lorentz-Berthelot mixing rules on two atoms.

    Args:
        a1 (tuple):
            Tuple of (epsilon, sigma)

        a2 (tuple):
            Tuple of (epsilon, sigma)
    """
    eps = (a1[0] * a2[0]) ** (1. / 2)
    sigma = (a1[1] + a2[1]) / 2.
    return eps, sigma


#################################
# LJ Object
#################################


class LJ(object):
    """ Stores all atoms for a simulation with their LJ parameters.

    A special dictionary with atom symbols for keys and the epsilon and
    sigma LJ parameters for the values. This object interfaces with the UFF
    LJ potential parameters but can also accept arbitrary parameters.

    Args:
        symbols (str or list(str)):
            Optional input. A single symbol or a list of symbols to add to
            the initial LJ list.

        ignore_pairs (list(sets)):
            List of sets where each set has two elements. Each element
            is a string for the symbol of the atom to ignore in that pair.
            Order in set is not important.

        cut_scale (float):
            Specifies the multiplicative factor to use on the sigma parameter
            to define the cutoffs. Default is 2.5.
    """

    def __init__(self, symbols=None, ignore_pairs=None, cut_scale=2.5):
        self.ljdict = dict()
        self.ignore = list()
        self.cutoff = dict()
        self.global_cutoff = None
        self.cut_scale = cut_scale
        if symbols:
            self.add_uff_params(symbols)

        if ignore_pairs:
            self.ignore_pairs(ignore_pairs)

    def add_uff_params(self, symbols, replace=False):
        """
        Adds UFF parameters to the LJ object. Will replace existing parameters
        if 'replace' is set to True. UFF parameters are loaded from the package.

        Args:
            symbols (str or list(str)):
                A single symbol or a list of symbols to add to the initial LJ list.

            replace (bool):
                Whether to replace existing symbols
        """
        if type(symbols) == str:
            symbols = [symbols]  # convert to list if one string
        uff = load_uff()
        for symbol in symbols:
            if not replace and symbol in self.ljdict:
                print("Warning: {} is already in LJ list and".format(symbol) +
                      " will not be included.\nTo include, use " +
                      "replace_UFF_params or toggle 'replace' boolean.\n")
            else:
                self.ljdict[symbol] = uff[symbol]

    def replace_uff_params(self, symbols, add=False):
        """
        Replaces current LJ parameters with UFF values. Will add new entries if
        'add' is set to True. UFF parameters are loaded from the package.

        Args:
            symbols (str or list(str)):
                A single symbol or a list of symbols to add to the initial LJ list.

            add (bool):
                Whether to replace existing symbols
        """
        if type(symbols) == str:
            symbols = [symbols]  # convert to list if one string
        uff = load_uff()
        for symbol in symbols:
            if symbol in self.ljdict or add:
                self.ljdict[symbol] = uff[symbol]
            else:
                print("Warning: {} is not in LJ list and".format(symbol) +
                      " cannot be replaced.\nTo include, use " +
                      "add_UFF_params or toggle 'add' boolean.\n")

    def add_param(self, symbol, data, replace=True):
        """
        Adds a custom parameter to the LJ object.

        Args:
            symbol (str):
                Symbol of atom type to add.

            data (tuple(float)):
                A two-element tuple of numbers to represent the epsilon and
                sigma LJ values.

            replace (bool):
                Whether to replace the item.
        """

        # check params
        good = (tuple == type(data) and len(data) == 2 and
                all([isinstance(item, (int, float)) for item in data]) and
                type(symbol) == str)
        if good:
            if symbol in self.ljdict:
                if replace:
                    self.ljdict[symbol] = data
                else:
                    print("Warning: {} exists and cannot be added.\n".format(symbol))
            else:
                self.ljdict[symbol] = data

        else:
            raise ValueError("Invalid data parameter.")

    def remove_param(self, symbol):
        """
        Removes an element from the LJ object. If item does not exist, nothing
        happens.

        Args:
            symbol (str):
                Symbol of atom type to remove.
        """
        # remove symbol from object
        self.ljdict.pop(symbol, None)
        # remove any ignore statements with symbol
        remove_list = list()
        for i, pair in enumerate(self.ignore):
            if symbol in pair:
                remove_list.append(i)

        for i in sorted(remove_list, reverse=True):
            del self.ignore[i]

    def ignore_pair(self, pair):
        """
        Adds a pair to the list of pairs that will be ignored when output to file.

        Args:
            pair (set):
                A two-element set where each entry is a string of the symbol in
                the pair to ignore.
        """
        self.__check_pair(pair)

        # check if pair exists already
        exists = False
        for curr_pair in self.ignore:
            if curr_pair == pair:
                return

        self.ignore.append(pair)

    def ignore_pairs(self, pairs):
        """
        Adds a list of pairs that will be ignored when output to file.

        Args:
            pairs (list(set)):
                A list of two-element sets where each entry of each set is a
                string of the symbol in the pair to ignore.
        """
        for pair in pairs:
            self.ignore_pair(pair)

    def acknowledge_pair(self, pair):
        """
        Removes the pair from ignore list and acknowledges it during the output.

        Args:
            pair (set):
                A two-element set where each entry is a string of the symbol in the
                pair to un-ignore.
        """
        self.__check_pair(pair)

        # check if pair exists already
        exists = False
        for i, curr_pair in enumerate(self.ignore):
            if curr_pair == pair:
                del self.ignore[i]
                return

        raise ValueError('Pair not found.')

    def acknowledge_pairs(self, pairs):
        """
        Removes pairs from ignore list.

        Args:
            pairs (list(set)):
                A list of two-elements sets where each entry in each set is a string of
                the symbol in the pair to un-ignore.
        """
        for pair in pairs:
            self.acknowledge_pair(pair)

    def custom_cutoff(self, pair, cutoff):
        """
        Sets a custom cutoff for a specific pair of atoms.

        Args:
            pair (set):
                A two-element set where each entry is a string of the symbol in the
                pair.

            cutoff (float):
                Custom cutoff to use. In Angstroms.
        """
        self.__check_pair(pair)
        self.__check_cutoff(cutoff)
        key = self.__get_cutkey(pair)
        self.cutoff[key] = cutoff

    def remove_custom_cutoff(self, pair):
        """
        Removes a custom cutoff for a pair of atoms.

        Args:
            pair (set):
                A two-element set where each entry is a string of the symbol in the
                pair.
        """
        self.__check_pair(pair)
        key = self.__get_cutkey(pair)
        self.cutoff.pop(key, None)

    def set_global_cutoff(self, cutoff):
        """
        Sets a global cutoff for all pairs.
        Warning: setting this will remove all other cutoff parameters.

        Args:
            cutoff (float):
                Custom cutoff to use. In Angstroms.
        """
        self.__check_cutoff(cutoff)
        self.global_cutoff = cutoff
        self.cutoff = dict()
        self.cut_scale = None

    def set_cut_scale(self, cut_scale):
        """
        Sets the amount to scale the sigma values of each pair by to set the cutoff.
        Warning: setting this will remove any global cutoff, but leave custom cutoffs.

        Args:
            cut_scale (float):
                Scaling factor to be used on sigma
        """
        self.__check_cutoff(cut_scale)
        self.global_cutoff = None
        self.cut_scale = cut_scale

    def create_file(self, filename='ljparams.txt', atom_order=None):
        """
        Outputs a GPUMD style LJ parameters file using the atoms defined in atom_order
        in the order defined in atom_order.

        Args:
            filename (str):
                The filename or full path with filename of the output.

            atom_order (list(str)):
                List of atom symbols to include LJ params output file. The order will
                determine the order in the output file. *Required*
                Ex. ['a', 'b', 'c'] = pairs => 'aa', 'ab', 'ac', 'ba', 'bb', 'bc', 'ca',
                'cb' 'cc' in this order.
        """
        if not atom_order:
            raise ValueError('atom_order is required.')

        # check if atoms in atom_order exist in LJ
        for symbol in atom_order:
            if symbol not in self.ljdict:
                raise ValueError('{} atom does not exist in LJ'.format(symbol))
        out_txt = 'lj {}\n'.format(len(atom_order))
        for i, sym1 in enumerate(atom_order):
            for j, sym2 in enumerate(atom_order):
                pair = {sym1, sym2}
                if pair in self.ignore:
                    if i + 1 == len(atom_order) and j + 1 == len(atom_order):
                        out_txt += '0 0 0'
                    else:
                        out_txt += '0 0 0\n'
                    continue

                a1 = self.ljdict[sym1]
                a2 = self.ljdict[sym2]

                eps, sig = lb_mixing(a1, a2)

                cutkey = self.__get_cutkey(pair)
                if self.global_cutoff:
                    cutoff = self.global_cutoff
                elif cutkey in self.cutoff:
                    cutoff = self.cutoff[cutkey]
                else:
                    cutoff = self.cut_scale * sig

                if i + 1 == len(atom_order) and j + 1 == len(atom_order):
                    out_txt += '{} {} {}'.format(eps, sig, cutoff)
                else:
                    out_txt += '{} {} {}\n'.format(eps, sig, cutoff)

        with open(filename, 'w') as f:
            f.writelines(out_txt)

    def __get_cutkey(self, pair):
        keylist = sorted(list(pair))
        if len(keylist) == 1:
            keylist.append(keylist[0])
        key = ' '.join(keylist)
        return key

    def __check_pair(self, pair):
        # check params
        if not (type(pair) == set and (len(pair) == 2 or len(pair) == 1)):
            raise ValueError('Invalid pair.')

        # check pair
        good = True
        for item in list(pair):
            good = good and item in self.ljdict

        if not good:
            raise ValueError('Elements in pair not found in LJ object.')

    def __check_cutoff(self, cutoff):
        if not (isinstance(cutoff, (int, float)) and cutoff > 0):
            raise ValueError('Invalid cutoff.')

    def __str__(self):
        out_str = 'Symbol: Epsilon (eV), Sigma (Angs.)\n'
        for key in self.ljdict:
            cur = self.ljdict[key]
            out_str += "{}: {}, {}\n".format(key, cur[0], cur[1])

        if self.cut_scale:
            out_str += "\nCutoff scaling factor = {}\n".format(self.cut_scale)
        else:  # Global cutoff
            out_str += "\nGlobal cutoff = {} Angstroms\n".format(self.global_cutoff)

        if len(self.cutoff) > 0:
            out_str += "\nCustom Cutoffs\n"
            for pair in self.cutoff:
                lpair = pair.split()
                out_str += "[{}, {}] : {}\n".format(lpair[0], lpair[1],
                                                    self.cutoff[pair])

        if len(self.ignore) > 0:
            out_str += "\nIgnored Pairs (Order not important)\n"
            for pair in self.ignore:
                pair = list(pair)
                if len(pair) == 1:
                    pair.append(pair[0])
                out_str += "[{}, {}]\n".format(pair[0], pair[1])

        return out_str
