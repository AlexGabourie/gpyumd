import numpy as np
import pandas as pd
import os
import copy
import multiprocessing as mp
from functools import partial
from collections import deque
from typing import List, Dict, Union, Any

import gpyumd.util as util

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"

#########################################
# Helper Functions
#########################################


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
    path = util.get_path(directory, filename)
    data = pd.read_csv(path, delim_whitespace=True, header=None).to_numpy(dtype='float')
    if not (data.shape[0] / lines_per_frame).is_integer():
        raise ValueError("An integer number of frames cannot be created. Please check num_atoms.")
    return data.reshape(-1, lines_per_frame, 3)


#########################################
# Data-loading Related
#########################################

def load_omega2(filename: str = "omega2.out", directory: str = None) -> np.ndarray:
    """
    Loads data from omega2.out GPUMD output file.

    Args:
        filename: Name of force data file
        directory: Directory to load force file from

    Returns:
        Array of shape (N_kpoints,3*N_basis) in units of THz. N_kpoints is number of k points in kpoint.in and
        N_basis is the number of basis atoms defined in basis.in
    """
    path = util.get_path(directory, filename)
    data = pd.read_csv(path, delim_whitespace=True, header=None).to_numpy(dtype='float')
    data = np.sqrt(data)/(2*np.pi)
    return data


def load_force(num_atoms: int, filename: str = "force.out", directory: str = None) -> np.ndarray:
    """
    Loads data from force.out GPUMD output file. Currently supports loading a single run.

    Args:
        num_atoms: Number of atoms force is output for
        filename: Name of force data file
        directory: Directory to load force file from

    Returns:
        Numpy array of shape (-1,n,3) containing all forces (ev/A) from filename
    """
    return basic_frame_loader(num_atoms, directory, filename)


def load_velocity(num_atoms: int, filename: str = "velocity.out", directory: str = None) -> np.ndarray:
    """
    Loads data from velocity.out GPUMD output file. Currently supports loading a single run.

    Args:
        num_atoms: Number of atoms velocity is output for
        filename: Name of velocity data file
        directory: Directory to load velocity file from

    Returns:
        Numpy array of shape (-1,n,3) containing all forces (A/ps) from filename
    """
    return basic_frame_loader(num_atoms, directory, filename)


def load_compute(quantities: List[str], directory: str = None, filename: str = 'compute.out') \
        -> Dict[str, Union[Union[np.ndarray, int], Any]]:
    """
    Loads data from compute.out GPUMD output file. Currently supports loading a single run.

    Args:
        quantities: Quantities to extract from compute.out Accepted quantities are:
            ['temperature', 'U', 'F', 'W', 'jp', 'jk']. Other quantity will be ignored.
            Note: temperature=temperature, U=potential, F=force, W=virial,
            jp=heat current (potential), jk=heat current (kinetic)
        directory: Directory to load compute file from
        filename: file to load compute from

    Returns:
        Dictionary containing the data from compute.out

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,temperature,U,F,W,jp,jk,Ein,Eout
       **units**,K,eV,|c1|,eV,|c2|,|c2|,eV,eV

   .. |c1| replace:: eVA\ :sup:`-1`
   .. |c2| replace:: eV\ :sup:`3/2` amu\ :sup:`-1/2`
    """
    quantities = util.check_list(quantities, varname='quantities', dtype=str)
    compute_path = util.get_path(directory, filename)
    data = pd.read_csv(compute_path, delim_whitespace=True, header=None)

    num_col = len(data.columns)
    q_count = {'temperature': 1, 'U': 1, 'F': 3, 'W': 3, 'jp': 3, 'jk': 3}

    count = 0
    for value in quantities:
        count += q_count[value]

    m = int(num_col / count)
    out = dict()
    if 'temperature' in quantities:
        m = int((num_col - 2) / count)
        out['Ein'] = np.array(data.iloc[:, -2])
        out['Eout'] = np.array(data.iloc[:, -1])

    out['m'] = m
    start = 0
    for quantity in q_count.keys():
        if quantity in quantities:
            end = start + q_count[quantity]*m
            out[quantity] = data.iloc[:, start:end].to_numpy(dtype='float')
            start = end
    if start == 0:
        print("Warning: no valid quantities were passed to load_compute.")

    return out


def load_thermo(filename: str = "thermo.out", directory: str = None) -> Dict[str, np.ndarray]:
    """
    Loads data from thermo.out GPUMD output file.

    Args:
        filename: Name of thermal data file
        directory: Directory to load thermal data file from

    Returns:
        'output' dictionary containing the data from thermo.out

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,temperature,K,U,Px,Py,Pz,Lx,Ly,Lz,ax,ay,az,bx,by,bz,cx,cy,cz
       **units**,K,eV,eV,GPa,GPa,GPa,A,A,A,A,A,A,A,A,A,A,A,A

    """
    thermo_path = util.get_path(directory, filename)
    data = pd.read_csv(thermo_path, delim_whitespace=True, header=None)
    labels = ['temperature', 'K', 'U', 'Px', 'Py', 'Pz']
    # Orthogonal
    if data.shape[1] == 9:
        labels += ['Lx', 'Ly', 'Lz']
    elif data.shape[1] == 15:
        labels += ['ax', 'ay', 'az', 'bx', 'by', 'bz', 'cx', 'cy', 'cz']

    out = dict()
    for i in range(data.shape[1]):
        out[labels[i]] = data[i].to_numpy(dtype='float')

    return out


def load_sdc(num_corr_points: Union[int, List[int]],
             filename: str = "sdc.out",
             directory: str = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads data from sdc.out GPUMD output file.

    Args:
        num_corr_points: Number of time correlation points the VAC/SDC is computed for
        filename: File to load SDC from
        directory: Directory to load 'sdc.out' file from (dir. of simulation)

    Returns:
        Dictonary with SDC/VAC data. The outermost dictionary stores each individual run

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,t,VACx,VACy,VACz,SDCx,SDCy,SDCz
       **units**,ps,|sd1|,|sd1|,|sd1|,|sd2|,|sd2|,|sd2|

    .. |sd1| replace:: A\ :sup:`2` ps\ :sup:`-2`
    .. |sd2| replace:: A\ :sup:`2` ps\ :sup:`-1`
    """
    num_corr_points = util.check_list(num_corr_points, varname='nc', dtype=int)
    sdc_path = util.get_path(directory, filename)
    data = pd.read_csv(sdc_path, delim_whitespace=True, header=None)
    util.check_range(num_corr_points, data.shape[0])
    labels = ['t', 'VACx', 'VACy', 'VACz', 'SDCx', 'SDCy', 'SDCz']
    return split_data_by_runs(num_corr_points, data, labels)


def load_vac(num_corr_points: Union[int, List[int]],
             filename: str = "mvac.out",
             directory: str = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads data from mvac.out GPUMD output file.

    Args:
        num_corr_points: Number of time correlation points the VAC is computed for
        filename: File to load VAC from
        directory: Directory to load 'mvac.out' file from

    Returns:
        Dictonary with VAC data. The outermost dictionary stores each individual run

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,t,VACx,VACy,VACz
       **units**,ps,|v1|,|v1|,|v1|

    .. |v1| replace:: A\ :sup:`2` ps\ :sup:`-2`
    """
    num_corr_points = util.check_list(num_corr_points, varname='nc', dtype=int)
    sdc_path = util.get_path(directory, filename)
    data = pd.read_csv(sdc_path, delim_whitespace=True, header=None)
    util.check_range(num_corr_points, data.shape[0])
    labels = ['t', 'VACx', 'VACy', 'VACz']
    return split_data_by_runs(num_corr_points, data, labels)


def load_dos(num_dos_points: Union[int, List[int]],
             filename: str = 'dos.out',
             directory: str = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads data from dos.out GPUMD output file.

    Args:
        num_dos_points: Number of frequency points the DOS is computed for.
        filename: File to load DOS from.
        directory: Directory to load 'dos.out' file from (dir. of simulation)

    Returns:
        Dictonary with DOS data. The outermost dictionary stores each individual run.

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,nu,DOSx,DOSy,DOSz
       **units**,THz,|d1|,|d1|,|d1|

    .. |d1| replace:: THz\ :sup:`-1`
    """
    num_dos_points = util.check_list(num_dos_points, varname='num_dos_points', dtype=int)
    dos_path = util.get_path(directory, filename)
    data = pd.read_csv(dos_path, delim_whitespace=True, header=None)
    util.check_range(num_dos_points, data.shape[0])
    labels = ['nu', 'DOSx', 'DOSy', 'DOSz']
    out = split_data_by_runs(num_dos_points, data, labels)
    for key in out.keys():
        out[key]['nu'] /= (2 * np.pi)
    return out


def load_shc(num_corr_points: Union[int, List[int]], num_omega: Union[int, List[int]],
             filename: str = "shc.out", directory: str = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads the data from shc.out GPUMD output file.

    Args:
        num_corr_points: Maximum number of correlation steps. If multiple shc runs, can provide a list of nc.
        num_omega: Number of frequency points. If multiple shc runs, can provide a list of num_omega.
        filename: File to load SHC from.
        directory: Directory to load 'shc.out' file from (dir. of simulation)

    Returns:
        Dictionary of in- and out-of-plane shc results (average)

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,t, Ki, Ko, nu, jwi, jwo
       **units**,ps, |sh1|,|sh1|, THz, |sh2|, |sh2|

    .. |sh1| replace:: A eV ps\ :sup:`-1`
    .. |sh2| replace:: A eV ps\ :sup:`-1` THz\ :sup:`-1`
    """
    num_corr_points = util.check_list(num_corr_points, varname='nc', dtype=int)
    num_omega = util.check_list(num_omega, varname='num_omega', dtype=int)
    if not len(num_corr_points) == len(num_omega):
        raise ValueError('nc and num_omega must be the same length.')
    shc_path = util.get_path(directory, filename)
    data = pd.read_csv(shc_path, delim_whitespace=True, header=None)
    util.check_range(np.array(num_corr_points) * 2 - 1 + np.array(num_omega), data.shape[0])
    if not all([i>0 for i in num_corr_points]) or not all([i > 0 for i in num_omega]):
        raise ValueError('Only strictly positive numbers are allowed.')
    labels_corr = ['t', 'Ki', 'Ko']
    labels_omega = ['nu', 'jwi', 'jwo']

    out = dict()
    start = 0
    for run_num, varlen in enumerate(zip(num_corr_points, num_omega)):
        run = dict()
        num_corr_points_in_run = varlen[0] * 2 - 1
        num_omega_i = varlen[1]
        end = start + num_corr_points_in_run
        for label_num, key in enumerate(labels_corr):
            run[key] = data[label_num][start:end].to_numpy(dtype='float')
        start = end
        end += num_omega_i
        for label_num, key in enumerate(labels_omega):
            run[key] = data[label_num][start:end].to_numpy(dtype='float')
        run['nu'] /= (2 * np.pi)
        start = end
        out[f"run{run_num}"] = run
    return out


def load_kappa(filename: str = "kappa.out", directory: str = None) -> Dict[str, np.ndarray]:
    """
    Loads data from kappa.out GPUMD output file which contains HNEMD kappa.

    Args:
        filename: The kappa data file
        directory: Directory containing kappa data file

    Returns:
        A dictionary with keys corresponding to the columns in 'kappa.out'

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,kxi, kxo, kyi, kyo, kz
       **units**,|k1|,|k1|,|k1|,|k1|,|k1|

    .. |k1| replace:: Wm\ :sup:`-1` K\ :sup:`-1`
    """
    kappa_path = util.get_path(directory, filename)
    data = pd.read_csv(kappa_path, delim_whitespace=True, header=None)
    labels = ['kxi', 'kxo', 'kyi', 'kyo', 'kz']
    out = dict()
    for i, key in enumerate(labels):
        out[key] = data[i].to_numpy(dtype='float')
    return out


def load_hac(num_corr_points: Union[int, List[int]], output_interval: Union[int, List[int]],
             filename: str = "hac.out", directory: str = None) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Loads data from hac.out GPUMD output file.

    Args:
        num_corr_points: Number of correlation steps
        output_interval: Output interval for HAC and RTC data
        filename: The hac data file
        directory: Directory containing hac data file

    Returns:
        A dictionary containing the data from hac runs

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,t, kxi, kxo, kyi, kyo, kz, jxijx, jxojx, jyijy, jyojy, jzjz
       **units**,ps,|h1|,|h1|,|h1|,|h1|,|h1|,|h2|,|h2|,|h2|,|h2|,|h2|

    .. |h1| replace:: Wm\ :sup:`-1` K\ :sup:`-1`
    .. |h2| replace:: eV\ :sup:`3` amu\ :sup:`-1`
    """
    num_corr_points = util.check_list(num_corr_points, varname='nc', dtype=int)
    output_interval = util.check_list(output_interval, varname='output_interval', dtype=int)
    if not len(num_corr_points) == len(output_interval):
        raise ValueError('nc and output_interval must be the same length.')

    npoints = [int(x / y) for x, y in zip(num_corr_points, output_interval)]
    hac_path = util.get_path(directory, filename)
    data = pd.read_csv(hac_path, delim_whitespace=True, header=None)
    util.check_range(npoints, data.shape[0])
    labels = ['t', 'jxijx', 'jxojx', 'jyijy', 'jyojy', 'jzjz',
              'kxi', 'kxo', 'kyi', 'kyo', 'kz']
    start = 0
    out = dict()
    for run_num, varlen in enumerate(npoints):
        end = start + varlen
        run = dict()
        for label_num, label in enumerate(labels):
            run[label] = data[label_num][start:end].to_numpy(dtype='float')
        start = end
        out[f"run{run_num}"] = run
    return out


def read_modal_analysis_file(nbins: int, nsamples: int, datapath: str, ndiv: int, multiprocessing: bool = False,
                             ncore: int = None, block_size: int = 65536) -> np.ndarray:
    """
    Core reader for the modal analysis methods. Recommend using the load_heatmode or load_kappamode functions instead.

    Args:
        nbins: Number of frequency bins
        nsamples: Number of samples for simulation
        datapath: Full path of the data file
        ndiv: Divisor for shrinking the number of bins
        multiprocessing: Whether or not to use multi-core processing
        ncore: Number of cores to use if using multiprocessing
        block_size: Number of bytes to read at once from the output files

    Returns:
        3D array with of data with dimension (nbins, nsamples, 5). Note: ndiv will change nbins.
    """

    def process_sample(num_bins: int, sample_num: int) -> np.ndarray:
        """
        Args:
            num_bins: Number of bins used in the GPUMD simulation
            sample_num: The current sample from a run to analyze

        Returns:
            A 2D array of each bin and output for a sample
        """
        out = list()
        for bin_num in range(num_bins):
            out += [float(x) for x in malines[bin_num + sample_num * num_bins].split()]
        return np.array(out).reshape((num_bins, 5))

    # Get full set of results
    datalines = nbins * nsamples
    with open(datapath, "rb") as f:
        if multiprocessing:
            malines = util.tail(f, datalines, block_size=block_size)
        else:
            malines = deque(util.tail(f, datalines, block_size=block_size))

    if multiprocessing:  # TODO Improve memory efficiency of multiprocessing
        if not ncore:
            ncore = mp.cpu_count()

        func = partial(process_sample, nbins)
        pool = mp.Pool(ncore)
        data = np.array(pool.map(func, range(nsamples)), dtype='float32').transpose((1, 0, 2))
        pool.close()

    else:  # Faster if single thread
        data = np.zeros((nbins, nsamples, 5), dtype='float32')
        for sample_idx in range(nsamples):
            for bin_idx in range(nbins):
                measurements = malines.popleft().split()
                data[bin_idx, sample_idx, 0] = float(measurements[0])
                data[bin_idx, sample_idx, 1] = float(measurements[1])
                data[bin_idx, sample_idx, 2] = float(measurements[2])
                data[bin_idx, sample_idx, 3] = float(measurements[3])
                data[bin_idx, sample_idx, 4] = float(measurements[4])

    if ndiv:
        nbins = int(np.ceil(data.shape[0] / ndiv))  # overwrite nbins
        npad = nbins * ndiv - data.shape[0]
        data = np.pad(data, [(0, npad), (0, 0), (0, 0)])
        data = np.sum(data.reshape((-1, ndiv, data.shape[1], data.shape[2])), axis=1)

    return data


def load_heatmode(nbins: int,
                  nsamples: int,
                  inputfile: str = "heatmode.out",
                  directory: str = None,
                  directions: str = "xyz",
                  ndiv: int = None,
                  outputfile: str = "heatmode.npy",
                  save: bool = False,
                  multiprocessing: bool = False,
                  ncore: int = None,
                  block_size: int = None,
                  return_data: bool = True) -> Union[None, Dict[str, np.ndarray]]:
    """
    Loads data from heatmode.out GPUMD file. Option to save as binary file for fast re-load later.
    WARNING: If using multiprocessing, memory usage may be significantly larger than file size

    Args:
        nbins: Number of bins used during the GPUMD simulation
        nsamples: Number of times heat flux was sampled with GKMA during GPUMD simulation
        inputfile: Modal heat flux file output by GPUMD
        directory: Name of directory storing the input file to read
        directions: Directions to gather data from. Any order of 'xyz' is accepted. Excluding directions also allowed
            (i.e. 'xz' is accepted)
        ndiv: Integer used to shrink number of bins output. If originally have 10 bins, but want 5, ndiv=2. nbins/ndiv
            need not be an integer
        outputfile: File name to save read data to. Output file is a binary dictionary. Loading from a binary file is
            much faster than re-reading data files and saving is recommended
        save: Toggle saving data to binary dictionary. Loading from save file is much faster and recommended
        multiprocessing: Toggle using multi-core processing for conversion of text file
        ncore: Number of cores to use for multiprocessing. Ignored if multiprocessing is False
        block_size: Size of block (in bytes) to be read per read operation. File reading performance depend on this
            parameter and file size
        return_data: Toggle returning the loaded modal heat flux data. If this is False, the user should ensure that
            save is True

        Returns:
            Dictionary with all modal heat fluxes requested

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,nbins, nsamples, jmxi, jmxo, jmyi, jmyo, jmz
       **units**,N/A, N/A,|jm1|,|jm1|,|jm1|,|jm1|,|jm1|

    .. |jm1| replace:: eV\ :sup:`3/2` amu\ :sup:`-1/2` *x*\ :sup:`-1`

    Here *x* is the size of the bins in THz. For example, if there are 4 bins per THz, *x* = 0.25 THz.
    """
    jm_path = util.get_path(directory, inputfile)
    out_path = util.get_path(directory, outputfile)
    data = read_modal_analysis_file(nbins, nsamples, jm_path, ndiv, multiprocessing, ncore, block_size)
    out = dict()
    directions = util.get_direction(directions)
    if 'x' in directions:
        out['jmxi'] = data[:, :, 0]
        out['jmxo'] = data[:, :, 1]
    if 'y' in directions:
        out['jmyi'] = data[:, :, 2]
        out['jmyo'] = data[:, :, 3]
    if 'z' in directions:
        out['jmz'] = data[:, :, 4]

    out['nbins'] = nbins
    out['nsamples'] = nsamples

    if save:
        np.save(out_path, out)

    return out if return_data else None


def load_kappamode(nbins: int,
                   nsamples: int,
                   inputfile: str = "kappamode.out",
                   directory: str = None,
                   directions: str = "xyz",
                   ndiv: int = None,
                   outputfile: str = "kappamode.npy",
                   save: bool = False,
                   multiprocessing: bool = False,
                   ncore: int = None,
                   block_size: int = None,
                   return_data: bool = True) -> Union[None, Dict[str, np.ndarray]]:
    """
    Loads data from kappamode.out GPUMD file. Option to save as binary file for fast re-load later.
    WARNING: If using multiprocessing, memory useage may be significantly larger than file size

    Args:
        nbins: Number of bins used during the GPUMD simulation
        nsamples: Number of times thermal conductivity was sampled with HNEMA during GPUMD simulation
        inputfile: Modal thermal conductivity file output by GPUMD
        directory: Name of directory storing the input file to read
        directions: Directions to gather data from. Any order of 'xyz' is accepted. Excluding directions also allowed
            (i.e. 'xz' is accepted)
        ndiv: Integer used to shrink number of bins output. If originally have 10 bins, but want 5, ndiv=2. nbins/ndiv
            need not be an integer
        outputfile: File name to save read data to. Output file is a binary dictionary. Loading from a binary file is
            much faster than re-reading data files and saving is recommended
        save: Toggle saving data to binary dictionary. Loading from save file is much faster and recommended
        multiprocessing: Toggle using multi-core processing for conversion of text file
        ncore: Number of cores to use for multiprocessing. Ignored if multiprocessing is False
        block_size: Size of block (in bytes) to be read per read operation. File reading performance depend on this
            parameter and file size
        return_data: Toggle returning the loaded modal thermal conductivity data. If this is False, the user should
            ensure that save is True

        Returns:
            Dictionary with all modal thermal conductivities requested

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,nbins,nsamples,kmxi,kmxo,kmyi,kmyo,kmz
       **units**,N/A,N/A,|hn1|,|hn1|,|hn1|,|hn1|,|hn1|

    .. |hn1| replace:: Wm\ :sup:`-1` K\ :sup:`-1` *x*\ :sup:`-1`

    Here *x* is the size of the bins in THz. For example, if there are 4 bins per THz, *x* = 0.25 THz.
    """
    km_path = util.get_path(directory, inputfile)
    out_path = util.get_path(directory, outputfile)
    data = read_modal_analysis_file(nbins, nsamples, km_path, ndiv, multiprocessing, ncore, block_size)
    out = dict()
    directions = util.get_direction(directions)
    if 'x' in directions:
        out['kmxi'] = data[:, :, 0]
        out['kmxo'] = data[:, :, 1]
    if 'y' in directions:
        out['kmyi'] = data[:, :, 2]
        out['kmyo'] = data[:, :, 3]
    if 'z' in directions:
        out['kmz'] = data[:, :, 4]

    out['nbins'] = nbins
    out['nsamples'] = nsamples

    if save:
        np.save(out_path, out)

    return out if return_data else None


def load_saved_kappamode(filename='kappamode.npy', directory=None):
    """
    Loads data saved by the 'load_kappamode' function and returns the original dictionary.

    Args:
        filename (str):
            Name of the file to load

        directory (str):
            Directory the data file is located in

    Returns:
        dict: Dictionary with all modal thermal conductivities previously requested

    """
    path = util.get_path(directory, filename)
    return np.load(path, allow_pickle=True).item()


def load_saved_heatmode(filename='heatmode.npy', directory=None):
    """
    Loads data saved by the 'load_heatmode' or 'get_gkma_kappa' function and returns the original dictionary.

    Args:
        filename (str):
            Name of the file to load

        directory (str):
            Directory the data file is located in

    Returns:
        dict: Dictionary with all modal heat flux previously requested

    """

    path = util.get_path(directory, filename)
    return np.load(path, allow_pickle=True).item()


def get_frequency_info(bin_f_size: float, eigfile: str = "eigenvector.out", directory: str = None) -> dict:
    """
    Gathers eigen-frequency information from the eigenvector file and sorts
    it appropriately based on the selected frequency bins (identical to
    internal GPUMD representation).

    Args:
        bin_f_size: The frequency-based bin size (in THz)
        directory: Directory eigfile is stored
        eigfile: The filename of the eigenvector output/input file created by GPUMD phonon package

    Returns:
        Dictionary with the system eigen-freqeuency information along with binning information

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,fq,fmax,fmin,shift,nbins,bin_count,bin_f_size
       **units**,THz,THz,THz,N/A,N/A,N/A,THz
    """
    eigpath = os.path.join(directory, eigfile) if directory else os.path.join(os.getcwd(), eigfile)
    with open(eigpath, 'r') as eig_filehandle:
        om2 = [float(x) for x in eig_filehandle.readline().split()]

    epsilon = 1.e-6  # tolerance for float errors
    fq = np.sign(om2) * np.sqrt(abs(np.array(om2))) / (2 * np.pi)
    fmax = (np.floor(np.abs(fq[-1]) / bin_f_size) + 1) * bin_f_size
    fmin = np.floor(np.abs(fq[0]) / bin_f_size) * bin_f_size
    shift = int(np.floor(np.abs(fmin) / bin_f_size + epsilon))
    nbins = int(np.floor((fmax - fmin) / bin_f_size + epsilon))
    bin_count = np.zeros(nbins)
    for freq in fq:
        bin_count[int(np.floor(np.abs(freq) / bin_f_size) - shift)] += 1
    return {'fq': fq, 'fmax': fmax, 'fmin': fmin, 'shift': shift,
            'nbins': nbins, 'bin_count': bin_count, 'bin_f_size': bin_f_size}


def reduce_frequency_info(freq: dict, ndiv: int = 1) -> dict:
    """
    Recalculates frequency binning information based on how many times larger bins are wanted.

    Args:
        freq: Dictionary with frequency binning information from the get_frequency_info function output
        ndiv: Divisor used to shrink number of bins output. If originally have 10 bins, but want 5, ndiv=2. nbins/ndiv
            need not be an integer

    Returns:
        Dictionary with the system eigen freqeuency information along with binning information
    """
    epsilon = 1.e-6  # tolerance for float errors
    freq = copy.deepcopy(freq)
    freq['bin_f_size'] = freq['bin_f_size'] * ndiv
    freq['fmax'] = (np.floor(np.abs(freq['fq'][-1]) / freq['bin_f_size']) + 1) * freq['bin_f_size']
    nbins_new = int(np.ceil(freq['nbins'] / ndiv - epsilon))
    npad = nbins_new * ndiv - freq['nbins']
    freq['nbins'] = nbins_new
    freq['bin_count'] = np.pad(freq['bin_count'], [(0, npad)])
    freq['bin_count'] = np.sum(freq['bin_count'].reshape(-1, ndiv), axis=1)
    freq['ndiv'] = ndiv
    return freq
