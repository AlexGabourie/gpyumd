import numpy as np
import pandas as pd
import os
import copy
import multiprocessing as mp
from functools import partial
from collections import deque
from typing import BinaryIO, List

from gpyumd.util import get_path, get_direction, check_list, check_range

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"

#########################################
# Helper Functions
#########################################


def __process_sample(nbins, i):
    """
    A helper function for the multiprocessing of kappamode.out files

    Args:
        nbins (int):
            Number of bins used in the GPUMD simulation

        i (int):
            The current sample from a run to analyze

    Returns:
        np.ndarray: A 2D array of each bin and output for a sample


    """
    out = list()
    for j in range(nbins):
        out += [float(x) for x in malines[j + i * nbins].split()]
    return np.array(out).reshape((nbins,5))


def tail(file_handle: BinaryIO, nlines: int, block_size: int = 32768) -> List[bytes]:
    """
    Reads the last nlines of a file.

    Args:
        file_handle: File handle of file to be read
        nlines: Number of lines to be read from end of file
        block_size: Size of block (in bytes) to be read per read operation. Performance depends on this parameter and
            file size.

    Returns:
        final nlines of file

    Additional Information:
    Since GPUMD output files are mostly append-only, this becomes
    useful when a simulation prematurely ends (i.e. cluster preempts
    run, but simulation restarts elsewhere). In this case, it is not
    necessary to clean the directory before re-running. File outputs
    will be too long (so there still is a storage concern), but the
    proper data can be extracted from the end of file.
    This may also be useful if you want to only grab data from the
    final m number of runs of the simulation
    """
    # block_size is in bytes (must decode to string)
    file_handle.seek(0, 2)
    bytes_remaining = file_handle.tell()
    idx = -block_size
    blocks = list()
    # Make no assumptions about line length
    lines_left = nlines
    end_of_file = False
    first = True

    # block_size is smaller than file
    if block_size <= bytes_remaining:
        while lines_left > 0 and not end_of_file:
            if bytes_remaining > block_size:
                file_handle.seek(idx, 2)
                blocks.append(file_handle.read(block_size))
            else:  # if reached end of file
                file_handle.seek(0, 0)
                blocks.append(file_handle.read(bytes_remaining))
                end_of_file = True

            idx -= block_size
            bytes_remaining -= block_size
            num_lines = blocks[-1].count(b'\n')
            if first:
                lines_left -= num_lines - 1
                first = False
            else:
                lines_left -= num_lines

            # since whitespace removed from end_of_file, must compare to 1 here
            if end_of_file and lines_left > 1:
                raise ValueError("More lines requested than exist.")

        # Corrects for reading too many lines with large buffer
        if bytes_remaining > 0:
            skip = 1 + abs(lines_left)
            blocks[-1] = blocks[-1].split(b'\n', skip)[skip]
        text = b''.join(reversed(blocks)).strip()
    else:  # block_size is bigger than file
        file_handle.seek(0, 0)
        block = file_handle.read()
        num_lines = block.count(b'\n')
        if num_lines < nlines:
            raise ValueError("More lines requested than exist.")
        skip = num_lines - nlines
        text = block.split(b'\n', skip)[skip].strip()
    return text.split(b'\n')


def __modal_analysis_read(nbins, nsamples, datapath,
                          ndiv, multiprocessing, ncore, block_size):

    global malines
    # Get full set of results
    datalines = nbins * nsamples
    with open(datapath, 'rb') as f:
        if multiprocessing:
            malines = tail(f, datalines, block_size=block_size)
        else:
            malines = deque(tail(f, datalines, block_size=block_size))

    if multiprocessing:  # TODO Improve memory efficiency of multiprocessing
        if not ncore:
            ncore = mp.cpu_count()

        func = partial(__process_sample, nbins)
        pool = mp.Pool(ncore)
        data = np.array(pool.map(func, range(nsamples)), dtype='float32').transpose((1, 0, 2))
        pool.close()

    else:  # Faster if single thread
        data = np.zeros((nbins, nsamples, 5), dtype='float32')
        for j in range(nsamples):
            for i in range(nbins):
                measurements = malines.popleft().split()
                data[i, j, 0] = float(measurements[0])
                data[i, j, 1] = float(measurements[1])
                data[i, j, 2] = float(measurements[2])
                data[i, j, 3] = float(measurements[3])
                data[i, j, 4] = float(measurements[4])

    del malines
    if ndiv:
        nbins = int(np.ceil(data.shape[0] / ndiv))  # overwrite nbins
        npad = nbins * ndiv - data.shape[0]
        data = np.pad(data, [(0, npad), (0, 0), (0, 0)])
        data = np.sum(data.reshape((-1, ndiv, data.shape[1], data.shape[2])), axis=1)

    return data


def __basic_reader(points, data, labels):
    start = 0
    out = dict()
    for i, npoints in enumerate(points):
        end = start + npoints
        run = dict()
        for j, key in enumerate(labels):
            run[key] = data[j][start:end].to_numpy(dtype='float')
        start = end
        out['run{}'.format(i)] = run
    return out


def __basic_frame_loader(n, directory, filename):
    path = get_path(directory, filename)
    data = pd.read_csv(path, delim_whitespace=True, header=None).to_numpy(dtype='float')
    if not (data.shape[0] / n).is_integer():
        raise ValueError("An integer number of frames cannot be created. Please check n.")
    return data.reshape(-1, n, 3)


#########################################
# Data-loading Related
#########################################

def load_omega2(directory=None, filename='omega2.out'):
    """
    Loads data from omega2.out GPUMD output file.\n

    Args:
        directory (str):
            Directory to load force file from

        filename (str):
            Name of force data file

    Returns:
        Numpy array of shape (N_kpoints,3*N_basis) in units of THz. N_kpoints is number of k points in kpoint.in and
        N_basis is the number of basis atoms defined in basis.in

    """
    path = get_path(directory, filename)
    data = pd.read_csv(path, delim_whitespace=True, header=None).to_numpy(dtype='float')
    data = np.sqrt(data)/(2*np.pi)
    return data


def load_force(n, directory=None, filename='force.out'):
    """
    Loads data from force.out GPUMD output file.\n
    Currently supports loading a single run.

    Args:
        n (int):
            Number of atoms force is output for

        directory (str):
            Directory to load force file from

        filename (str):
            Name of force data file

    Returns:
        Numpy array of shape (-1,n,3) containing all forces (ev/A) from filename

    """
    return __basic_frame_loader(n, directory, filename)


def load_velocity(n, directory=None, filename='velocity.out'):
    """
    Loads data from velocity.out GPUMD output file.\n
    Currently supports loading a single run.

    Args:
        n (int):
            Number of atoms velocity is output for

        directory (str):
            Directory to load velocity file from

        filename (str):
            Name of velocity data file

    Returns:
        Numpy array of shape (-1,n,3) containing all forces (A/ps) from filename

    """
    return __basic_frame_loader(n, directory, filename)


def load_compute(quantities=None, directory=None, filename='compute.out'):
    """
    Loads data from compute.out GPUMD output file.\n
    Currently supports loading a single run.

    Args:
        quantities (str or list(str)):
            Quantities to extract from compute.out Accepted quantities are:\n
            ['temperature', 'U', 'F', 'W', 'jp', 'jk']. \n
            Other quantity will be ignored.\n
            temperature=temperature, U=potential, F=force, W=virial, jp=heat current (potential), jk=heat current (kinetic)

        directory (str):
            Directory to load compute file from

        filename (str):
            file to load compute from

    Returns:
        Dictionary containing the data from compute.out

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,temperature,U,F,W,jp,jk,Ein,Eout
       **units**,K,eV,|c1|,eV,|c2|,|c2|,eV,eV

   .. |c1| replace:: eVA\ :sup:`-1`
   .. |c2| replace:: eV\ :sup:`3/2` amu\ :sup:`-1/2`
    """
    # TODO Add input checking
    if not quantities:
        return None
    compute_path = get_path(directory, filename)
    data = pd.read_csv(compute_path, delim_whitespace=True, header=None)

    num_col = len(data.columns)
    q_count = {'temperature': 1, 'U': 1, 'F': 3, 'W': 3, 'jp': 3, 'jk': 3}
    out = dict()

    count = 0
    for value in quantities:
        count += q_count[value]

    m = int(num_col / count)
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

    return out


def load_thermo(directory=None, filename='thermo.out'):
    """
    Loads data from thermo.out GPUMD output file.

    Args:
        directory (str):
            Directory to load thermal data file from

        filename (str):
            Name of thermal data file

        Returns:
            'output' dictionary containing the data from thermo.out

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,temperature,K,U,Px,Py,Pz,Lx,Ly,Lz,ax,ay,az,bx,by,bz,cx,cy,cz
       **units**,K,eV,eV,GPa,GPa,GPa,A,A,A,A,A,A,A,A,A,A,A,A

    """
    thermo_path = get_path(directory, filename)
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


def load_heatmode(nbins, nsamples, directory=None,
                  inputfile='heatmode.out', directions='xyz',
                  outputfile='heatmode.npy', ndiv=None, save=False,
                  multiprocessing=False, ncore=None, block_size=65536, return_data=True):
    """
    Loads data from heatmode.out GPUMD file. Option to save as binary file for fast re-load later.
    WARNING: If using multiprocessing, memory usage may be significantly larger than file size

    Args:
        nbins (int):
            Number of bins used during the GPUMD simulation

        nsamples (int):
            Number of times heat flux was sampled with GKMA during GPUMD simulation

        directory (str):
            Name of directory storing the input file to read

        inputfile (str):
            Modal heat flux file output by GPUMD

        directions (str):
            Directions to gather data from. Any order of 'xyz' is accepted. Excluding directions also allowed (i.e. 'xz'
            is accepted)

        outputfile (str):
            File name to save read data to. Output file is a binary dictionary. Loading from a binary file is much
            faster than re-reading data files and saving is recommended

        ndiv (int):
            Integer used to shrink number of bins output. If originally have 10 bins, but want 5, ndiv=2. nbins/ndiv
            need not be an integer

        save (bool):
            Toggle saving data to binary dictionary. Loading from save file is much faster and recommended

        multiprocessing (bool):
            Toggle using multi-core processing for conversion of text file

        ncore (bool):
            Number of cores to use for multiprocessing. Ignored if multiprocessing is False

        block_size (int):
            Size of block (in bytes) to be read per read operation. File reading performance depend on this parameter
            and file size

        return_data (bool):
            Toggle returning the loaded modal heat flux data. If this is False, the user should ensure that
            save is True


        Returns:
                dict: Dictionary with all modal heat fluxes requested

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,nbins, nsamples, jmxi, jmxo, jmyi, jmyo, jmz
       **units**,N/A, N/A,|jm1|,|jm1|,|jm1|,|jm1|,|jm1|

    .. |jm1| replace:: eV\ :sup:`3/2` amu\ :sup:`-1/2` *x*\ :sup:`-1`


    Here *x* is the size of the bins in THz. For example, if there are 4 bins per THz, *x* = 0.25 THz.
    """
    jm_path = get_path(directory, inputfile)
    out_path = get_path(directory, outputfile)
    data = __modal_analysis_read(nbins, nsamples, jm_path, ndiv, multiprocessing, ncore, block_size)
    out = dict()
    directions = get_direction(directions)
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

    if return_data:
        return out
    return


def load_kappamode(nbins, nsamples, directory=None,
                   inputfile='kappamode.out', directions='xyz',
                   outputfile='kappamode.npy', ndiv=None, save=False,
                   multiprocessing=False, ncore=None, block_size=65536, return_data=True):
    """
    Loads data from kappamode.out GPUMD file. Option to save as binary file for fast re-load later.
    WARNING: If using multiprocessing, memory useage may be significantly larger than file size

    Args:
        nbins (int):
            Number of bins used during the GPUMD simulation

        nsamples (int):
            Number of times thermal conductivity was sampled with HNEMA during GPUMD simulation

        directory (str):
            Name of directory storing the input file to read

        inputfile (str):
            Modal thermal conductivity file output by GPUMD

        directions (str):
            Directions to gather data from. Any order of 'xyz' is accepted. Excluding directions also allowed (i.e. 'xz'
            is accepted)

        outputfile (str):
            File name to save read data to. Output file is a binary dictionary. Loading from a binary file is much
            faster than re-reading data files and saving is recommended

        ndiv (int):
            Integer used to shrink number of bins output. If originally have 10 bins, but want 5, ndiv=2. nbins/ndiv
            need not be an integer

        save (bool):
            Toggle saving data to binary dictionary. Loading from save file is much faster and recommended

        multiprocessing (bool):
            Toggle using multi-core processing for conversion of text file

        ncore (bool):
            Number of cores to use for multiprocessing. Ignored if multiprocessing is False

        block_size (int):
            Size of block (in bytes) to be read per read operation. File reading performance depend on this parameter
            and file size

        return_data (bool):
            Toggle returning the loaded modal thermal conductivity data. If this is False, the user should ensure that
            save is True


        Returns:
                dict: Dictionary with all modal thermal conductivities requested

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,nbins,nsamples,kmxi,kmxo,kmyi,kmyo,kmz
       **units**,N/A,N/A,|hn1|,|hn1|,|hn1|,|hn1|,|hn1|

    .. |hn1| replace:: Wm\ :sup:`-1` K\ :sup:`-1` *x*\ :sup:`-1`

    Here *x* is the size of the bins in THz. For example, if there are 4 bins per THz, *x* = 0.25 THz.
    """
    km_path = get_path(directory, inputfile)
    out_path = get_path(directory, outputfile)
    data = __modal_analysis_read(nbins, nsamples, km_path, ndiv, multiprocessing, ncore, block_size)
    out = dict()
    directions = get_direction(directions)
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

    if return_data:
        return out
    return


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
    path = get_path(directory, filename)
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

    path = get_path(directory, filename)
    return np.load(path, allow_pickle=True).item()


def load_sdc(nc, directory=None, filename='sdc.out'):
    """
    Loads data from sdc.out GPUMD output file.

    Args:
        nc (int or list(int)):
            Number of time correlation points the VAC/SDC is computed for

        directory (str):
            Directory to load 'sdc.out' file from (dir. of simulation)

        filename (str):
            File to load SDC from

    Returns:
        dict(dict):
            Dictonary with SDC/VAC data. The outermost dictionary stores each individual run

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,t,VACx,VACy,VACz,SDCx,SDCy,SDCz
       **units**,ps,|sd1|,|sd1|,|sd1|,|sd2|,|sd2|,|sd2|

    .. |sd1| replace:: A\ :sup:`2` ps\ :sup:`-2`
    .. |sd2| replace:: A\ :sup:`2` ps\ :sup:`-1`
    """
    nc = check_list(nc, varname='nc', dtype=int)
    sdc_path = get_path(directory, filename)
    data = pd.read_csv(sdc_path, delim_whitespace=True, header=None)
    check_range(nc, data.shape[0])
    labels = ['t', 'VACx', 'VACy', 'VACz', 'SDCx', 'SDCy', 'SDCz']
    return __basic_reader(nc, data, labels)


def load_vac(nc, directory=None, filename='mvac.out'):
    """
    Loads data from mvac.out GPUMD output file.

    Args:
        nc (int or list(int)):
            Number of time correlation points the VAC is computed for

        directory (str):
            Directory to load 'mvac.out' file from

        filename (str):
            File to load VAC from

    Returns:
        dict(dict):
            Dictonary with VAC data. The outermost dictionary stores each individual run

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,t,VACx,VACy,VACz
       **units**,ps,|v1|,|v1|,|v1|

    .. |v1| replace:: A\ :sup:`2` ps\ :sup:`-2`
    """
    nc = check_list(nc, varname='nc', dtype=int)
    sdc_path = get_path(directory, filename)
    data = pd.read_csv(sdc_path, delim_whitespace=True, header=None)
    check_range(nc, data.shape[0])
    labels = ['t', 'VACx', 'VACy', 'VACz']
    return __basic_reader(nc, data, labels)


def load_dos(num_dos_points, directory=None, filename='dos.out'):
    """
    Loads data from dos.out GPUMD output file.

    Args:
        num_dos_points (int or list(int)):
            Number of frequency points the DOS is computed for.

        directory (str):
            Directory to load 'dos.out' file from (dir. of simulation)

        filename (str):
            File to load DOS from.

    Returns:
        dict(dict)): Dictonary with DOS data. The outermost dictionary stores
        each individual run.

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,nu,DOSx,DOSy,DOSz
       **units**,THz,|d1|,|d1|,|d1|

    .. |d1| replace:: THz\ :sup:`-1`

    """
    num_dos_points = check_list(num_dos_points, varname='num_dos_points', dtype=int)
    dos_path = get_path(directory, filename)
    data = pd.read_csv(dos_path, delim_whitespace=True, header=None)
    check_range(num_dos_points, data.shape[0])
    labels = ['nu', 'DOSx', 'DOSy', 'DOSz']
    out = __basic_reader(num_dos_points, data, labels)
    for key in out.keys():
        out[key]['nu'] /= (2 * np.pi)
    return out


def load_shc(nc, num_omega, directory=None, filename='shc.out'):
    """
    Loads the data from shc.out GPUMD output file.

    Args:
        nc (int or list(int)):
            Maximum number of correlation steps. If multiple shc runs, can provide a list of nc.

        num_omega (int or list(int)):
            Number of frequency points. If multiple shc runs, can provide a list of num_omega.

        directory (str):
            Directory to load 'shc.out' file from (dir. of simulation)

        filename (str):
            File to load SHC from.

    Returns:
        dict: Dictionary of in- and out-of-plane shc results (average)

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,t, Ki, Ko, nu, jwi, jwo
       **units**,ps, |sh1|,|sh1|, THz, |sh2|, |sh2|

    .. |sh1| replace:: A eV ps\ :sup:`-1`
    .. |sh2| replace:: A eV ps\ :sup:`-1` THz\ :sup:`-1`
    """

    nc = check_list(nc, varname='nc', dtype=int)
    num_omega = check_list(num_omega, varname='num_omega', dtype=int)
    if not len(nc) == len(num_omega):
        raise ValueError('nc and num_omega must be the same length.')
    shc_path = get_path(directory, filename)
    data = pd.read_csv(shc_path, delim_whitespace=True, header=None)
    check_range(np.array(nc) * 2 - 1 + np.array(num_omega), data.shape[0])
    if not all([i>0 for i in nc]) or not all([i > 0 for i in num_omega]):
        raise ValueError('Only strictly positive numbers are allowed.')
    labels_corr = ['t', 'Ki', 'Ko']
    labels_omega = ['nu', 'jwi', 'jwo']

    out = dict()
    start = 0
    for i, varlen in enumerate(zip(nc, num_omega)):
        run = dict()
        Nc_i = varlen[0] * 2 - 1
        num_omega_i = varlen[1]
        end = start + Nc_i
        for j, key in enumerate(labels_corr):
            run[key] = data[j][start:end].to_numpy(dtype='float')
        start = end
        end += num_omega_i
        for j, key in enumerate(labels_omega):
            run[key] = data[j][start:end].to_numpy(dtype='float')
        run['nu'] /= (2 * np.pi)
        start = end
        out['run{}'.format(i)] = run
    return out


def load_kappa(directory=None, filename='kappa.out'):
    """
    Loads data from kappa.out GPUMD output file which contains HNEMD kappa.

    Args:
        directory (str):
            Directory containing kappa data file

        filename (str):
            The kappa data file

    Returns:
        dict: A dictionary with keys corresponding to the columns in 'kappa.out'

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,kxi, kxo, kyi, kyo, kz
       **units**,|k1|,|k1|,|k1|,|k1|,|k1|

    .. |k1| replace:: Wm\ :sup:`-1` K\ :sup:`-1`

    """

    kappa_path = get_path(directory, filename)
    data = pd.read_csv(kappa_path, delim_whitespace=True, header=None)
    labels = ['kxi', 'kxo', 'kyi', 'kyo', 'kz']
    out = dict()
    for i, key in enumerate(labels):
        out[key] = data[i].to_numpy(dtype='float')
    return out


def load_hac(nc, output_interval, directory=None, filename='hac.out'):
    """
    Loads data from hac.out GPUMD output file.

    Args:
        nc (int or list(int)):
            Number of correlation steps

        output_interval (int or list(int)):
            Output interval for HAC and RTC data

        directory (str):
            Directory containing hac data file

        filename (str):
            The hac data file

    Returns:
        dict: A dictionary containing the data from hac runs

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,t, kxi, kxo, kyi, kyo, kz, jxijx, jxojx, jyijy, jyojy, jzjz
       **units**,ps,|h1|,|h1|,|h1|,|h1|,|h1|,|h2|,|h2|,|h2|,|h2|,|h2|

    .. |h1| replace:: Wm\ :sup:`-1` K\ :sup:`-1`
    .. |h2| replace:: eV\ :sup:`3` amu\ :sup:`-1`
    """

    nc = check_list(nc, varname='nc', dtype=int)
    output_interval = check_list(output_interval, varname='output_interval', dtype=int)
    if not len(nc) == len(output_interval):
        raise ValueError('nc and output_interval must be the same length.')

    npoints = [int(x / y) for x, y in zip(nc, output_interval)]
    hac_path = get_path(directory, filename)
    data = pd.read_csv(hac_path, delim_whitespace=True, header=None)
    check_range(npoints, data.shape[0])
    labels = ['t', 'jxijx', 'jxojx', 'jyijy', 'jyojy', 'jzjz',
              'kxi', 'kxo', 'kyi', 'kyo', 'kz']
    start = 0
    out = dict()
    for i, varlen in enumerate(npoints):
        end = start + varlen
        run = dict()
        for j, key in enumerate(labels):
            run[key] = data[j][start:end].to_numpy(dtype='float')
        start = end
        out['run{}'.format(i)] = run
    return out


def get_frequency_info(bin_f_size, eigfile='eigenvector.out', directory=None):
    """
    Gathers eigen-frequency information from the eigenvector file and sorts
    it appropriately based on the selected frequency bins (identical to
    internal GPUMD representation).

    Args:
        bin_f_size (float):
            The frequency-based bin size (in THz)

        eigfile (str):
            The filename of the eigenvector output/input file created by GPUMD
            phonon package

        directory (str):
            Directory eigfile is stored

    Returns:
        dict: Dictionary with the system eigen-freqeuency information along
        with binning information

    .. csv-table:: Output dictionary
       :stub-columns: 1

       **key**,fq,fmax,fmin,shift,nbins,bin_count,bin_f_size
       **units**,THz,THz,THz,N/A,N/A,N/A,THz
    """
    if not directory:
        eigpath = os.path.join(os.getcwd(), eigfile)
    else:
        eigpath = os.path.join(directory, eigfile)

    with open(eigpath, 'r') as f:
        om2 = [float(x) for x in f.readline().split()]

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


def reduce_frequency_info(freq, ndiv=1):
    """
    Recalculates frequency binning information based on how many times larger bins are wanted.

    Args:
        freq (dict): Dictionary with frequency binning information from the get_frequency_info function output

        ndiv (int):
            Integer used to shrink number of bins output. If originally have 10 bins, but want 5, ndiv=2. nbins/ndiv
            need not be an integer

    Returns:
        dict: Dictionary with the system eigen freqeuency information along with binning information

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
