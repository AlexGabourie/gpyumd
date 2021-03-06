from typing import Union, Dict
import numpy as np
from gpyumd import util, math
from scipy.integrate import cumtrapz
import copy

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


def calc_gkma_kappa(data: dict,
                    dt: float,
                    sample_interval: int,
                    temperature: float = 300,
                    vol: float = 1,
                    max_tau: float = None,
                    directions: str = "xyz",
                    outputfile: str = "heatmode.npy",
                    save: bool = False,
                    directory: str = None) -> Union[None, Dict[str, np.ndarray]]:
    """
    Calculate the Green-Kubo thermal conductivity from modal heat current
    data from 'load_heatmode'

    Args:
        data: Dictionary with heat currents loaded by 'load_heatmode'
        dt: Time step during data collection in fs
        sample_interval: Number of time steps per sample of modal heat
         flux
        temperature: Temperature of system during data collection
        vol: Volume of system in angstroms^3
        max_tau: Correlation time to calculate up to. Units of ns
        directions: Directions to gather data from. Any order of 'xyz'
         is accepted. Excluding directions also allowed (i.e. 'xz'
         is accepted)
        outputfile: File name to save read data to. Output file is a
         binary dictionary. Loading from a binary file is much
         faster than re-reading data files and saving is recommended
        save: Toggle saving data to binary dictionary. Loading from
         save file is much faster and recommended
        directory: Name of directory storing the input file to read
    """
    nbins = data['nbins']
    nsamples = data['nsamples']

    def kappa_scaling() -> float:  # Keep to understand unit conversion
        # Units:     eV^3/amu -> Jm^2/s^2*eV         fs -> s       K/(eV*Ang^3) -> K/(eV*m^3) w/ Boltzmann
        scaling = (1.602176634e-19 * 9.651599e7) * (1. / 1.e15) * (1.e30 / 8.617333262145e-5)
        return scaling / (temperature * temperature * vol)

    out_path = util.get_path(directory, outputfile)
    scale = kappa_scaling()
    # set the heat flux sampling time: rate * timestep * scaling
    srate = sample_interval * dt  # [fs]

    # Calculate total time
    tot_time = srate * (nsamples - 1)  # [fs]

    # set the integration limit (i.e. tau)
    if max_tau is None:
        max_tau = tot_time  # [fs]
    else:
        max_tau = max_tau * 1e6  # [fs]

    max_lag = int(np.floor(max_tau / srate))
    size = max_lag + 1
    data['tau'] = np.squeeze(np.linspace(0, max_lag * srate, max_lag + 1))  # [ns]

    # AUTOCORRELATION #
    directions = util.get_direction(directions)
    cplx = np.complex128
    # Note: loops necessary due to memory constraints
    #  (can easily max out cluster mem.)
    if 'x' in directions:
        if 'jmxi' not in data.keys() or 'jmxo' not in data.keys():
            raise ValueError("x direction data is missing")

        jx = np.sum(data['jmxi']+data['jmxo'], axis=0)
        data['jmxijx'] = np.zeros((nbins, size))
        data['jmxojx'] = np.zeros((nbins, size))
        data['kmxi'] = np.zeros((nbins, size))
        data['kmxo'] = np.zeros((nbins, size))
        for m in range(nbins):
            data['jmxijx'][m, :] = math.correlate(data['jmxi'][m, :].astype(cplx), jx.astype(cplx), max_lag)
            data['kmxi'][m, :] = cumtrapz(data['jmxijx'][m, :], data['tau'], initial=0) * scale

            data['jmxojx'][m, :] = math.correlate(data['jmxo'][m, :].astype(cplx), jx.astype(cplx), max_lag)
            data['kmxo'][m, :] = cumtrapz(data['jmxojx'][m, :], data['tau'], initial=0) * scale
        del jx

    if 'y' in directions:
        if 'jmyi' not in data.keys() or 'jmyo' not in data.keys():
            raise ValueError("y direction data is missing")

        jy = np.sum(data['jmyi']+data['jmyo'], axis=0)
        data['jmyijy'] = np.zeros((nbins, size))
        data['jmyojy'] = np.zeros((nbins, size))
        data['kmyi'] = np.zeros((nbins, size))
        data['kmyo'] = np.zeros((nbins, size))
        for m in range(nbins):
            data['jmyijy'][m, :] = math.correlate(data['jmyi'][m, :].astype(cplx), jy.astype(cplx), max_lag)
            data['kmyi'][m, :] = cumtrapz(data['jmyijy'][m, :], data['tau'], initial=0) * scale

            data['jmyojy'][m, :] = math.correlate(data['jmyo'][m, :].astype(cplx), jy.astype(cplx), max_lag)
            data['kmyo'][m, :] = cumtrapz(data['jmyojy'][m, :], data['tau'], initial=0) * scale
        del jy

    if 'z' in directions:
        if 'jmz' not in data.keys():
            raise ValueError("z direction data is missing")

        jz = np.sum(data['jmz'], axis=0)
        data['jmzjz'] = np.zeros((nbins, size))
        data['kmz'] = np.zeros((nbins, size))
        for m in range(nbins):
            data['jmzjz'][m, :] = math.correlate(data['jmz'][m, :].astype(cplx), jz.astype(cplx), max_lag)
            data['kmz'][m, :] = cumtrapz(data['jmzjz'][m, :], data['tau'], initial=0) * scale
        del jz

    data['tau'] = data['tau'] / 1.e6

    if save:
        np.save(out_path, data)


def calc_spectral_kappa(shc: dict, driving_force: float, temperature: float, volume: float) -> None:
    """
    Spectral thermal conductivity calculation from the spectral heat
    current from an SHC run. Updates the shc dict from data.load_shc()

    Args:
        shc: The data from a single SHC run as output by load_shc
        driving_force: HNEMD force in (1/A)
        temperature: HNEMD run temperature (K)
        volume: Volume (A^3) during HNEMD run

    Returns:
        New dict entries of spectral thermal conductivity. Units are [kwi,
         kwo -> W(m^-1)(K^-1)(THz^-1)].
    """
    if 'jwi' not in shc.keys() or 'jwo' not in shc.keys():
        raise ValueError("shc argument must be from load_shc and contain in/out heat currents.")

    # ev*A/ps/THz * 1/A^3 *1/K * A ==> W/m/K/THz
    convert = 1602.17662
    shc['kwi'] = shc['jwi'] * convert / (driving_force * temperature * volume)
    shc['kwo'] = shc['jwo'] * convert / (driving_force * temperature * volume)


def calc_reduced_freq_info(freq: dict, ndiv: int = 1) -> dict:
    """
    Recalculates modal analysis frequency binning information based on how
    many times larger bins are wanted.

    Args:
        freq: Dictionary with frequency binning information from the
         get_frequency_info function output
        ndiv: Divisor used to shrink number of bins output. If
         originally have 10 bins, but want 5, ndiv=2. nbins/ndiv need
         not be an integer

    Returns:
        Dictionary with the system eigen freqeuency information along
         with binning information
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
