from typing import Union, Dict
import numpy as np
from scipy.integrate import cumtrapz

from gpyumd.util import get_direction, get_path
from gpyumd.math.correlate import correlation

__author__ = "Alexander Gabourie"
__email__ = "agabourie47@gmail.com"


def get_gkma_kappa(data: dict,
                   nbins: int,
                   nsamples: int,
                   dt: float,
                   sample_interval: int,
                   temperature: float = 300,
                   vol: float = 1,
                   max_tau: float = None,
                   directions: str = "xyz",
                   outputfile: str = "heatmode.npy",
                   save: bool = False,
                   directory: str = None,
                   return_data: bool = True) -> Union[None, Dict[str, np.ndarray]]:
    """
    Calculate the Green-Kubo thermal conductivity from modal heat current data from 'load_heatmode'

    Args:
        data: Dictionary with heat currents loaded by 'load_heatmode'
        nbins: Number of bins used during the GPUMD simulation
        nsamples: Number of times heat flux was sampled with GKMA during GPUMD simulation
        dt: Time step during data collection in fs
        sample_interval: Number of time steps per sample of modal heat flux
        temperature: Temperature of system during data collection
        vol: Volume of system in angstroms^3
        max_tau: Correlation time to calculate up to. Units of ns
        directions: Directions to gather data from. Any order of 'xyz' is accepted. Excluding directions also allowed
            (i.e. 'xz' is accepted)
        outputfile: File name to save read data to. Output file is a binary dictionary. Loading from a binary file is
            much faster than re-reading data files and saving is recommended
        save: Toggle saving data to binary dictionary. Loading from save file is much faster and recommended
        directory: Name of directory storing the input file to read
        return_data: Toggle returning the loaded modal heat flux data. If this is False, the user should ensure that
            save is True

    Returns:
        Input data dict but with correlation, thermal conductivity, and lag time data included

    .. csv-table:: Output dictionary (new entries)
       :stub-columns: 1

       **key**,tau,kmxi,kmxo,kmyi,kmyo,kmz,jmxijx,jmxojx,jmyijy,jmyojy,jmzjz
       **units**,ns,|gk1|,|gk1|,|gk1|,|gk1|,|gk1|,|gk2|,|gk2|,|gk2|,|gk2|,|gk2|

    .. |gk1| replace:: Wm\ :sup:`-1` K\ :sup:`-1` *x*\ :sup:`-1`
    .. |gk2| replace:: eV\ :sup:`3` amu\ :sup:`-1` *x*\ :sup:`-1`

    Here *x* is the size of the bins in THz. For example, if there are 4 bins per THz, *x* = 0.25 THz.
    """
    def kappa_scaling() -> float:  # Keep to understand unit conversion
        # Units:     eV^3/amu -> Jm^2/s^2*eV         fs -> s       K/(eV*Ang^3) -> K/(eV*m^3) w/ Boltzmann
        scaling = (1.602176634e-19 * 9.651599e7) * (1. / 1.e15) * (1.e30 / 8.617333262145e-5)
        return scaling / (temperature * temperature * vol)

    out_path = get_path(directory, outputfile)
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
    directions = get_direction(directions)
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
            data['jmxijx'][m, :] = correlation(data['jmxi'][m, :].astype(cplx), jx.astype(cplx), max_lag)
            data['kmxi'][m, :] = cumtrapz(data['jmxijx'][m, :], data['tau'], initial=0) * scale

            data['jmxojx'][m, :] = correlation(data['jmxo'][m, :].astype(cplx), jx.astype(cplx), max_lag)
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
            data['jmyijy'][m, :] = correlation(data['jmyi'][m, :].astype(cplx), jy.astype(cplx), max_lag)
            data['kmyi'][m, :] = cumtrapz(data['jmyijy'][m, :], data['tau'], initial=0) * scale

            data['jmyojy'][m, :] = correlation(data['jmyo'][m, :].astype(cplx), jy.astype(cplx), max_lag)
            data['kmyo'][m, :] = cumtrapz(data['jmyojy'][m, :], data['tau'], initial=0) * scale
        del jy

    if 'z' in directions:
        if 'jmz' not in data.keys():
            raise ValueError("z direction data is missing")

        jz = np.sum(data['jmz'], axis=0)
        data['jmzjz'] = np.zeros((nbins, size))
        data['kmz'] = np.zeros((nbins, size))
        for m in range(nbins):
            data['jmzjz'][m, :] = correlation(data['jmz'][m, :].astype(cplx), jz.astype(cplx), max_lag)
            data['kmz'][m, :] = cumtrapz(data['jmzjz'][m, :], data['tau'], initial=0) * scale
        del jz

    data['tau'] = data['tau'] / 1.e6

    if save:
        np.save(out_path, data)

    return data if return_data else None


def running_ave(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Args:
        y: Dependent variable
        x: Independent variable

    Returns:
        Running average of y
    """
    return cumtrapz(y, x, initial=0) / x


def hnemd_spectral_kappa(shc: dict, driving_force: float, temperature: float, volume: float) -> None:
    """
    Spectral thermal conductivity calculation from the spectral heat current from an shc run. Updates the shc dict from
    data.load_shc()

    Args:
        shc: The data from a single SHC run as output by thermo.gpumd.data.load_shc
        driving_force: HNEMD force in (1/A)
        temperature: HNEMD run temperature (K)
        volume: Volume (A^3) during HNEMD run

    .. csv-table:: Output dictionary (new entries)
       :stub-columns: 1

       **key**,kwi,kwo
       **units**,|sh3|,|sh3|

    .. |sh3| replace:: Wm\ :sup:`-1` K\ :sup:`-1` THz\ :sup:`-1`
    """
    if 'jwi' not in shc.keys() or 'jwo' not in shc.keys():
        raise ValueError("shc argument must be from load_shc and contain in/out heat currents.")

    # ev*A/ps/THz * 1/A^3 *1/K * A ==> W/m/K/THz
    convert = 1602.17662
    shc['kwi'] = shc['jwi'] * convert / (driving_force * temperature * volume)
    shc['kwo'] = shc['jwo'] * convert / (driving_force * temperature * volume)
