import pyfftw
import multiprocessing
import numpy as np
from scipy.integrate import cumtrapz


def autocorrelate(f: np.ndarray, max_lag: float) -> np.ndarray:
    """
    Computes a fast autocorrelation function <f*f> and returns up to
    max_lag.

    Args:
        f: Vector for autocorrelation
        max_lag: Lag at which to calculate up to

    Returns:
        Autocorrelation vector
    """
    npoints = len(f)
    divisor = npoints - np.arange(npoints)
    # https://dsp.stackexchange.com/questions/741/why-should-i-zero-pad-a-signal-before-taking-the-fourier-transform
    f = np.pad(f, (0, npoints), 'constant', constant_values=(0, 0))
    fvi = np.zeros(2*npoints, dtype=type(f[0]))
    fwd = pyfftw.FFTW(f, fvi, flags=('FFTW_ESTIMATE',), threads=multiprocessing.cpu_count())
    fwd()
    inv_arg = np.conjugate(fvi)*fvi
    acf = np.zeros_like(inv_arg)
    rev = pyfftw.FFTW(inv_arg, acf, direction='FFTW_BACKWARD',
                      flags=('FFTW_ESTIMATE', ), threads=multiprocessing.cpu_count())
    rev()
    acf = acf[:npoints]/divisor
    return np.real(acf[:max_lag+1])


def correlate(f: np.ndarray, g: np.ndarray, max_lag: float) -> np.ndarray:
    """
    Computes fast correlation function <f*g> and returns up to max_lag.
    Assumes f and g are same length.

    Args:
        f: Vector for correlation
        g: Vector for correlation
        max_lag: Lag at which to calculate up to

    Returns:
        Correlation vector
    """
    if not len(f) == len(g):
        raise ValueError('corr arguments must be the same length.')

    npoints = len(f)
    divisor = npoints - np.arange(npoints)
    f = np.pad(f, (0, npoints), 'constant', constant_values=(0, 0))
    g = np.pad(g, (0, npoints), 'constant', constant_values=(0, 0))

    fvi = np.zeros(2*npoints, dtype=type(f[0]))
    gvi = np.zeros(2*npoints, dtype=type(g[0]))

    fwd_f = pyfftw.FFTW(f, fvi, flags=('FFTW_ESTIMATE',), threads=multiprocessing.cpu_count())
    fwd_f()

    fwd_g = pyfftw.FFTW(g, gvi, flags=('FFTW_ESTIMATE',), threads=multiprocessing.cpu_count())
    fwd_g()

    inv_arg = np.conjugate(fvi)*gvi
    cf = np.zeros_like(inv_arg)
    rev = pyfftw.FFTW(inv_arg, cf, direction='FFTW_BACKWARD',
                      flags=('FFTW_ESTIMATE', ), threads=multiprocessing.cpu_count())
    rev()
    cf = cf[:npoints]/divisor
    return np.real(cf[:max_lag+1])


def running_ave(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """
    Args:
        y: Dependent variable
        x: Independent variable

    Returns:
        Running average of y
    """
    return cumtrapz(y, x, initial=0) / x