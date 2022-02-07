import pyfftw
import multiprocessing
import numpy as np


def autocorr(f, max_lag):
    """
    Computes a fast autocorrelation function <f*f> and returns up to max_lag.

    Args:
        f (ndarray):
            Vector for autocorrelation

        max_lag (float):
            Lag at which to calculate up to

    Returns:
        ndarray: Autocorrelation vector

    """
    N = len(f)
    d = N - np.arange(N)
    # https://dsp.stackexchange.com/questions/741/why-should-i-zero-pad-a-signal-before-taking-the-fourier-transform
    f = np.pad(f, (0, N), 'constant', constant_values=(0, 0))
    fvi = np.zeros(2*N, dtype=type(f[0]))
    fwd = pyfftw.FFTW(f, fvi, flags=('FFTW_ESTIMATE',), threads=multiprocessing.cpu_count())
    fwd()
    inv_arg = np.conjugate(fvi)*fvi
    acf = np.zeros_like(inv_arg)
    rev = pyfftw.FFTW(inv_arg, acf, direction='FFTW_BACKWARD',
                      flags=('FFTW_ESTIMATE', ), threads=multiprocessing.cpu_count())
    rev()
    acf = acf[:N]/d
    return np.real(acf[:max_lag+1])


def corr(f, g, max_lag):
    """
    Computes fast correlation function <f*g> and returns up to max_lag. Assumes f and g are same length.

    Args:
        f (ndarray):
            Vector for correlation

        g (ndarray):
            Vector for correlation

        max_lag (float):
            Lag at which to calculate up to

    Returns:
        ndarray: Correlation vector

    """
    if not len(f) == len(g):
        raise ValueError('corr arguments must be the same length.')

    N = len(f)
    d = N - np.arange(N)
    f = np.pad(f, (0, N), 'constant', constant_values=(0, 0))
    g = np.pad(g, (0, N), 'constant', constant_values=(0, 0))

    fvi = np.zeros(2*N, dtype=type(f[0]))
    gvi = np.zeros(2*N, dtype=type(g[0]))

    fwd_f = pyfftw.FFTW(f, fvi, flags=('FFTW_ESTIMATE',), threads=multiprocessing.cpu_count())
    fwd_f()

    fwd_g = pyfftw.FFTW(g, gvi, flags=('FFTW_ESTIMATE',), threads=multiprocessing.cpu_count())
    fwd_g()

    inv_arg = np.conjugate(fvi)*gvi
    cf = np.zeros_like(inv_arg)
    rev = pyfftw.FFTW(inv_arg, cf, direction='FFTW_BACKWARD',
                      flags=('FFTW_ESTIMATE', ), threads=multiprocessing.cpu_count())
    rev()
    cf = cf[:N]/d
    return np.real(cf[:max_lag+1])
