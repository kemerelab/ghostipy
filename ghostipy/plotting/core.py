import numpy as np
import xarray as xr
import hvplot.xarray
import matplotlib.pyplot as plt

__all__ = ['plot_fourier_spectrogram',
           'plot_wavelet_spectrogram']

def plot_fourier_spectrogram(s, f, t):

    xa = xr.DataArray(s,
                      dims=['frequency', 'time'],
                      coords={'frequency':f, 'time':t})

    plot = xa.hvplot(xa)
    return plot



def plot_wavelet_spectrogram(s, f, t, *,
                             kind='amplitude',
                             **kwargs):

    if kind == 'amplitude':
        s = np.sqrt(s.real**2 + s.imag**2)
        name = 'amplitude'
    elif kind == 'power':
        s = s.real**2 + s.imag**2
        name = 'power'
    else:
        raise ValueError(f"Invalid 'kind' {f} specified")

    xa = xr.DataArray(s,
                      dims=['frequency', 'time'],
                      coords={'frequency':f, 'time':t}, 
                      name=name)

    plot = xa.hvplot.quadmesh(**kwargs)

    return plot


def plot_frequency_response(backend):
    b = firdesign(3001, [145, 150, 250, 260], [0, 1, 1, 0], fs=1000)
    w, h = sig.freqz(b, fs=1000)

    if backend == 'matplotlib':
        hv.extension('matplotlib')
        fig, ax = plt.subplots(1, 3)
        ax[0].plot(np.arange(len(b)), b)
        ax[1].plot(w, 20*np.log10(np.abs(h)))
        ax[2].plot(w, np.angle(h))
        plt.tight_layout()
        return ax

    else:
        hv.extension('bokeh')
        return (hv.Curve((np.arange(len(b)), b)) + hv.Curve((w, 20*np.log10(np.abs(h))), kdims=['frequency', 'response'])).cols(1)