# test
# ----
# from mne import time_frequency.tfr as tfr
#
# # import any dataset
#
# freqs = np.arange(1., 25.1, 0.5)
# W = trf.morlet(sfreq, freqs, n_cycles=freqs)
# TFR1 = tfr._cwt(signal, W)
# ...
import numpy as np


def dB(x):
    return 10 * np.log10(x)


# - [ ] add detrending (1 + x + 1/x)
def transform_spectrum(spectrum, dB=False, normalize=False, detrend=False):
    """Common spectrum transformations.

    Parameters
    ----------
    spectrum : numpy array
        channels x frequencies
    """

    if dB:
        spectrum = 10 * np.log10(spectrum)
    if normalize:
        if dB:
            # move whole spectrum up, so that normalization
            # does not return weird results
            min_val = spectrum.min() - 0.01
            spectrum -= min_val
        spectrum /= spectrum.sum(axis=1)[:, np.newaxis]
    return spectrum


def plot_topo_and_psd(inst, mean_psd, freqs, channels):
    from matplotlib import gridspec
    import matplotlib.pyplot as plt
    from mne.viz.topomap import plot_psds_topomap

    fig = plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
    ax = [plt.subplot(g) for g in gs]

    plot_psds_topomap(psds=mean_psd, freqs=freqs, pos=inst.info,
                      dB=False, axes=[ax[0]], bands=[(4., 8., 'Theta')],
                      normalize=False, cmap='inferno', show=False);

    # highlight channels
    circles = ax[0].findobj(plt.Circle)
    for ch in channels:
        circles[ch].set_color('r')
        circles[ch].set_radius(0.025)

    plot_freq = (freqs > 1.) & (freqs < 15.)
    ax[1].plot(freqs[plot_freq], mean_psd[channels, :][:, plot_freq].T);
    chan_avg = mean_psd[channels, :].mean(axis=0)
    ax[1].plot(freqs[plot_freq], chan_avg[plot_freq], color='k', lw=2)


# - [ ] average param
# - [ ] use_fft?
def _my_cwt(inst, Ws, times=None, picks=None, fast_dot=True):
    """Compute cwt with dot products at specific times.

    Parameters
    ----------
    inst - Epochs | Raw
        Object containing data to process.
    Ws - list of numpy arrays
        List of wavelets to use in time-frequency decomposition.
    times - array-like
        Time points in seconds defining centers of consecutive time windows.
    picks - channels to use in time-frequency decomposition.

    Returns
    -------
    tfr : numpy array
        single-trial time-frequency decomposition of `epochs`. Numpy array of
        size (n_epochs, n_channels, n_wavelets, n_times_of_interest)
    """
    from mne.io import _BaseRaw
    from mne.utils import _get_fast_dot

    is_raw = isinstance(inst, _BaseRaw)
    X = inst._data if is_raw else inst.get_data()
    if is_raw:
        n_channels, n_times = X.shape
    else:
        n_epochs, n_channels, n_times = X.shape

    # check wavelets
    W_sizes = [W.size for W in Ws]

    if picks is None:
        picks = list(range(n_channels))
    if times is None:
        # some auto settings...
        pass

    times_ind = inst.time_as_index(times)
    n_times_out = len(times)
    n_freqs = len(Ws)

    # each wavelet has its own timesteps (in case things don't fit)
    w_time_lims = list()
    for ws in W_sizes:
        hlf = ws / 2
        l, h = map(int, (np.ceil(hlf), np.floor(hlf)))
        good_times = (times_ind >= l) & (times_ind <= (n_times - h))
        w_time_lims.append(np.where(good_times)[0][[0, -1]])
    w_time_lims = np.vstack(w_time_lims)

    # reshape data, prepare output
    if not is_raw:
        X = X.reshape((n_epochs * n_channels, n_times))
        tfr = np.empty((n_freqs, n_epochs * n_channels, n_times_out),
                       dtype=np.complex128)
    else:
        tfr = np.empty((n_freqs, n_channels, n_times_out),
                       dtype=np.complex128)
    tfr.fill(np.nan)

    # get dot operator:
    if fast_dot:
        dot = _get_fast_dot()
    else:
        dot = np.dot

    # Loop across wavelets, compute power
    for ii, W in enumerate(Ws):
        l, r = map(int, np.floor(W_sizes[ii] / 2. * np.array([-1, 1])))   
        t_start, t_end = w_time_lims[ii, :] + [0, 1]
        # loop across time windows
        for ti, tind in enumerate(times_ind[t_start:t_end]):
            tfr[ii, :, ti + t_start] = dot(X[:, tind + l:tind + r], W)

    if is_raw:
        tfr = np.transpose(tfr, (1, 0, 2))
    else:
        tfr = np.transpose(tfr.reshape((n_freqs, n_epochs, n_channels,
            n_times_out)), (1, 2, 0, 3))
    return tfr
