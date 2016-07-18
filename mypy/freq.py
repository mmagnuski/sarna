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


def _my_cwt(epochs, Ws, times=None, picks=None):
    """Compute cwt with dot products at specific times.

    Parameters
    ----------

    Returns
    -------
    tfr : numpy array
        single-trial time-frequency decomposition of `epochs`. Numpy array of
        size (n_epochs, n_channels, n_wavelets, n_times_of_interest)
    """
    X = epochs.get_data()
    n_epochs, n_channels, n_times = X.shape

    # check wavelets
    W_sizes = [W.size for W in Ws]
    # Ws_max_size = max(W_sizes)

    if picks is None:
        picks = list(range(n_channels))
    if times is None:
        # some auto settings...
        pass

    times_ind = find_index(epochs.times, times)
    n_times_out = len(times)
    n_freqs = len(Ws)

    # ADD check which time indices can be used for which wavelets...

    # reshape data, prepare output
    data = data.reshape((n_epochs * n_channels, n_times))
    tfr = np.zeros((n_freqs, n_epochs * n_channels, n_times_out),
                   dtype=np.complex128)

    # Loop across wavelets
    for ii, W in enumerate(Ws):
        l, r = W_sizes[ii] / 2. * np.array([-1, 1])
        l, r = np.floor(l), np.floor(r)
        # loop across time windows
        for ti, tind in enumerate(times_ind):
            tfr[ii, :, t] = np.dot(data[:, tind - l:tind + r], W)
    return np.transpose(tfr.reshape((n_freqs, n_epochs, n_channels, n_times)),
                        (1, 2, 0, 3))
