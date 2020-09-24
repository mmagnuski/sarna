# this module is currently way more messy than it should be
# but this should change pretty soon, hopfully

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
import mne
from warnings import warn
from sarna.utils import group, _invert_selection, _transfer_selection_to_raw
# from numba import jit


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


# - [?] figure out a good way of including rej
#       (currently - don't include)
# - [ ] switch to scipy.signal.welch ?
def segments_freq(eeg, win_len=2., win_step=0.5, n_fft=None,
                  n_overlap=None, picks=None, progress=True):
    from mne.io import _BaseRaw
    from mne.epochs import _BaseEpochs
    from mne.utils import _get_inst_data
    from mne.time_frequency import psd_welch

    sfreq = eeg.info['sfreq']
    t_min = eeg.times[0]
    time_length = len(eeg.times) / eeg.info['sfreq']
    n_win = int(np.floor((time_length - win_len) / win_step) + 1.)
    win_samples = int(np.floor(win_len * sfreq))

    # check and set n_fft and n_overlap
    if n_fft is None:
        n_fft = int(np.floor(sfreq))
    if n_overlap is None:
        n_overlap = int(np.floor(n_fft / 4.))
    if n_fft > win_samples:
        n_fft = win_samples
        n_overlap = 0
    if picks is None:
        picks = range(_get_inst_data(eeg).shape[-2])

    n_freqs = int(np.floor(n_fft / 2)) + 1
    if isinstance(eeg, _BaseRaw):
        n_channels, _ = _get_inst_data(eeg).shape
        psd = np.zeros((n_win, len(picks), n_freqs))
    elif isinstance(eeg, _BaseEpochs):
        n_epochs, n_channels, _ = _get_inst_data(eeg).shape
        psd = np.zeros((n_win, n_epochs, len(picks), n_freqs))
    else:
        raise TypeError('unsupported data type - has to be epochs or '
                        'raw, got {}.'.format(type(eeg)))

    # BTW: doing this with n_jobs=2 is about 100 times slower than with one job
    p_bar = progressbar.ProgressBar(max_value=n_win)
    for w in range(n_win):
        psd_temp, freqs = psd_welch(
            eeg, tmin=t_min + w * win_step,
            tmax=t_min + w * win_step + win_len,
            n_fft=n_fft, n_overlap=n_overlap, n_jobs=1,
            picks=picks, verbose=False, proj=True)
        psd[w, :] = psd_temp
        if progress:
            p_bar.update(w)
    return psd.swapaxes(0, 1), freqs

# - [ ] consider moving to utils
# - [x] warn if sfreq not given and some values are float
# - [x] treat floats as time and int as samples
# - [ ] maybe a smarter API or a class...
def window_steps(window_length, window_step, signal_len, sfreq=None):
    is_float = [isinstance(x, float) for x in \
                [window_length, window_step, signal_len]]
    any_float = any(is_float)
    if any_float and sfreq is None:
        raise TypeError('Some variables are float but sfreq was not given.')

    if any_float and sfreq is not None:
        if is_float[0]:
            window_length = int(np.round(window_length * sfreq))
        if is_float[1]:
            window_step = int(np.round(window_step * sfreq))
        if is_float[2]:
            signal_len = int(np.round(signal_len * sfreq))

    num_steps = int(np.floor((signal_len - window_length) / window_step)) + 1
    for w in range(num_steps):
        yield slice(w * window_step, window_length + w * window_step)


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

    # reshape data
    if not is_raw:
        X = X.reshape((n_epochs * n_channels, n_times))

    # specialized loop
    tfr = _cwt_loop(X, times_ind, Ws, W_sizes, w_time_lims)

    if is_raw:
        tfr = np.transpose(tfr, (1, 0, 2))
    else:
        tfr = np.transpose(tfr.reshape((n_freqs, n_epochs, n_channels,
            n_times_out)), (1, 2, 0, 3))
    return tfr


# @jit
def _cwt_loop(X, times_ind, Ws, W_sizes, w_time_lims):
    # allocate output
    n_freqs = len(Ws)
    tfr = np.empty([n_freqs] + [X.shape[0]] + [len(times_ind)],
                   dtype=np.complex128)
    tfr.fill(np.nan)

    # Loop across wavelets, compute power
    for ii, W in enumerate(Ws):
        l, r = map(int, np.floor(W_sizes[ii] / 2. * np.array([-1, 1])))
        t_start, t_end = w_time_lims[ii, :] + [0, 1]
        # loop across time windows
        for ti, tind in enumerate(times_ind[t_start:t_end]):
            tfr[ii, :, ti + t_start] = np.dot(X[:, tind + l:tind + r], W)
    return tfr


def _correct_overlap(periods):
    '''

    Parameters
    ----------
    periods : np.ndarray
        Numpy array of (n_periods, 3) shape. The columns are: epoch index,
        within-epoch sample index of period start, within-epoch sample index of
        period end.

    Returns
    -------
    periods : np.ndarray
        Corrected numpy array of (n_periods, 3) shape. The columns are: epoch
        index, within-epoch sample index of period start, within-epoch sample
        index of period end.

    '''
    n_rows = periods.shape[0]
    current_period = periods[0, :].copy()
    correct = list()

    for idx in range(1, n_rows):
        overlap = ((periods[idx, 0] == current_period[0])
                   and (periods[idx, 1] <= current_period[2]))

        if overlap:
            current_period[-1] = periods[idx, -1]
        else:
            correct.append(current_period)
            current_period = periods[idx, :].copy()

    correct.append(current_period)
    periods = np.stack(correct, axis=0)

    return periods


def _find_high_amplitude_periods(data, amp_z_thresh=2.5, min_period=0.1,
                                 extend=None):
    '''
    Find segments of high amplitude in filtered, hilbert-transformed signal.

    Parameters
    ----------
    data : mne.Epochs
        Epoched data. Must be filtered and hilbert-transformed.
    amp_z_thresh : float
        Z score threshold defining high amplitude periods. Defaults to ``2.5``.
    min_period : float
        Minimum length of high amplitude period in seconds.
        Defaults to ``0.1``.
    extend : float | None
        Extend each period by this many seconds on both sides (before and
        after). Defaults to ``None`` which does not extend the periods.

    Returns
    -------
    periods : np.ndarray
        Numpy array of (n_periods, 3) shape. The columns are: epoch index,
        within-epoch sample index of period start, within-epoch sample index of
        period end.
    '''
    from scipy.stats import zscore

    # amplitude periods
    n_epochs, n_channels, n_samples = data._data.shape
    comp_data = data._data.transpose([1, 0, 2]).reshape((n_channels, -1))

    # find segments with elevated amplitude
    comp_data_abs = np.abs(comp_data)
    envelope = np.nanmean(comp_data_abs, axis=0)
    envelope_z = zscore(envelope, nan_policy='omit')
    grp = group(envelope_z > amp_z_thresh)

    if len(grp) == 0:
        raise ValueError('No high amplitude periods were found.')
    # check if there are some segments that start at one epoch
    # and end in another
    # -> if so, they could be split, but we will ignore them for now
    epoch_idx = np.floor(grp / n_samples)
    epoch_diff = np.diff(epoch_idx, axis=0)
    epochs_joint = epoch_diff > 0
    if epochs_joint.any():
        msg = ('{:d} high-amplitude segments will be ignored because'
               ' the developer was lazy.')
        warn(msg.format(epochs_joint.sum()))
        epoch_diff = epoch_diff[~epochs_joint]

    segment_len = np.diff(grp, axis=1)
    good_length = segment_len[:, 0] * (1 / data.info['sfreq']) > min_period
    grp = grp[good_length, :]
    epoch_idx = np.floor(grp[:, [0]] / n_samples).astype('int')
    grp -= epoch_idx * n_samples

    if extend is not None:
        extend_samples = int(np.round(extend * data.info['sfreq']))
        extend_samples = np.array([-extend_samples, extend_samples])
        grp += extend_samples[np.newaxis, :]

        # check for limits
        msk1 = grp[:, 0] < 0
        msk2 = grp[:, 1] >= n_samples
        grp[msk1, 0] = 0
        grp[msk2, 1] = n_samples - 1

    periods = np.append(epoch_idx, grp, axis=1)
    periods = periods if extend is None else _correct_overlap(periods)
    return periods


def create_amplitude_annotations(raw, freq=None, events=None, event_id=None,
                                 picks=None, tmin=-0.2, tmax=0.5,
                                 amp_z_thresh=2., min_period=0.1,
                                 extend=None):
    '''

    Parameters
    ----------
    raw : mne.Raw
        Raw file to use.
    events: numpy array | None
        Mne events array of shape (n_events, 3). If None (default) `tmin` and
        `tmax` are not calculated with respect to events but the whole time
        range of the `raw` file.
    event_id: list | numpy array
        Event types (IDs) to use in defining segments for which psd is
        computed. If None (default) and events were passed all event types are
        used.
    freq : list | numpy array
        Frequency limits defining a range for which low amplitude periods will
        be calculated.
    picks : list
        List of channels for which low amplitude periods will be calculated.
    tmin : float
        Start time before event.
    tmax : float
        End time after event.
    amp_z_thresh : float
        Z score threshold defining high amplitude periods. Defaults to ``2.5``.
    min_period : float
        Minimum length of high amplitude period in seconds.
        Defaults to ``0.1``.
    extend : float | None
        Extend each period by this many seconds on both sides (before and
        after). Defaults to ``None`` which does not extend the periods.

    Returns
    -------
    raw_annot : mne.Raw
        Raw files with annotations.

    '''

    filt_raw = raw.copy().filter(freq[0], freq[1])

    if freq is None:
        raise TypeError('Frequencies have to be defined')
    if events is None:
        raise TypeError('Events have to be defined')
    if event_id is None:
        event_id = np.unique(events[:, 2]).tolist()

    epochs = mne.Epochs(filt_raw, events=events, event_id=event_id, tmin=tmin,
                        tmax=tmax, baseline=None, preload=True,
                        reject_by_annotation=False)

    if picks is None:
        filt_hilb_data = epochs.copy().apply_hilbert()
    else:
        filt_hilb_data = epochs.copy().pick(picks).apply_hilbert()

    hi_amp_epochs = _find_high_amplitude_periods(filt_hilb_data,
                                                 amp_z_thresh=amp_z_thresh,
                                                 min_period=min_period,
                                                 extend=extend)

    hi_amp_raw = _transfer_selection_to_raw(epochs, raw,
                                            selection=hi_amp_epochs)
    amp_inv_samples = _invert_selection(raw, selection=hi_amp_raw)

    sfreq = raw.info['sfreq']
    amp_inv_annot_sec = amp_inv_samples / sfreq

    n_segments = amp_inv_samples.shape[0]
    amp_annot = mne.Annotations(amp_inv_annot_sec[:, 0],
                                amp_inv_annot_sec[:, 1],
                                ['BAD_lowamp'] * n_segments)

    return amp_annot
