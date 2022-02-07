import numpy as np
import mne
from warnings import warn
from sarna.utils import group, _invert_selection, _transfer_selection_to_raw
# from numba import jit


def dB(x):
    return 10 * np.log10(x)


# - [ ] add detrending (1 + x + 1/x) or FOOOF cooperation
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


# - [ ] consider moving to utils
# - [x] warn if sfreq not given and some values are float
# - [x] treat floats as time and int as samples
# - [ ] maybe a smarter API or a class...
def window_steps(window_length, window_step, signal_len, sfreq=None):
    is_float = [isinstance(x, float)
                for x in [window_length, window_step, signal_len]]
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
                      normalize=False, cmap='inferno', show=False)

    # highlight channels
    circles = ax[0].findobj(plt.Circle)
    for ch in channels:
        circles[ch].set_color('r')
        circles[ch].set_radius(0.025)

    plot_freq = (freqs > 1.) & (freqs < 15.)
    ax[1].plot(freqs[plot_freq], mean_psd[channels, :][:, plot_freq].T)
    chan_avg = mean_psd[channels, :].mean(axis=0)
    ax[1].plot(freqs[plot_freq], chan_avg[plot_freq], color='k', lw=2)
    return fig


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


# FIXME: check out of bounds indices
# FIXME: check if the stop indices are python-slice compliant (inclusive)
#        or python-indexing compliant
def _find_sel_amplitude_periods(epochs, threshold=2.5, min_period=0.1,
                                periods='high', extend=None):
    '''
    Find segments of high or low amplitude in filtered, hilbert-transformed
    signal.

    The channels are averaged, so make sure you pick channels before passing
    the data to this function if you don't want to use all.

    Parameters
    ----------
    epochs : mne.Epochs
        Epoched data. Should be filtered and hilbert-transformed.
    threshold : float, str
        Threshold defining high amplitude periods to select: if float, it
        is interpreted as a z value threshold; if str, as percentage of
        fragments with in the highest amplitude in form of ``'xx%'``
        (for example with ``'25%'`` 25% of singal with highest amplitude will
        be selected ). Defaults to ``2.5``.
    min_period : float
        Minimum length of high amplitude period in seconds.
        Defaults to ``0.1``.
    periods : str
        Sepcification of perionds to find. Might be 'high' or 'low' amplitude.
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
    n_epochs, n_channels, n_samples = epochs._data.shape
    comp_data = epochs._data.transpose([1, 0, 2]).reshape((n_channels, -1))

    # find segments with elevated amplitude
    envelope = np.nanmean(comp_data, axis=0)

    if isinstance(threshold, str) and '%' in threshold:
        perc = 100 - float(threshold.replace('%', ''))
        threshold = np.nanpercentile(envelope, perc)
    else:
        envelope = zscore(envelope, nan_policy='omit')

    if periods == 'high':
        grp = group(envelope > threshold)
    elif periods == 'low':
        grp = group(envelope < threshold)
    else:
        raise ValueError('Unrecognised `periods` option "{}".'.format(periods))    

    if len(grp) == 0:
        raise ValueError('No {} amplitude periods were found.'.format(periods))
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
    good_length = segment_len[:, 0] * (1 / epochs.info['sfreq']) > min_period
    grp = grp[good_length, :]
    epoch_idx = np.floor(grp[:, [0]] / n_samples).astype('int')
    grp -= epoch_idx * n_samples

    if extend is not None:
        extend_samples = int(np.round(extend * epochs.info['sfreq']))
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
                                 threshold=2., min_period=0.1, periods='high',
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
    threshold : float, str
        Threshold defining high amplitude periods to select: if float, it
        is interpreted as a z value threshold; if str, as percentage of
        fragments with in the highest amplitude in form of ``'xx%'``
        (for example with ``'25%'`` 25% of singal with highest amplitude will
        be selected ). Defaults to ``2.5``.
    min_period : float
        Minimum length of high amplitude period in seconds.
        Defaults to ``0.1``.
    periods : str
        Sepcification of perionds to find. Might be 'high' or 'low' amplitude.
    extend : float | None
        Extend each period by this many seconds on both sides (before and
        after). Defaults to ``None`` which does not extend the periods.

    Returns
    -------
    raw_annot : mne.Raw
        Raw files with annotations.
    '''

    if freq is None:
        raise TypeError('Frequencies have to be defined')
    if events is None:
        raise TypeError('Events have to be defined')
    if event_id is None:
        event_id = np.unique(events[:, 2]).tolist()

    filt_raw = raw.copy().filter(freq[0], freq[1])

    filt_raw_nan = filt_raw.copy()
    filt_raw_nan._data = raw.get_data(reject_by_annotation='NaN')

    epochs_nan = mne.Epochs(filt_raw_nan, events=events, event_id=event_id,
                            tmin=tmin, tmax=tmax, baseline=None, preload=True,
                            reject_by_annotation=False)
    epochs = mne.Epochs(filt_raw, events=events, event_id=event_id, tmin=tmin,
                        tmax=tmax, baseline=None, preload=True,
                        reject_by_annotation=False)

    del filt_raw_nan

    filt_hilb_data = np.abs(epochs.copy().pick(picks).apply_hilbert()._data)
    filt_hilb_data[np.isnan(epochs_nan.copy().pick(picks)._data)] = np.nan
    epochs_nan._data = filt_hilb_data

    hi_amp_epochs = _find_sel_amplitude_periods(epochs_nan,
                                                 threshold=threshold,
                                                 min_period=min_period,
                                                 extend=extend)

    hi_amp_raw = _transfer_selection_to_raw(epochs, raw,
                                            selection=hi_amp_epochs)
    amp_inv_samples = _invert_selection(raw, selection=hi_amp_raw)

    sfreq = raw.info['sfreq']
    amp_inv_annot_sec = amp_inv_samples / sfreq

    n_segments = amp_inv_samples.shape[0]
    annot_name = 'BAD_{}amp'.format(periods)
    amp_annot = mne.Annotations(amp_inv_annot_sec[:, 0],
                                amp_inv_annot_sec[:, 1],
                                annot_name * n_segments)
    return amp_annot


def grand_average_psd(psd_list):
    '''Perform grand average on a list of PSD objects.

    Parameters
    ----------
    psd_list : list
        List of ``borsar.freq.PSD`` objects.

    Returns
    -------
    grand_psd : borsar.freq.PSD
        Grand averaged spectrum.
    '''
    assert isinstance(psd_list, list)

    # make sure that all psds have the same number and order of channels
    # and the same frequencies
    freq1 = psd_list[0].freqs
    ch_names1 = psd_list[0].ch_names
    n_channels = len(ch_names1)
    for psd in psd_list:
        assert len(psd.freqs) == len(freq1)
        assert (psd.freqs == freq1).all()
        assert len(psd.ch_names) == n_channels
        assert all([ch_names1[idx] == psd.ch_names[idx]
                    for idx in range(n_channels)])

    all_psds = list()
    for this_psd in psd_list:
        # upewniamy się, że epoki są uśrednione
        this_psd = this_psd.copy().average()
        all_psds.append(this_psd.data)

    # łączymy widma w macierz (osoby x kanały x częstotliwości)
    all_psds = np.stack(all_psds, axis=0)

    # uśredniamy wymiar osób, zostają nam kanały x częstotliwości
    all_psds = all_psds.mean(axis=0)

    # kopiujemy wybrane psd z listy
    grand_psd = this_psd.copy()
    # i wypełniamy wartościamy średniej po osobach
    grand_psd._data = all_psds

    return grand_psd
