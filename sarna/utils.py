import os
import warnings
from copy import deepcopy
from itertools import product

import numpy as np
from borsar.utils import get_info, find_index, find_range


def extend_slice(slc, val, maxval, minval=0):
    '''Extend slice `slc` by `val` in both directions but not exceeding
    `minval` or `maxval`.

    Parameters
    ----------
    slc : slice
        Slice to extend.
    val : int or float
        Value by which to extend the slice.
    maxval : int or float
        Maximum value that cannot be exceeded.

    Returns
    -------
    slc : slice
        New, extended slice.
    '''
    start, stop, step = slc.start, slc.stop, slc.step
    # start
    if not start == minval:
        start -= val
        if start < minval:
            start = minval
    # stop
    if not stop == maxval:
        stop += val
        if stop > maxval:
            stop = maxval
    return slice(start, stop, step)


# TODO:
# - [ ] ! more detailed docs
# - [ ] ! add tests !
# - [ ] profile, maybe check numba version
def group(vec, diff=False, return_slice=False):
    '''
    Group values in a vector into ranges of adjacent identical values.
    '''
    in_grp = False
    group_lims = list()
    if diff:
        vec = np.append(vec, np.max(vec) + 1)
        vec = np.diff(vec) > 0
    else:
        vec = np.append(vec, False)

    # group
    for ii, el in enumerate(vec):
        if not in_grp and el:
            in_grp = True
            start_ind = ii
        elif in_grp and not el:
            in_grp = False
            group_lims.append([start_ind, ii-1])
    grp = np.array(group_lims)

    # format output
    if diff:
        grp[:, 1] += 1
    if return_slice:
        slc = list()
        for start, stop in grp:
            slc.append(slice(start, stop + 1))
        return slc
    else:
        return grp


def mne_types():
    from mne import Evoked
    from mne.io.meas_info import Info
    from mne.io import BaseRaw
    from mne.epochs import BaseEpochs

    types = dict(raw=BaseRaw, epochs=BaseEpochs, info=Info, evoked=Evoked)
    return types


# TODO
# CHECK and COMPARE: it might actually be better to use Xarray than this
# - [ ] more input validation
#       validate dim_names, dim_values
# - [x] groups could be any of following
#   * dict of int -> (dict of int -> str)
#   * instead of int -> str there could be tuple -> str
#   * or str -> list mapping
# - [x] support list of lists for groups as well
def array2df(arr, dim_names=None, groups=None, value_name='value'):
    '''
    Melt array into a pandas DataFrame.

    The resulting DataFrame has one row per array value and additionally
    one column per array dimension.

    Parameters
    ----------
    arr : numpy array
        Array to be transformed to DataFrame.
    dim_names : list of str or dict of int to str mappings
        Names of consecutive array dimensions - used as column names of the
        resulting DataFrame.
    groups : list of dicts or dict of dicts
        FIXME - here more detailed explanation
    value_name : ...
        ...

    Returns
    -------
    df : pandas DataFrame
        ...

    Examples
    --------
    >> arr = np.arange(4).reshape((2, 2))
    >> array2df(arr)

      value dim_i dim_j
    0     0    i0    j0
    1     1    i0    j1
    2     2    i1    j0
    3     3    i1    j1

    >> arr = np.arange(12).reshape((4, 3))
    >> array2df(arr, dim_names=['first', 'second'], value_name='array_value',
    >>          groups=[{'A': [0, 2], 'B': [1, 3]},
    >>                  {(0, 2): 'abc', (1,): 'd'}])

       array_value first second
    0            0     A    abc
    1            1     A      d
    2            2     A    abc
    3            3     B    abc
    4            4     B      d
    5            5     B    abc
    6            6     A    abc
    7            7     A      d
    8            8     A    abc
    9            9     B    abc
    10          10     B      d
    11          11     B    abc
    '''
    import pandas as pd
    n_dim = arr.ndim
    shape = arr.shape

    dim_letters = list('ijklmnop')[:n_dim]
    if dim_names is None:
        dim_names = {dim: 'dim_{}'.format(l)
                     for dim, l in enumerate(dim_letters)}
    if groups is None:
        groups = {dim: {i: dim_letters[dim] + str(i)
                        for i in range(shape[dim])} for dim in range(n_dim)}
    else:
        if isinstance(groups, dict):
            groups = {dim: _check_dict(groups[dim], shape[dim])
                      for dim in groups.keys()}
        elif isinstance(groups, list):
            groups = [_check_dict(groups[dim], shape[dim])
                      for dim in range(len(groups))]

    # initialize DataFrame
    col_names = [value_name] + [dim_names[i] for i in range(n_dim)]
    df = pd.DataFrame(columns=col_names, index=np.arange(arr.size))

    # iterate through dimensions producing tuples of relevant dims...
    for idx, adr in enumerate(product(*map(range, shape))):
        df.loc[idx, value_name] = arr[adr]  # this could be vectorized easily
        # add relevant values to dim columns
        for dim_idx, dim_adr in enumerate(adr):
            df.loc[idx, dim_names[dim_idx]] = groups[dim_idx][dim_adr]

    # column dtype inference
    df = df.infer_objects()
    return df


# utility function used by array2df (what does it do?)
# FIXME - better docs, comments, assert fail message
def _check_dict(dct, dim_len):
    if isinstance(dct, dict):
        str_keys = all(isinstance(k, str) for k in dct.keys())
        if not str_keys:
            tuple_keys = all(isinstance(k, tuple) for k in dct.keys())

        if str_keys:
            vals_set = set()
            new_dct = dict()
            for k in dct.keys():
                for val in dct[k]:
                    new_dct[val] = k
                    vals_set.add(val)
            assert len(vals_set) == dim_len
        elif tuple_keys:
            new_dct = dict()
            i_set = set()
            for k, val in dct.items():
                for i in k:
                    new_dct[i] = val
                    i_set.add(i)
            assert len(i_set) == dim_len
        else:
            new_dct = dct
    else:
        # validate if equal to num dims
        assert len(dct) == dim_len
        new_dct = dct
    return new_dct


# - [ ] add round-trip test
# - [ ] later consider support for reading partial epochs into mne
#       partial='ignore' or partial='nan' (or maybe a list of epochs then
#       but that wouldn't be that useful)
def epochs_to_ft(epochs, fname, var_name='data', trialinfo=None):
    '''Save epochs to a .mat file in fieldtrip trials representation.

    Parameters
    ----------
    epochs : mne.Epochs
        Instance of mne epochs object. The epochs data to save to
        the .mat file in fieldtrip representation.
    fname : str
        Name (or full path) of the file to save data to.
    var_name : str, optional
        Variable info to put all the information in, 'data' by default.
        If False or None - all the components (trial, time, label etc.) will
        be saved directly in the .mat file. This makes a slight difference in
        reading in matlab where ``var_name=False`` could be seen as more
        convenient. With ``var_name='data'``:
        > temp = load('saved_file.mat');
        > data = temp.data;
        With ``var_name=False``:
        > data = load('saved_file.mat')
    trialinfo : 2d numpy array, optional
        n_trials x n_features numpy array where features are information the
        user wants to keep track of for respective trials (for example -
        experimental conditions and/or behavioral performance). By default
        trialinfo is None, which leads to saving just the trial numbers in
        data.trialinfo.
    '''
    import mne
    from scipy.io import savemat
    from borsar.channels import get_ch_pos

    # safety checks
    if not isinstance(epochs, mne.Epochs):
        raise TypeError('epochs must be an instance of mne.Epochs,'
                        'got %s.' % type(epochs))

    if not isinstance(fname, str):
        raise TypeError('fname must be a str, got %s.' % type(fname))

    # get basic information from the epochs file
    sfreq = epochs.info['sfreq']
    n_trials, n_channels, n_samples = epochs._data.shape
    ch_names = np.array(epochs.ch_names, dtype='object'
                        ).reshape((n_channels, 1))

    if trialinfo is not None:
        assert isinstance(trialinfo, np.ndarray), ('trialinfo must be a numpy'
                                                   ' array.')
        assert trialinfo.ndim == 2, 'trialinfo must be 2d.'
        if not trialinfo.shape[0] == n_trials:
            msg = ('trialinfo must have n_trials rows, n_trials is {:d} while'
                   ' trialinfo number of rows is {:d}.').format(
                   n_trials, trialinfo.shape[0])
            raise ValueError(msg)
    else:
        # get event_id as the first column and epoch index as the second
        epoch_idx = np.arange(1, n_trials + 1, dtype='float')
        trialinfo = np.stack([epochs.events[:, -1], epoch_idx], axis=1)

    # get channel position, multiply by 100 because fieldtrip wants
    # units in cm, not meters
    pos = get_ch_pos(epochs) * 100
    n_samples_pre = (epochs.times < 0.).sum()

    # double brackets to have n_channels x 1 array
    chantype = np.array([['eeg']] * n_channels, dtype='object')
    chanunit = np.array([['V']] * n_channels, dtype='object')

    # reconstruct epoch sample limits from epochs.events
    sample_limits = np.round(epochs.times[[0, -1]] * sfreq).astype('int')
    sampleinfo = np.stack([epochs.events[:, 0] + sample_limits[0] + 1,
                           epochs.events[:, 0] + sample_limits[1] + 1],
                          axis=1)

    # construct cell array of times and trial data
    time = np.empty((1, n_trials), dtype='object')
    trial = np.empty((1, n_trials), dtype='object')
    for idx in range(n_trials):
        time[0, idx] = epochs.times
        trial[0, idx] = epochs._data[idx]

    data = dict()
    # header could have elec, but it does not seem to be necessary
    data['hdr'] = dict(label=ch_names, nChans=n_channels, Fs=sfreq,
                       nSamples=n_samples, nSamplesPre=n_samples_pre,
                       nTrials=n_trials, chantype=chantype, chanunit=chanunit)
    data['label'] = ch_names
    data['time'] = time
    data['trial'] = trial
    data['fsample'] = sfreq
    # sampleinfo and trialinfo are used in matlab in the default double format
    data['sampleinfo'] = sampleinfo.astype('float')
    data['trialinfo'] = trialinfo.astype('float')
    data['elec'] = dict(chanpos=pos, chantype=chantype, chanunit=chanunit,
                        elecpos=pos, label=ch_names, unit='cm')
    data['elec']['type'] = 'egi64'

    # pack into a variable if var_name is defined
    if var_name:
        data = {var_name: data}

    savemat(fname, data)


# - [ ] move to borsar, try using autonotebook if not str
# - [ ] later allow for tqdm progressbar as first arg
def progressbar(progressbar, total=None):
    # progressbar=True should give text progressbar
    if isinstance(progressbar, bool) and progressbar:
        progressbar = 'text'

    if progressbar == 'notebook':
        from tqdm import tqdm_notebook
        pbar = tqdm_notebook(total=total)
    elif progressbar == 'text':
        from tqdm import tqdm
        pbar = tqdm(total=total)
    else:
        pbar = EmptyProgressbar(total=total)
    return pbar


class EmptyProgressbar(object):
    def __init__(self, total=None):
        self.total = total

    def update(self, val):
        pass


def _transfer_selection_to_raw(epochs, raw, selection):
    '''
    Translate epoch-level selections back to raw signal.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs file to use.
    raw: mne.Raw
        Raw file to use.
    selection : numpy.ndarray
        High amplitude periods in samples for epochs. Numpy array of
        (n_periods, 3) shape. The columns are: epoch index, within-epoch
        sample index of period start, within-epoch sample index of period end.

    Returns
    -------
    hi_amp_raw : numpy.ndarrays
        High amplitude periods in samples for raw file. Numpy array of
        (n_periods, 2) shape. The columns are: sample index of period start,
        sample index of period end.
    '''
    selection_raw = np.zeros((selection.shape[0], 2))
    sfreq = raw.info['sfreq']
    epoch_events = epochs.events[:, 0].copy()
    tmin = epochs.tmin
    tmin_samples = int(np.round(tmin * sfreq))

    for idx in range(selection.shape[0]):
        epoch_idx, start, end = selection[idx, :]
        selection_raw[idx, 0] = (start + epoch_events[epoch_idx] +
                                 tmin_samples)
        selection_raw[idx, 1] = (end + epoch_events[epoch_idx] +
                                 tmin_samples)

    return selection_raw


def _invert_selection(raw, selection):
    '''
    Parameters
    ----------
    raw : mne.Raw
        Raw file to use.
    hi_amp_raw : numpy.ndarray
        High amplitude periods in samples for raw file. Numpy array of
        (n_periods, 2) shape. The columns are: sample index of period start,
        sample index of period end.
    Returns
    -------
    amp_inv_samples : numpy.ndarray
        Inverted periods in samples. Numpy array of (n_periods, 2) shape. The
        columns are: sample index of period start, sample index of period end.
    '''
    amp_inv_samples = np.zeros((selection.shape[0] + 1, 2))
    start, _ = selection[0, :]
    amp_inv_samples[0, :] = [0, start]

    for idx in range(selection.shape[0] - 1):
        _, end = selection[idx, :]
        start, _ = selection[idx + 1, :]
        amp_inv_samples[idx + 1, 0] = end
        amp_inv_samples[idx + 1, 1] = start - amp_inv_samples[idx + 1, 0]

    n_samples = raw._data.shape[1]
    _, end = selection[-1, :]
    raw_start_samples = end
    amp_inv_samples[-1, :] = [raw_start_samples, n_samples - raw_start_samples]

    return amp_inv_samples


def fix_channel_pos(inst, project_to_radius=0.095):
    '''Scale channel positions to default mne head radius.
    FIXME - add docs'''
    import borsar
    from mne.bem import _fit_sphere

    # get channel positions matrix
    pos = borsar.channels.get_ch_pos(inst)

    # ignore channels without positions
    no_pos = np.isnan(pos).any(axis=1) | (pos == 0).any(axis=1)
    pos = pos[~no_pos, :]

    # fit sphere to channel positions
    radius, origin = _fit_sphere(pos)
    scale = radius / project_to_radius

    info = get_info(inst)
    for idx, chs in enumerate(info['chs']):
        if chs['kind'] == 2:
            chs['loc'][:3] -= origin
            chs['loc'][:3] /= scale

    return inst


# - [ ] CHECK: this seems to already be present in mne with sphere='eeglab'
def create_eeglab_sphere(inst):
    '''Create sphere settings (x, y, z, radius) that produce eeglab-like
    topomap projection. The projection places Oz channel at the head outline
    because it is at the level of head circumference in the 10-20 system.

    Parameters
    ----------
    inst : mne object instance
        Mne object that contains info dictionary.

    Returns
    -------
    (x, y, z, radius)
        First three values are x, y and z coordinates of the sphere center.
        The last value is the sphere radius.
    '''
    check_ch = ['oz', 'fpz', 't7', 't8']
    ch_names_lower = [ch.lower() for ch in inst.ch_names]
    ch_idx = [ch_names_lower.index(ch) for ch in check_ch]
    pos = np.stack([inst.info['chs'][idx]['loc'][:3] for idx in ch_idx])

    # first we obtain the x, y, z of the sphere center:
    x = pos[0, 0]
    y = pos[-1, 1]
    z = pos[:, -1].mean()

    # now we calculate the radius from T7 and T8 x position
    # but correcting for sphere center
    pos_corrected = pos - np.array([[x, y, z]])
    radius1 = np.abs(pos_corrected[[2, 3], 0]).mean()
    radius2 = np.abs(pos_corrected[[0, 1], 1]).mean()
    radius = np.mean([radius1, radius2])

    return (x, y, z, radius)
