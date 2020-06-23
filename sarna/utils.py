import warnings
from copy import deepcopy
from itertools import product

import numpy as np
from borsar.utils import get_info, find_index, find_range


# TODO:
# - [ ] should check type and size of the vars (check how mne-python does it)
# - [ ] detect if in notebook/qtconsole and ignore 'Out' and similar vars
# - [ ] display numpy arrays as "5 x 2 int array"
#       lists as "list of strings" etc.
def whos():
    """Print the local variables in the caller's frame.
    Copied from stack overflow:
    http://stackoverflow.com/questions/6618795/get-locals-from-calling-namespace-in-python"""
    import inspect
    frame = inspect.currentframe()
    ignore_vars = []
    ignore_starting = ['__']
    try:
        lcls = frame.f_back.f_locals
        # test if ipython
        ipy_vars = ['In', 'Out', 'get_ipython', '_oh', '_sh']
        in_ipython = all([var in lcls for var in ipy_vars])
        if in_ipython:
            ignore_vars = ipy_vars + ['_']
            ignore_starting += ['_i']
        for name, var in lcls.items():
            if name in ignore_vars:
                continue
            if any([name.startswith(s) for s in ignore_starting]):
                continue
            print(name, type(var), var)
    finally:
        del frame


def find_files(directory, ends=None):
    '''FIXME - add docs.'''
    files = os.listdir(directory)
    if ends is not None:
        files = [f for f in files if f.endswith(ends)]
    return files


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
# - [ ] more detailed docs
# - [ ] profile, compare to cythonized version?
# - [x] diff mode
# - [x] option to return slice
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
        if in_grp and not el:
            in_grp = False
            group_lims.append([start_ind, ii-1])
        elif not in_grp and el:
            in_grp = True
            start_ind = ii
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


# - [ ] I never actually used this, maybe just remove
def subselect_keys(key, mapping, sep='/'):
    '''
    Select keys with subselection by a separator.
    This code was shared by Dennis Engemann on github.

    Parameters
    ----------
    key : string | list of strings
        Keys to subselect with.
    mapping : dict
        Dictionary that is being selected.
    sep : string
        Separator to use in subselection.
    '''

    if isinstance(key, str):
        key = [key]

    mapping = deepcopy(mapping)

    if any(sep in k_i for k_i in mapping.keys()):
        if any(k_e not in mapping for k_e in key):
            # Select a given key if the requested set of
            # '/'-separated types are a subset of the types in that key

            for k in mapping.keys():
                if not all(set(k_i.split('/')).issubset(k.split('/'))
                           for k_i in key):
                    del mapping[k]

            if len(key) == 0:
                raise KeyError('Attempting selection of keys via '
                               'multiple/partial matching, but no '
                               'event matches all criteria.')
    else:
        raise ValueError('Your keys are bad formatted.')
    return mapping


# TODO: add evoked (for completeness)
# - [ ] check: mne now has _validate_type ...
def mne_types():
    import mne
    types = dict()
    from mne.io.meas_info import Info
    try:
        from mne.io import _BaseRaw
        from mne.epochs import _BaseEpochs
        types['raw'] = _BaseRaw
        types['epochs'] = _BaseEpochs
    except ImportError:
        from mne.io import BaseRaw
        from mne.epochs import BaseEpochs
        types['raw'] = BaseRaw
        types['epochs'] = BaseEpochs
    types['info'] = Info
    return types


# see if there is a standard library implementation of something similar
class AtribDict(dict):
    """Just like a dictionary, except that you can access keys with obj.key.

    Copied from psychopy.data.TrialType
    """
    def __getattribute__(self, name):
        try:  # to get attr from dict in normal way (passing self)
            return dict.__getattribute__(self, name)
        except AttributeError:
            try:
                return self[name]
            except KeyError:
                msg = "TrialType has no attribute (or key) \'%s\'"
                raise AttributeError(msg % name)


# TODO
# - [ ] move to borsar
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
        FIXME - here more datailed explanation
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
        df.loc[idx, value_name] = arr[adr] # this could be vectorized easily
        # add relevant values to dim columns
        for dim_idx, dim_adr in enumerate(adr):
            df.loc[idx, dim_names[dim_idx]] = groups[dim_idx][dim_adr]

    # column dtype inference
    try: # for pandas 0.22 or higher
        df = df.infer_objects()
    except: # otherwise
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = df.convert_objects(convert_numeric=True)
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
        the .mat file in fieldtrip represetnation.
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
