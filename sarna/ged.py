import numpy as np
import scipy

import mne
from borsar.viz import Topo


# TODO: add plot_eig for plotting eigenvalues
# TODO: split ged computation and object construction
# TODO: we use eig.real - maybe a warning should be added if we get complex
#       values?
class GED(object):
    def __init__(self, cov_S=None, cov_R=None, reg=None, eig=None,
                 filters=None, patterns=None, description=None):
        '''
        Compute Generalized Eigendecomposition (GED) of two covariance
        matrices.

        Parameters
        ----------
        cov_S : mne Covariance | array
            Covariance matrix for the signal of interest.
        cov_R : mne Covariance | array
            Covariance matrix for the reference signal.
        reg : None | float
            Regularization factor, from 0 - 1. If ``None`` then regularization
            is not performed.

        Notes
        -----
        The ``.filters`` attribute contains n_channels x n_filters array.
        The ``.patterns`` attribute contains n_patterns x n_channels array.
        '''
        # CONSIDER - could also be first constructed and the ``.fit``
        #            to be complementary with sklearn...
        if filters is None and eig is None:
            # compute the GED from covariance matrices
            if not isinstance(cov_R, np.ndarray):
                cov_R = cov_R.data
            if not isinstance(cov_S, np.ndarray):
                cov_S = cov_S.data

            n_channels = cov_R.shape[0]

            # regularization
            if reg is not None:
                reg_eig = reg * np.abs(np.linalg.eig(cov_R)[0].mean())
                cov_R = (1 - reg) * cov_R + reg_eig * np.eye(n_channels)

            # compute GED and sort by eigenvalue
            eig, filters = scipy.linalg.eig(cov_S, cov_R)
            eig = eig.real
            srt = np.argsort(eig)[::-1]
            eig, filters = eig[srt], filters[:, srt]

        # compose attributes
        self.eig = eig
        self.filters = filters
        self.patterns = (patterns if patterns is not None
                         else _get_patterns(self.filters, cov_S))
        self.description = description

    def plot(self, info, idx=None, axes=None, **args):
        '''
        Plot GED component topographical patterns.

        Parameters
        ----------
        info: mne Info instance
            mne-python Info object - used for topomap plotting.
        idx : int or array-like of int
            Index or indices for components to plot.
        axes : matplotlib.Axes
            Axes to plot the topomaps in.
        **args : dict
            Additional arguments are passed to ``borsar.viz.Topo``.

        Returns
        -------
        topo : borsar.viz.Topo
            Topography object. Allows to fine-tune topography presentation.
        '''
        if idx is None:
            idx = np.arange(6)
        idx = _deal_with_idx(idx)
        ptr = self.patterns[idx].T
        if ptr.shape[0] == 1:
            ptr = ptr[0]

        return Topo(ptr, info, axes=axes, **args)

    def apply(self, inst, comp_idx):
        comp_idx = _deal_with_idx(comp_idx)
        inst_copy = inst.copy()

        # extract component time-courses
        if isinstance(inst, mne.io.BaseRaw):
            comp_data = self.filters[:, comp_idx].T @ inst._data
        elif isinstance(inst, mne.BaseEpochs):
            comp_data = [self.filters[:, comp_idx].T @ inst._data[idx]
                         for idx in range(inst._data.shape[0])]
            comp_data = np.stack(comp_data, axis=0)

        # return Raw / Epochs object
        ch_renames = {ch_name: 'comp_{:02d}'.format(ch_idx + 1)
                      for ch_idx, ch_name in enumerate(inst_copy.ch_names)}
        inst_copy.rename_channels(ch_renames)
        ch_picks = ['comp_{:02d}'.format(idx + 1) for idx in comp_idx]
        inst_copy.pick_channels(ch_picks)
        inst_copy._data = comp_data

        return inst_copy

    def save(self, fname, overwrite=False):
        '''Save to fitted GED object to hdf5 file.

        Parameters
        ----------
        fname : str
            File name or full path to the file.
        overwrite : bool
            Whether to overwrite the file if it exists.
        '''

        from mne.externals import h5io

        data_dict = {'eig': self.eig, 'filters': self.filters,
                     'patterns': self.patterns,
                     'description': self.description}
        h5io.write_hdf5(fname, data_dict, overwrite=overwrite)


def read_ged(fname):
    '''Read GED object from hdf5 file.

    Parameters
    ----------
    fname : str
        File name or full file path.

    Returns
    -------
    ged : sarna.ged.GED
        Read GED object.
    '''
    from mne.externals import h5io

    data_dict = h5io.read_hdf5(fname)
    ged = GED(
        eig=data_dict['eig'], filters=data_dict['filters'],
        patterns=data_dict['patterns'],
        description=data_dict['description'])
    return ged


def _get_patterns(vec, cov):
    '''Turn filters to patterns.'''
    n_signals = vec.shape[0]
    patterns = np.zeros((n_signals, n_signals))

    for idx in range(n_signals):
        patterns[idx, :] = vec[:, [idx]].T @ cov.data

    return patterns


def _deal_with_idx(idx):
    '''Helper function to deal with various ways in which indices can be
    passed. Lists and arrays are passed unchanged, but ranges are turned
    to lists, and all other values are wrapped in a list.'''
    if not isinstance(idx, (list, np.ndarray)):
        if isinstance(idx, range):
            idx = list(idx)
        else:
            idx = [idx]
    return idx


# TODO: this could be changed to work on annotations too (I likely meant: to
#       compute covariance only for time periods contained in given annotation)
# TODO: default arguments should be changed
def compute_cov_raw(raw, events, event_id=11, tmin=1., tmax=60.):
    '''Compute covariance on long segments of raw data marked by events.
    '''
    # epoch data and turn bad annotations to NaNs
    raw = raw.copy()
    raw._data = raw.get_data(reject_by_annotation='NaN', verbose=False)
    epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True, verbose=False,
                        reject_by_annotation=False)

    # roll out the data to channels x times
    n_epochs, n_channels, n_samples = epochs._data.shape
    data = epochs._data.transpose((1, 0, 2)).reshape(
        n_channels, n_epochs * n_samples)

    # ignore NaN's and compute covariance
    not_nan = ~np.isnan(data).any(axis=0)
    data = data[:, not_nan]
    cov = (data @ data.T) / (not_nan.sum() * n_epochs)
    return cov


# TODO: make it work also for Epochs
# TODO: allow to pass arguments to compute_cov
def compute_narrowband_ged(raw, events, freq, cov_raw=None, freq_around=None,
                           filter_args=None, reg=0.05):
    # default values
    from numbers import Real

    freq_around = np.array([-0.5, 0.5]) if freq_around is None else freq_around
    filter_args = dict(h_trans_bandwidth=0.5, l_trans_bandwidth=0.5,
                       verbose=False) if filter_args is None else filter_args
    if isinstance (freq_around, Real):
        freq_around = np.array([freq_around] * 2)

    if cov_raw is None:
        cov_raw = compute_cov_raw(raw, events)

    f1, f2 = freq + freq_around
    raw_theta = raw.copy().filter(f1, f2, **filter_args)
    cov_theta = compute_cov_raw(raw_theta, events)
    ged = GED(cov_theta, cov_raw, reg=reg)
    return ged


# TODO: could use more than one component per frequency and then return
#       an array of components x channels x frequencies (for example)
def ged_scan_freqs(raw, events, freq_centers, freq_around=None,
                   filter_args=None, reg=0.05):
    '''Compute narrowband GED for a range of frequencies.

    Only the first component per frequency is saved (or rather - its filter and
    pattern).

    Parameters
    ----------
    raw : mne.io.Raw
        Instance of raw data to use.
    freq_centers : numpy.ndarray
        Frequency centers to use during filtering. GED will be computed for
        each frequency.
    freq_around : float | numpy.ndarray
        Width of the filter. Either one value specifying full width or
        two-element array specifying distance from frequency center on the left
        and right.
    filter_args : dict
        Dictionary with additional arguments used during filtering.
    reg : float
        Regularization value used in GED computation. Defaults to ``0.5``.

    Returns
    -------
    eigs : numpy.ndarray
        Array of component eigenvalues - one per frequency tested. Only the
        first component's eigenvalue is saved for each frequency.
    maps : numpy.ndarray
        Channels by frequencies array of component patterns. Only the
        first component's pattern is saved for each frequency.
    vecs : numpy.ndarray
        Channels by frequencies array of component eigenvectors. Only the
        first component's eigenvector is saved for each frequency.
    '''
    from tqdm import tqdm
    cov_raw = compute_cov_raw(raw, events)

    eigs, maps, vecs = list(), list(), list()
    for freq in tqdm(freq_centers):
        ged = compute_narrowband_ged(raw, events, freq, cov_raw=cov_raw,
                                     freq_around=freq_around,
                                     filter_args=filter_args, reg=reg)
        eigs.append(ged.eig[0])
        maps.append(ged.patterns[0, :])
        vecs.append(ged.filters[:, 0])

    maps = np.stack(maps, axis=1)
    vecs = np.stack(vecs, axis=1)
    eigs = np.array(eigs)

    return eigs, maps, vecs


# TODO: allow to show other number of components than 6
# TODO: in eigenvalues plot mark also which components are shown
def plot_ged_comps(ged, info, psd=None, fmin=1., fmax=15.):
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15, 5))
    gs = plt.GridSpec(2, 5)

    psd_ax = fig.add_subplot(gs[:2, :2])
    topo_ax = list()
    for row in range(2):
        topo_ax += [fig.add_subplot(gs[row, idx]) for idx in range(2, 5)]
    ged.plot(info, axes=topo_ax)

    if psd is not None:
        psd = psd.crop(fmin=1, fmax=15)
        psd_ax.plot(psd.freqs, psd.data.T)
        psd_ax.legend(np.arange(1, 7))
    else:
        psd_ax.plot(ged.eig)
