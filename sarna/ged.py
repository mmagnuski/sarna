import numpy as np
import scipy

import mne
from borsar.viz import Topo


# TODO: add plot_eig for plotting eigenvalues
# TODO: add save and read_ged?
# TODO: we use eig.real - maybe a warning should be added if we get complex
#       values
class GED(object):
    def __init__(self, cov_S=None, cov_R=None, reg=None, eig=None,
                 filters=None, patterns=None, description=None):
        '''
        Compute Generalized Eigendecomposition (GED) of two covariance
        matrices.

        Prameters
        ---------
        cov_S : mne Covariance | array
            Covariance matrix for the signal of interest.
        cov_R : mne Covariance | array
            Covariance matrix for the reference signal.
        reg : None | float
            Regularization factor, from 0 - 1. If ``None`` then regularization
            is not performed.
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
