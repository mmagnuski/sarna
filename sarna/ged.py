import numpy as np
import scipy

import mne
from borsar.viz import Topo


# TODO: add plot_eig for plotting eigenvalues
# TODO: add save and read_ged?
# TODO: we use eig.real - maybe a warning should be added if we get complex
#       values
class GED(object):
    def __init__(self, cov_S, cov_R, reg=None):
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
        eig, vec = scipy.linalg.eig(cov_S, cov_R)
        eig = eig.real
        srt = np.argsort(eig)[::-1]
        eig, vec = eig[srt], vec[:, srt]

        # compose attributes
        self.eig = eig
        self.filters = vec
        self.patterns = _get_patterns(self.filters, cov_S)

    def plot(self, info, idx=None, axes=None, **args):
        '''
        Plot GED component topographical patterns.

        info  : mne Info instance
            mne-python Info object - used for topomap plotting.
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

    def save(self):
        '''save to hdf5: filters, patterns, eig'''
        raise NotImplementedError


def _get_patterns(vec, cov):
    n_signals = vec.shape[0]
    patterns = np.zeros((n_signals, n_signals))

    for idx in range(n_signals):
        patterns[idx, :] = vec[:, [idx]].T @ cov.data

    return patterns


def _deal_with_idx(idx):
    if not isinstance(idx, (list, np.ndarray)):
        idx = [idx]
    return idx
