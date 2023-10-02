import os
from os.path import split
from functools import partial

import numpy as np
from scipy import sparse, signal
from scipy.io import loadmat

import mne
from mne.stats import permutation_cluster_test, ttest_1samp_no_p

import borsar
from borsar.utils import find_index
from borsar.stats import _find_stat_fun, _compute_threshold_via_permutations
from borsar.cluster import Clusters, construct_adjacency_matrix

from . import utils
from .stats import ttest_ind_no_p, ttest_rel_no_p


base_dir = split(__file__)[0]
chan_path = os.path.join(base_dir, 'data', 'chan')


# read channel connectivity
# consider renaming to read_neighbours
def get_neighbours(captype):
    assert isinstance(captype, str), 'captype must be a string.'
    if os.path.exists(captype):
        # file path was given
        file_name = captype
    else:
        # cap type was given
        fls = [f for f in os.listdir(chan_path) if f.endswith('.mat') and
               '_neighbours' in f]
        good_file = [f for f in fls if captype in f]
        if len(good_file) > 0:
            file_name = os.path.join(chan_path, good_file[0])
        else:
            raise ValueError('Could not find specified cap type.')
    return loadmat(file_name, squeeze_me=True)['neighbours']


def find_adjacency(inst, picks=None):
    '''Find channel adjacency matrix.'''
    from scipy.spatial import Delaunay
    from mne.channels.layout import _find_topomap_coords
    try:
        from mne.source_estimate import spatial_tris_connectivity as adjacency
    except:
        from mne.source_estimate import spatial_tris_adjacency as adjacency

    n_channels = len(inst.ch_names)
    picks = np.arange(n_channels) if picks is None else picks
    ch_names = [inst.info['ch_names'][pick] for pick in picks]
    xy = _find_topomap_coords(inst.info, picks)

    # first on 2x, y
    coords = xy.copy()
    coords[:, 0] *= 2
    tri = Delaunay(coords)
    neighbors1 = adjacency(tri.simplices)

    # then on x, 2y
    coords = xy.copy()
    coords[:, 1] *= 2
    tri = Delaunay(coords)
    neighbors2 = adjacency(tri.simplices)

    adjacency = neighbors1.toarray() | neighbors2.toarray()
    return adjacency, ch_names


def cluster(data, adjacency=None, min_adj_ch=0):
    from borsar.cluster.label import _get_cluster_fun
    clst_fun = _get_cluster_fun(data, adjacency, min_adj_ch=min_adj_ch)
    return clst_fun(data, adjacency, min_adj_ch=min_adj_ch)


# TODO: do not convert to sparse if already sparse
def cluster_1d(data, connectivity=None):
    from mne.stats.cluster_level import _find_clusters
    if connectivity is not None:
        connectivity = sparse.coo_matrix(connectivity)
    return _find_clusters(data, 0.5, connectivity=connectivity)


# TODO: this needs docs!
def cluster_spread(cluster, connectivity):
    n_chan = connectivity.shape[0]
    spread = np.zeros((n_chan, n_chan), 'int')
    unrolled = [cluster[ch, :].ravel() for ch in range(n_chan)]
    for ch in range(n_chan - 1):  # last chan will be already checked
        ch1 = unrolled[ch]

        # get unchecked neighbours
        neighbours = np.where(connectivity[ch + 1:, ch])[0]
        if neighbours.shape[0] > 0:
            neighbours += ch + 1

            for ngb in neighbours:
                ch2 = unrolled[ngb]
                num_connected = (ch1 & ch2).sum()
                spread[ch, ngb] = num_connected
                spread[ngb, ch] = num_connected
    return spread


# - [x] add min_channel_neighbours
# - [ ] add docs!
# - [ ] min_neighbours as a 0 - 1 float
# - [ ] include_channels (what was the idea here?)
def filter_clusters(mat, min_neighbours=4, min_channels=0, connectivity=None):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    mat = mat.copy()
    size = mat.shape
    if mat.ndim == 2:
        mat = mat[np.newaxis, :, :]
        size = mat.shape

    for ch in range(size[0]):
        enough_ngb = (signal.convolve2d(mat[ch, :, :], kernel, mode='same')
                      >= min_neighbours)
        mat[ch, :, :] = mat[ch, :, :] & enough_ngb
    if min_channels > 0:
        assert connectivity is not None
        for ch in range(size[0]):
            ngb = np.where(connectivity[ch, :])[0]
            mat[ch, :, :] = mat[ch, :, :] & (mat[ngb, :, :].sum(
                axis=0) >= min_channels)
    return mat


def remove_links(mat, min_pixels=5):
    '''Remove clusters that are smaller than min_pixels within any given
    slice (channel) of the matrix. These small blobs sometimes create
    weak links between otherwise strong clusters.'''
    from skimage.measure import label

    # label each channel separately
    n_chan = mat.shape[0]
    mat = mat.copy()
    for ch in range(n_chan):
        clusters = label(mat[ch, :, :], connectivity=1, background=False)
        n_clusters = clusters.max()
        for c in range(n_clusters):
            msk = clusters == (c + 1)
            if (msk).sum() < min_pixels:
                clusters[msk] = 0
        mat[ch, clusters == 0] = False
    return mat


def relabel_mat(mat, label_map):
    '''Change values in a matrix of integers such that mapping given
    in label_map dict is fulfilled.

    parameters
    ----------
    mat - numpy array of integers
    label_map - dictionary, how to remap integer labels

    returns
    -------
    mat_relab - relabeled numpy array
    '''
    mat_relab = mat.copy()
    for k, v in label_map.items():
        mat_relab[mat == k] = v
    return mat_relab


def smooth(matrix, sd=2.):
    from scipy.ndimage.filters import gaussian_filter
    matrix = matrix.copy()
    if matrix.ndim > 2:
        n_chan = matrix.shape[0]
        for ch in range(n_chan):
            matrix[ch, :] = gaussian_filter(matrix[ch, :], sd)
    else:
        matrix = gaussian_filter(matrix, sd)
    return matrix


def check_list_inst(data, inst):
    tps = list()
    for this_data in data:
        if not isinstance(this_data, inst):
            raise TypeError('One of the objects in data list does not '
                            'belong to supported mne objects (Evoked, '
                            'AverageTFR).')
        tps.append(type(this_data))
    all_same_type = [tp == tps[0] for tp in tps[1:]] if len(tps) > 1 else True
    if not all_same_type:
        raise TypeError('Not all objects in the data list are of the same'
                        ' mne object class.')


# - [ ] add TFR tests!
# - [ ] make sure min_adj_ch works with 2d
# - [ ] add Epochs to supported types if single_trial
# - [ ] one_sample is not passed to lower functions...
# - [ ] add 2-step tests?
def permutation_cluster_ttest(data1, data2, paired=False, n_permutations=1000,
                              threshold=None, p_threshold=0.05,
                              adjacency=None, tmin=None, tmax=None,
                              fmin=None, fmax=None, trial_level=False,
                              min_adj_ch=0):
    '''Perform cluster-based permutation test with t test as statistic.

    Parameters
    ----------
    data1 : list of mne objects
        List of objects (Evokeds, TFRs) belonging to condition one.
    data2 : list of mne objects
        List of objects (Evokeds, TFRs) belonging to condition two.
    paired : bool
        Whether to perform a paired t test. Defaults to ``True``.
    n_permutations : int
        How many permutations to perform. Defaults to ``1000``.
    threshold : value
        Cluster entry threshold defined by the value of the statistic. Defaults
        to ``None`` which calculates threshold from p value (see
        ``p_threshold``)
    p_threshold : value
        Cluster entry threshold defined by the p value.
    adjacency : boolean array | sparse array
        Information about channel adjacency.
    tmin : float
        Start of the time window of interest (in seconds). Defaults to ``None``
        which takes the earliest possible time.
    tmax : float
        End of the time window of interest (in seconds). Defaults to ``None``
        which takes the latest possible time.
    fmin : float
        Start of the frequency window of interest (in seconds). Defaults to
        ``None`` which takes the lowest possible frequency.
    fmax : float
        End of the frequency window of interest (in seconds). Defaults to
        ``None`` which takes the highest possible frequency.
    min_adj_ch: int
        Minimum number of adjacent in-cluster channels to retain a point in
        the cluster.

    Returns
    -------
    clst : borsar.cluster.Clusters
        Obtained clusters.
    '''
    if data2 is not None:
        one_sample = False
        stat_fun = ttest_rel_no_p if paired else ttest_ind_no_p
    else:
        one_sample = True
        stat_fun = lambda data: ttest_1samp_no_p(data[0])

    try:
        kwarg = 'connectivity'
        from mne.source_estimate import spatial_tris_connectivity
    except:
        kwarg = 'adjacency'
        from mne.source_estimate import spatial_tris_adjacency

    inst = data1[0]
    len1 = len(data1)
    len2 = len(data2) if data2 is not None else 0

    if paired:
        assert len1 == len2

    threshold = _compute_threshold([data1, data2], threshold, p_threshold,
                                   trial_level, paired, one_sample)

    # data1 and data2 have to be Evokeds or TFRs
    supported_types = (mne.Evoked, borsar.freq.PSD,
                       mne.time_frequency.AverageTFR,
                       mne.time_frequency.EpochsTFR)
    check_list_inst(data1, inst=supported_types)
    if data2 is not None:
        check_list_inst(data2, inst=supported_types)

    # find time and frequency ranges
    # ------------------------------
    if isinstance(inst, (mne.Evoked, mne.time_frequency.AverageTFR)):
        tmin = 0 if tmin is None else inst.time_as_index(tmin)[0]
        tmax = (len(inst.times) if tmax is None
                else inst.time_as_index(tmax)[0] + 1)
        time_slice = slice(tmin, tmax)

    if isinstance(inst, (borsar.freq.PSD, mne.time_frequency.AverageTFR)):
        fmin = 0 if fmin is None else find_index(data1[0].freqs, fmin)
        fmax = (len(inst.freqs) if fmax is None
                else find_index(data1[0].freqs, fmax))
        freq_slice = slice(fmin, fmax + 1)

    # handle object-specific data
    # ---------------------------
    if isinstance(inst, mne.time_frequency.AverageTFR):
        # + fmin, fmax
        assert not trial_level
        # data are in observations x channels x frequencies x time
        data1 = np.stack([tfr.data[:, freq_slice, time_slice]
                          for tfr in data1], axis=0)
        data2 = (np.stack([tfr.data[:, freq_slice, time_slice]
                           for tfr in data2], axis=0)
                 if data2 is not None else data2)
    elif isinstance(inst, mne.time_frequency.EpochsTFR):
        assert trial_level
        data1 = inst.data[..., freq_slice, time_slice]
        data2 = (data2[0].data[..., freq_slice, time_slice]
                 if data2 is not None else data2)
    elif isinstance(inst, borsar.freq.PSD):
        if not inst._has_epochs:
            assert not trial_level
            data1 = np.stack([psd.data[:, freq_slice].T for psd in data1],
                             axis=0)
            data2 = (np.stack([psd.data[:, freq_slice].T for psd in data2],
                              axis=0) if data2 is not None else data2)
        else:
            assert trial_level
            data1 = data1[0].data[..., freq_slice].transpose((0, 2, 1))
            data2 = (data2[0].data[..., freq_slice].transpose((0, 2, 1))
                     if data2 is not None else data2)
    else:
        data1 = np.stack([erp.data[:, time_slice].T for erp in data1], axis=0)
        data2 = (np.stack([erp.data[:, time_slice].T for erp in data2], axis=0)
                 if data2 is not None else data2)

    data_3d = data1.ndim > 3
    if (isinstance(adjacency, np.ndarray) and not sparse.issparse(adjacency)
        and not data_3d):
        adjacency = sparse.coo_matrix(adjacency)

    # perform cluster-based test
    # --------------------------
    # TODO: now our cluster-based works also for 1d and 2d etc.
    if not data_3d:
        assert min_adj_ch == 0
        adj_param = {kwarg: adjacency}
        stat, clusters, cluster_p, _ = permutation_cluster_test(
            [data1, data2], stat_fun=stat_fun, threshold=threshold,
            n_permutations=n_permutations, out_type='mask', **adj_param)
        if isinstance(inst, mne.Evoked):
            dimcoords = [inst.ch_names, inst.times[time_slice]]
            dimnames = ['chan', 'time']
        elif isinstance(inst, borsar.freq.PSD):
            dimcoords = [inst.ch_names, inst.freqs[freq_slice]]
            dimnames = ['chan', 'freq']
        return Clusters(stat.T, [c.T for c in clusters], cluster_p,
                        info=inst.info, dimnames=dimnames,
                        dimcoords=dimcoords)

    else:
        stat, clusters, cluster_p = permutation_cluster_test_array(
            [data1, data2], adjacency, stat_fun, threshold=threshold,
            n_permutations=n_permutations, one_sample=one_sample,
            paired=paired, min_adj_ch=min_adj_ch)

        # pack into Clusters object
        dimcoords = [inst.ch_names, inst.freqs, inst.times[tmin:tmax]]
        return Clusters(stat, clusters, cluster_p, info=inst.info,
                        dimnames=['chan', 'freq', 'time'], dimcoords=dimcoords)
