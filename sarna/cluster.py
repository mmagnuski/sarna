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


# - [ ] add edit option (runs in interactive mode only)
# - [ ] new lines should have one color
# - [x] 'random' is actually misleading - it follows colorcycle...
# another approach to random colors:
# plt.cm.viridis(np.linspace(0., 1., num=15) , alpha=0.5)
def plot_neighbours(inst, adj_matrix, color='gray', kind='3d'):
    '''Plot channel adjacency.

    Parameters
    ----------
    inst : mne Raw, Epochs or info
        mne-python data container
    adj_matrix : boolean numpy array
        Defines which channels are adjacent to each other.
    color : matplotlib color or 'random'
        Color to plot the web of adjacency relations with.

    Returns
    -------
    fig : matplotlib figure
        Figure.
    '''
    tps = utils.mne_types()
    from .viz import set_3d_axes_equal
    from mne.viz import plot_sensors
    assert isinstance(inst, (tps['raw'], tps['epochs'], tps['info'],
                             mne.Evoked))
    info = utils.get_info(inst)

    if isinstance(adj_matrix, sparse.coo_matrix):
        adj_matrix = adj_matrix.toarray()

    if adj_matrix.dtype == 'int':
        max_lw = 5.
        max_conn = adj_matrix.max()
        def get_lw():
            return adj_matrix[ch, n] / max_conn * max_lw
    elif adj_matrix.dtype == 'bool' or (np.unique(adj_matrix) ==
                                        np.array([0., 1.])).all():
        def get_lw():
            return 1.

    if kind == '3d':
        fig = plot_sensors(info, kind=kind, show=False)
        pos = np.array([x['loc'][:3] for x in info['chs']])
        set_3d_axes_equal(fig.axes[0])
    elif kind == '2d':
        import matplotlib as mpl
        fig = plot_sensors(info, kind='topomap', show=False)
        fig.axes[0].axis('equal')
        path_collection = fig.axes[0].findobj(mpl.collections.PathCollection)
        pos = path_collection[0].get_offsets()
        path_collection[0].set_zorder(10)

    lines = dict()
    for ch in range(adj_matrix.shape[0]):
        ngb = np.where(adj_matrix[ch, :])[0]
        for n in ngb:
            this_pos = pos[[ch, n], :]
            chan_pair = [ch, n]
            sorted(chan_pair)
            this_color = color if not color == 'random' else np.random.random()
            if kind == '3d':
                lines[tuple(chan_pair)] = fig.axes[0].plot(
                    this_pos[:, 0], this_pos[:, 1], this_pos[:, 2],
                    color=this_color, lw=get_lw())[0]
            elif kind == '2d':
                lines[tuple(chan_pair)] = fig.axes[0].plot(
                    this_pos[:, 0], this_pos[:, 1],
                    color=this_color, lw=get_lw())[0]

    highlighted = list()
    highlighted_scatter = list()

    def onpick(event, axes=None, positions=None, highlighted=None,
               line_dict=None, highlighted_scatter=None, adj_matrix=None):
        node_ind = event.ind[0]
        if node_ind in highlighted:
            # change node color back to normal
            highlighted_scatter[0].remove()
            highlighted_scatter.pop(0)
            highlighted.pop(0)
            fig.canvas.draw()
        else:
            if len(highlighted) == 0:
                # add current node
                highlighted.append(node_ind)
                if kind == '3d':
                    scatter = axes.scatter(positions[node_ind, 0],
                                           positions[node_ind, 1],
                                           positions[node_ind, 2],
                                           c='r', s=100, zorder=15)
                elif kind == '2d':
                    scatter = axes.scatter(positions[node_ind, 0],
                                           positions[node_ind, 1],
                                           c='r', s=100, zorder=15)

                highlighted_scatter.append(scatter)
                fig.canvas.draw()
            else:
                # add or remove line
                both_nodes = [highlighted[0], node_ind]
                sorted(both_nodes)

                if tuple(both_nodes) in line_dict.keys():
                    # remove line
                    line_dict[tuple(both_nodes)].remove()
                    # remove line_dict entry
                    del line_dict[tuple(both_nodes)]
                    # clear adjacency matrix entry
                    adj_matrix[both_nodes[0], both_nodes[1]] = False
                    adj_matrix[both_nodes[1], both_nodes[0]] = False
                else:
                    # add line
                    selected_pos = positions[both_nodes, :]
                    if kind == '3d':
                        line = axes.plot(
                            selected_pos[:, 0], selected_pos[:, 1],
                            selected_pos[:, 2], lw=get_lw())[0]
                    elif kind == '2d':
                        line = axes.plot(selected_pos[:, 0],
                                         selected_pos[:, 1],
                                         lw=get_lw())[0]
                    # add line to line_dict
                    line_dict[tuple(both_nodes)] = line
                    # modify adjacency matrix
                    adj_matrix[both_nodes[0], both_nodes[1]] = True
                    adj_matrix[both_nodes[1], both_nodes[0]] = True

                # highlight new node, de-highligh previous
                highlighted.append(node_ind)
                if kind == '3d':
                    scatter = axes.scatter(positions[node_ind, 0],
                                           positions[node_ind, 1],
                                           positions[node_ind, 2],
                                           c='r', s=100, zorder=10)
                elif kind == '2d':
                    scatter = axes.scatter(positions[node_ind, 0],
                                           positions[node_ind, 1],
                                           c='r', s=100, zorder=10)

                highlighted_scatter.append(scatter)
                highlighted_scatter[0].remove()
                highlighted_scatter.pop(0)
                highlighted.pop(0)
                fig.canvas.draw()

    this_onpick = partial(onpick, axes=fig.axes[0], positions=pos,
                          highlighted=list(), line_dict=lines,
                          highlighted_scatter=list(), adj_matrix=adj_matrix)
    fig.canvas.mpl_connect('pick_event', this_onpick)
    return fig


def find_adjacency(inst, picks=None):
    '''Find channel adjacency matrix.'''
    from scipy.spatial import Delaunay
    from mne.channels.layout import _find_topomap_coords
    from mne.source_estimate import spatial_tris_adjacency

    n_channels = len(inst.ch_names)
    picks = np.arange(n_channels) if picks is None else picks
    ch_names = [inst.info['ch_names'][pick] for pick in picks]
    xy = _find_topomap_coords(inst.info, picks)

    # first on 2x, y
    coords = xy.copy()
    coords[:, 0] *= 2
    tri = Delaunay(coords)
    neighbors1 = spatial_tris_adjacency(tri.simplices)

    # then on x, 2y
    coords = xy.copy()
    coords[:, 1] *= 2
    tri = Delaunay(coords)
    neighbors2 = spatial_tris_adjacency(tri.simplices)

    adjacency = neighbors1.toarray() | neighbors2.toarray()
    return adjacency, ch_names


def cluster(data, adjacency=None):
    from borsar.cluster import _get_cluster_fun
    clst_fun = _get_cluster_fun(data, adjacency)
    return clst_fun(data, adjacency)


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


# - [x] +fmin, +fmax
# - [ ] add TFR tests!
# - [x] add support for PSD
# - [ ] add 2-step tests?
# - [ ] consider renaming to ..._ttest
# - [ ] or consider adding ANOVA... (then it would be permutation_cluster_test)
# - [ ] one_sample is not passed to lower functions...
def permutation_cluster_t_test(data1, data2, paired=False, n_permutations=1000,
                               threshold=None, p_threshold=0.05,
                               adjacency=None, tmin=None, tmax=None,
                               fmin=None, fmax=None, trial_level=False):
    '''Perform cluster-based permutation test with t test as statistic.

    data1 : list of mne objects
        List of objects (Evokeds, TFRs) belonging to condition one.
    data2 : list of mne objects
        List of objects (Evokeds, TFRs) belonging to condition two.
    paired : bool
        Whether to perform a paired t test. Defaults to ``True``.
    n_permutations : int
        How many permutations to perform. Defaults to ``1000``.
    threshold : value
        Cluster entry threshold defined by the value of the statistic. Defautls
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

    inst = data1[0]
    len1 = len(data1)
    len2 = len(data2) if data2 is not None else 0

    if paired:
        assert len1 == len2

    if threshold is None:
        from scipy.stats import distributions
        if trial_level:
            df = data1[0].data.shape[0] + data1[0].data.shape[1] - 2
        else:
            df = (len1 - 1 if paired or one_sample else
                  len1 + len2 - 2)
        threshold = np.abs(distributions.t.ppf(p_threshold / 2., df=df))

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

    if not data_3d:
        stat, clusters, cluster_p, _ = permutation_cluster_test(
            [data1, data2], stat_fun=stat_fun, threshold=threshold,
            connectivity=adjacency, n_permutations=n_permutations)
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
        stat, clusters, cluster_p = _permutation_cluster_test_3d(
            [data1, data2], adjacency, stat_fun, threshold=threshold,
            n_permutations=n_permutations)
        dimcoords = [inst.ch_names, inst.freqs, inst.times[tmin:tmax]]
        return Clusters(stat, clusters, cluster_p, info=inst.info,
                        dimnames=['chan', 'freq', 'time'], dimcoords=dimcoords)


# FIX this! (or is it in borsar?)
def _permutation_cluster_test_3d(data, adjacency, stat_fun, threshold=None,
                                 one_sample=True, p_threshold=0.05,
                                 n_permutations=1000, progressbar=True,
                                 return_distribution=False, backend='auto'):
    """FIXME: add docs."""

    from borsar.cluster import _get_cluster_fun

    if progressbar:
        from tqdm import tqdm_notebook
        pbar = tqdm_notebook(total=n_permutations)

    n_obs = data[0].shape[0]
    # signs = np.array([-1, 1])
    signs_size = tuple([n_obs] + [1] * (data[0].ndim - 1))

    pos_dist = np.zeros(n_permutations)
    neg_dist = np.zeros(n_permutations)

    # test on non-permuted data
    stat = stat_fun(data)

    # use 3d clustering
    cluster_fun = _get_cluster_fun(stat, adjacency=adjacency,
                                   backend=backend)

    # we need to transpose dimensions for 3d clustering
    # FIXME/TODO - this could be eliminated by creating a single unified
    #              clustering function / API
    # jdata_dims = list(range(data[0].ndim))
    # data_dims[1], data_dims[-1] = data_dims[-1], 1
    # stat = stat.transpose(data_dims[1:] - 1)
    pos_clusters = cluster_fun(stat > threshold, adjacency)
    neg_clusters = cluster_fun(stat < -threshold, adjacency)

    # FIXME/TODO - move the part below to separate clustering function
    #              consider numba optimization too...
    pos_cluster_id = np.unique(pos_clusters)[1:]
    neg_cluster_id = np.unique(neg_clusters)[1:]
    clusters = ([pos_clusters == id for id in pos_cluster_id]
                + [neg_clusters == id for id in neg_cluster_id])
    cluster_stats = np.array([stat[clst].sum() for clst in clusters])

    if not clusters:
        print('No clusters found, permutations are not performed.')
        return stat, clusters, cluster_stats
    else:
        msg = 'Found {} clusters, computing permutations.'
        print(msg.format(len(clusters)))

    # compute permutations
    for perm in range(n_permutations):
        # permute predictors
        if one_sample:
            idx = np.random.random_integers(0, 1, size=signs_size)
            # perm_signs = signs[idx]
            # perm_data = data[0] * perm_signs
            perm_stat = stat_fun(data)

        # FIXME/TODO - move the part below to separate clustering function
        #              consider numba optimization too...
        perm_pos_clusters = cluster_fun(perm_stat > threshold, adjacency)
        perm_neg_clusters = cluster_fun(perm_stat < -threshold, adjacency)

        perm_pos_cluster_id = np.unique(perm_pos_clusters)[1:]
        perm_neg_cluster_id = np.unique(perm_neg_clusters)[1:]
        perm_neg_clusters = [perm_neg_clusters == id
                             for id in perm_neg_cluster_id]
        perm_clusters = ([perm_pos_clusters == id for id in
                          perm_pos_cluster_id] + perm_neg_clusters)
        perm_cluster_stats = np.array([perm_stat[clst].sum()
                                       for clst in perm_clusters])

        # if any clusters were found - add max statistic
        if len(perm_cluster_stats) > 0:
            max_val = perm_cluster_stats.max()
            min_val = perm_cluster_stats.min()

            if max_val > 0:
                pos_dist[perm] = max_val
            if min_val < 0:
                neg_dist[perm] = min_val

        if progressbar:
            pbar.update(1)

    # compute permutation probability
    cluster_p = np.array([(pos_dist > cluster_stat).mean() if cluster_stat > 0
                          else (neg_dist < cluster_stat).mean()
                          for cluster_stat in cluster_stats])
    cluster_p *= 2 # because we use two-tail
    cluster_p[cluster_p > 1.] = 1. # probability has to be <= 1.

    # FIXME: this may not be needed because Clusters sorts by p val...
    # sort clusters by p value
    cluster_order = np.argsort(cluster_p)
    cluster_p = cluster_p[cluster_order]
    clusters = [clusters[i] for i in cluster_order]

    if return_distribution:
        return stat, clusters, cluster_p, dict(pos=pos_dist, neg=neg_dist)
    else:
        return stat, clusters, cluster_p
