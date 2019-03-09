import os
from os.path import split
from functools import partial

import numpy as np
from scipy import sparse, signal
from scipy.io import loadmat

import mne
from mne.stats import permutation_cluster_test
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
    assert isinstance(inst, (tps['raw'], tps['epochs'], tps['info']))
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
        print(node_ind)
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
                        line = axes.plot(selected_pos[:, 0], selected_pos[:, 1],
                                         selected_pos[:, 2], lw=get_lw())[0]
                    elif kind == '2d':
                        line = axes.plot(selected_pos[:, 0], selected_pos[:, 1],
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


def cluster_spread(cluster, connectivity):
    n_chan = connectivity.shape[0]
    spread = np.zeros((n_chan, n_chan), 'int')
    unrolled = [cluster[ch, :].ravel() for ch in range(n_chan)]
    for ch in range(n_chan - 1): # last chan will be already checked
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
        mat[ch, :, :] = mat[ch, :, :] & (signal.convolve2d(mat[ch, :, :],
                                         kernel, mode='same') >= min_neighbours)
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
    '''change values in a matrix of integers such that mapping given
    in label_map dict is fulfilled

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
            matrix[ch,:] = gaussian_filter(matrix[ch,:], sd)
    else:
        matrix = gaussian_filter(matrix, sd)
    return matrix


def permutation_cluster_t_test(data1, data2, paired=False, n_permutations=1000,
                               threshold=None, p_threshold=0.05,
                               adjacency=None, tmin=None, tmax=None):
    '''FIXME: add docs.'''
    stat_fun = ttest_rel_no_p if paired else ttest_ind_no_p

    inst = data1[0]
    len1, len2 = len(data1), len(data2)
    if paired:
        assert len1 == len2

    if threshold is None:
        from scipy.stats import distributions
        df = (len1 - 1 if paired else
              len1 + len2 - 2)
        threshold = np.abs(distributions.t.ppf(p_threshold / 2., df=df))

    # data1 and data2 have to be Evokeds
    assert all([isinstance(dt, mne.Evoked) for dt in data1])
    assert all([isinstance(dt, mne.Evoked) for dt in data2])

    tmin = 0 if tmin is None else inst.time_as_index(tmin)[0]
    tmax = (len(inst.times) if tmax is None
            else inst.time_as_index(tmax)[0] + 1)

    data1 = np.stack([erp.data[:, tmin:tmax].T for erp in data1], axis=0)
    data2 = np.stack([erp.data[:, tmin:tmax].T for erp in data2], axis=0)

    if isinstance(adjacency, np.ndarray) and not sparse.issparse(adjacency):
        adjacency = sparse.coo_matrix(adjacency)

    stat, clusters, cluster_p, _ = permutation_cluster_test(
        [data1, data2], stat_fun=stat_fun, threshold=threshold,
        connectivity=adjacency, n_permutations=n_permutations)

    dimcoords = [inst.ch_names, inst.times[tmin:tmax]]
    return Clusters([c.T for c in clusters], cluster_p, stat.T, info=inst.info,
                    dimnames=['chan', 'time'], dimcoords=dimcoords)


def has_numba():
    try:
        from numba import jit
        return True
    except ImportError:
        return False
