import os
from os.path import split
from functools import partial

import numpy as np
from scipy import sparse
from scipy.io import loadmat

from mypy import utils
# import matplotlib.pyplot as plt


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


def construct_adjacency_matrix(ch_names, neighbours, sparse=False):
    # check input
    assert isinstance(ch_names, list), 'ch_names must be a list.'
    assert all(map(lambda x: isinstance(x, str), ch_names)), \
        'ch_names must be a list of strings'

    from scipy import sparse as sprs

    if isinstance(neighbours, str):
        neighbours = get_neighbours(neighbours)
    n_channels = len(ch_names)
    conn = np.zeros((n_channels, n_channels), dtype='bool')

    for ii, chan in enumerate(ch_names):
        ngb_ind = np.where(neighbours['label'] == chan)[0]

        # safty checks:
        if len(ngb_ind) == 0:
            raise ValueError('channel {} was not found in neighbours.'.format(
                             chan))
        elif len(ngb_ind) == 1:
            ngb_ind = ngb_ind[0]
        else:
            raise ValueError('found more than one neighbours entry for '
                             'channel name {}.'.format(chan))

        # find connections and fill up adjacency matrix
        connections = [ch_names.index(ch) for ch in neighbours['neighblabel']
                       [ngb_ind] if ch in ch_names]
        chan_ind = ch_names.index(chan)
        conn[chan_ind, connections] = True
    if sparse:
        return sprs.coo_matrix(conn)
    else:
        return conn


# - [ ] add edit option (runs in interactive mode only)
# - [ ] 'random' is actually misleading - it follows colorcycle...
# another approach to random colors:
# plt.cm.viridis(np.linspace(0., 1., num=15) , alpha=0.5)
def plot_neighbours(inst, adj_matrix, color='gray'):
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
    assert isinstance(inst, (tps['raw'], tps['epochs'], tps['info']))

    if isinstance(adj_matrix, sparse.coo_matrix):
        adj_matrix = adj_matrix.todense()

    if adj_matrix.dtype == 'int':
        max_lw = 10.
        max_conn = adj_matrix.max()
        def get_lw():
            return adj_matrix[ch, n] / max_conn * max_lw
    elif adj_matrix.dtype == 'bool':
        def get_lw():
            return 2.

    if isinstance(inst, tps['info']):
        from mne.viz import plot_sensors
        fig = plot_sensors(inst, kind='3d', show=False)
        pos = np.array([x['loc'][:3] for x in inst['chs']])
    else:
        fig = inst.plot_sensors(kind='3d', show=False)
        pos = np.array([x['loc'][:3] for x in inst.info['chs']])
    set_3d_axes_equal(fig.axes[0])

    lines = dict()
    for ch in range(adj_matrix.shape[0]):
        ngb = np.where(adj_matrix[ch, :])[0]
        for n in ngb:
            this_pos = pos[[ch, n], :]
            chan_pair = [ch, n]
            sorted(chan_pair)
            if not color == 'random':
                lines[tuple(chan_pair)] = fig.axes[0].plot(
                    this_pos[:, 0], this_pos[:, 1], this_pos[:, 2],
                    color=color, lw=get_lw())[0]
            else:
                lines[tuple(chan_pair)] = fig.axes[0].plot(
                    this_pos[:, 0], this_pos[:, 1], this_pos[:, 2],
                    lw=get_lw())[0]

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
                scatter = axes.scatter(positions[node_ind, 0],
                                       positions[node_ind, 1],
                                       positions[node_ind, 2],
                                       c='r', s=100, zorder=10)
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
                    line = axes.plot(selected_pos[:, 0], selected_pos[:, 1],
                                     selected_pos[:, 2], lw=get_lw())[0]
                    # add line to line_dict
                    line_dict[tuple(both_nodes)] = line
                    # modify adjacency matrix
                    adj_matrix[both_nodes[0], both_nodes[1]] = True
                    adj_matrix[both_nodes[1], both_nodes[0]] = True

                # highlight new node, de-highligh previous
                highlighted.append(node_ind)
                scatter = axes.scatter(positions[node_ind, 0],
                                       positions[node_ind, 1],
                                       positions[node_ind, 2], c='r', s=100,
                                       zorder=10)
                highlighted_scatter.append(scatter)
                highlighted_scatter[0].remove()
                highlighted_scatter.pop(0)
                highlighted.pop(0)
                fig.canvas.draw()

            print(highlighted)

    this_onpick = partial(onpick, axes=fig.axes[0], positions=pos,
                          highlighted=list(), line_dict=lines,
                          highlighted_scatter=list(), adj_matrix=adj_matrix)
    fig.canvas.mpl_connect('pick_event', this_onpick)
    return fig


# TODO: do not convert to sparse if already sparse
def cluster_1d(data, connectivity=None):
    from mne.stats.cluster_level import _find_clusters
    if connectivity is not None:
        connectivity = sparse.coo_matrix(connectivity)
    return _find_clusters(data, 0.5, connectivity=connectivity)


def cluster_3d(matrix, chan_conn):
    '''
    parameters
    ----------
    matrix - 3d matrix: channels by dim2 by dim3
    chan_conn - 2d boolean matrix with information about
        channel adjacency. If chann_conn[i, j] is True that
        means channel i and j are adjacent.

    returns
    -------
    clusters - 3d integer matrix with cluster labels
    '''
    # matrix has to be bool
    assert matrix.dtype == np.bool

    # nested import
    from skimage.measure import label

    # label each channel separately
    clusters = np.zeros(matrix.shape, dtype='int')
    max_cluster_id = 0
    n_chan = matrix.shape[0]
    for ch in range(n_chan):
        clusters[ch, :, :] = label(matrix[ch, :, :],
            connectivity=1, background=False)

        # relabel so that layers do not have same cluster ids
        num_clusters = clusters[ch, :, :].max()
        clusters[ch, clusters[ch,:] > 0] += max_cluster_id
        max_cluster_id += num_clusters

    # unrolled views into clusters for ease of channel comparison:
    unrolled = [clusters[ch, :].ravel() for ch in range(n_chan)]
    # check channel neighbours and merge clusters across channels
    for ch in range(n_chan - 1): # last chan will be already checked
        ch1 = unrolled[ch]
        ch1_ind = np.where(ch1)[0]
        if ch1_ind.shape[0] == 0:
            continue # no clusters, no fun...

        # get unchecked neighbours
        neighbours = np.where(chan_conn[ch + 1:, ch])[0]
        if neighbours.shape[0] > 0:
            neighbours += ch + 1

            for ngb in neighbours:
                ch2 = unrolled[ngb]
                for ind in ch1_ind:
                    # relabel clusters if adjacent and not the same id
                    if ch2[ind] and not (ch1[ind] == ch2[ind]):
                        c1 = min(ch1[ind], ch2[ind])
                        c2 = max(ch1[ind], ch2[ind])
                        clusters[clusters == c2] = c1
    return clusters


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
# - [ ] include_channels (what was the idea here?)
# - [ ] min_neighbours as a 0-1 float
def filter(mat, min_neighbours=4, min_channels=0, connectivity=None):
    from scipy import signal

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
    slice (channel) of the matrix. These small blobs often/sometimes create
    weak links of otherwise strong clusters.'''
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
