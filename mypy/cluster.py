import os
from os.path import split

import numpy as np
from scipy import sparse
from scipy.io import loadmat
# import matplotlib.pyplot as plt


base_dir = split(split(__file__)[0])[0]
chan_path = os.path.join(base_dir, 'data', 'chan')


# read channel connectivity
def get_chan_conn():
    pth = os.path.join(chan_path, 'BioSemi64_chanconn.mat')
    return loadmat(pth)['chan_conn'].astype('bool')


def get_neighbours(captype):
    assert isinstance(captype, str), 'captype must be a string.'
    fls = [f for f in os.listdir(chan_path) if f.endswith('.mat') and
            '_neighbours' in f]
    good_file = [f for f in fls if captype in f]
    if len(good_file) > 0:
        return loadmat(os.path.join(chan_path, good_file[0]),
                       squeeze_me=True)['neighbours']
    else:
        raise ValueError('Could not find specified cap type.')


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

# another approach to random colors:
# plt.cm.viridis(np.linspace(0., 1., num=15) , alpha=0.5)
def plot_neighbours(inst, adj_matrix, color='gray'):
    '''Plot channel adjacency.

    Parameters
    ----------
    inst : mne Raw or Epochs
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
    from mne.io import _BaseRaw
    from mne.epochs import _BaseEpochs
    from .viz import set_3d_axes_equal
    assert isinstance(inst, (_BaseRaw, _BaseEpochs))

    if adj_matrix.dtype == 'int':
        max_lw = 10.
        max_conn = adj_matrix.max()
        def get_lw():
            return adj_matrix[ch, n] / max_conn * max_lw
    elif adj_matrix.dtype == 'bool':
        def get_lw():
            return 2.

    fig = inst.plot_sensors(kind='3d')
    set_3d_axes_equal(fig.axes[0])
    pos = np.array([x['loc'][:3] for x in inst.info['chs']])
    for ch in range(adj_matrix.shape[0]):
        ngb = np.where(adj_matrix[ch, :])[0]
        for n in ngb:
            this_pos = pos[[ch, n], :]
            if not color == 'random':
                fig.axes[0].plot(this_pos[:, 0], this_pos[:, 1],
                                 this_pos[:, 2], color=color,
                                 lw=get_lw())
            else:
                fig.axes[0].plot(this_pos[:, 0], this_pos[:, 1],
                                 this_pos[:, 2], lw=get_lw())
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
# - [ ] include_channels
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
