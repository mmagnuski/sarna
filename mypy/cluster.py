import os
from os.path import split

import numpy as np
from scipy import sparse
from scipy.io import loadmat
# import matplotlib.pyplot as plt

from mne.stats.cluster_level import _find_clusters


base_dir = split(split(__file__)[0])[0]
chan_path = os.path.join(base_dir, 'data', 'chan')


# read channel connectivity
def get_chan_conn():
    pth = os.path.join(chan_path, 'BioSemi64_chanconn.mat')
    return loadmat(pth)['chan_conn'].astype('bool')

# plt.imshow(chan_conn, interpolation='none')
# plt.show()


def cluster_1d(data, connectivity=None):
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
        clusters[ch, clusters[ch,:]>0] += max_cluster_id
        max_cluster_id += num_clusters

    # unrolled views into clusters for ease of channel comparison:
    unrolled = [clusters[ch, :].ravel() for ch in range(n_chan)]
    # check channel neighbours and merge clusters across channels
    for ch in range(n_chan-1): # last chan will be already checked
        ch1 = unrolled[ch]
        ch1_ind = np.where(ch1)[0]
        if ch1_ind.shape[0] == 0:
            continue # no clusters, no fun...

        # get unchecked neighbours
        neighbours = np.where(chan_conn[ch+1:, ch])[0]
        if neighbours.shape[0] > 0:
            neighbours += ch + 1

            for ngb in neighbours:
                ch2 = unrolled[ngb]
                for ind in ch1_ind:
                    # relabel clusters if adjacent and not the same id
                    if ch2[ind] and not (ch1[ind] == ch2[ind]):
                        c1 = min(ch1[ind], ch2[ind])
                        c2 = max(ch1[ind], ch2[ind])
                        clusters[clusters==c2] = c1
    return clusters


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
