import numpy as np
from numba import jit


def find_clusters_3d(data, adj):
    ch, d1, d2 = data.shape
    neighbours = find_neighbours_numba(adj)
    clusters = np.zeros((ch, d1, d2), dtype='int64')
    clusters = _find_clusters_3d_numba(data, clusters, neighbours)
    return clusters


@jit(nopython=True)
def _find_clusters_3d_numba(data, clusters, neighbours):
    current_cluster = 0
    ch, d1, d2 = data.shape

    for dim in range(ch):
        for idx1 in range(d1):
            for idx2 in range(d2):
                if data[dim, idx1, idx2]:
                    ngb_clusters = _check_neighbours_numba(
                        clusters, neighbours, dim, idx1, idx2)
                    if len(ngb_clusters) > 0:
                        clusters[dim, idx1, idx2] = ngb_clusters[0]
                    else:
                        current_cluster += 1
                        clusters[dim, idx1, idx2] = current_cluster

    return clusters


# - [ ] this is mostly for tests
def find_neighbours(adj):
    n_ch = len(adj)
    neighbours = list()
    for idx in range(1, n_ch):
        neighbours.append(np.nonzero(adj[idx, :idx])[0])
    return neighbours


@jit(nopython=True)
def find_neighbours_numba(adj):
    n_ch = len(adj)
    neighbours = list()
    for idx in range(1, n_ch):
        neighbours.append(np.nonzero(adj[idx, :idx])[0])
    return neighbours


# - [ ] neighbouring_clusters is bad will all the concatenations...
#       maybe construct list and only then turn to array
# - [ ] removed correspondence
@jit(nopython=True)
def _check_neighbours_numba(clusters, neighbours, ch, idx1, idx2):
    neighbouring_clusters = [np.int64(x) for x in range(0)]

    # adjacency-defined layer
    if ch > 0:
        ngb = neighbours[ch - 1]
        if len(ngb) > 0:
            neighbouring_clusters += unique_nonzero(clusters[ngb, idx1, idx2])

    # idx1 layer
    if idx1 > 0:
        val = clusters[ch, idx1 - 1, idx2]
        if val > 0:
            neighbouring_clusters.append(val)

    # idx2 layer
    if idx2 > 0:
        val = clusters[ch, idx1, idx2 - 1]
        if val > 0:
            neighbouring_clusters.append(val)

    if len(neighbouring_clusters) > 0:
        neighbouring_clusters = unique(neighbouring_clusters)

    # fill correspondence
    # if len(neighbouring_clusters) > 1:
    #     _handle_correspondence(correspondence, neighbouring_clusters)

    return neighbouring_clusters


@jit(nopython=True)
def is_in_array(element, array):
    for val in array:
        if element == val:
            return True
    return False


@jit(nopython=True)
def unique_nonzero(ngb):
    """Return unique non-zero values from vector."""
    uni = list()
    for n in ngb:
        if n > 0 and n not in uni:
            uni.append(n)
    return uni


@jit(nopython=True)
def unique(ngb):
    """Return unique values from vector."""
    uni = list()
    for n in ngb:
        if n not in uni:
            uni.append(n)
    return uni


# - [ ] cleanup names
@jit(nopython=True)
def _handle_correspondence(correspondence, same_clusters):
    if len(correspondence) == 0:
        correspondence.append(same_clusters)
    else:
        isin = list()
        nopair = list()
        for clst_idx in same_clusters:
            this_nopair = True
            for idx, group in enumerate(correspondence):
                if is_in_array(clst_idx, group):
                    isin.append(idx)
                    this_nopair = False
                    break
            nopair.append(this_nopair)

        if len(isin) == 0:
            correspondence.append(same_clusters)
        else:
            isin = unique(isin)

            if len(isin) == 1:
                # just join those not present:
                new_clst_id = same_clusters[np.array(nopair)]
                if len(new_clst_id) > 0:
                    comp = isin[0]
                    correspondence[comp] = np.concatenate(
                        (correspondence[comp], new_clst_id))
            else:
                first = isin[0]
                # merge and remove
                for idx in isin[1:][::-1]:
                    correspondence[first] = np.concatenate(
                        (correspondence[first], correspondence.pop(idx)))

                new_clst_id = same_clusters[np.array(nopair)]
                if len(new_clst_id) > 0:
                    correspondence[first] = np.concatenate(
                        (correspondence[first], new_clst_id))


# - [ ] clean up relabel
def _relabel():
    relabel = np.arange(0, last_clst + 1)
    # full relabel
    for corresp in csp:
        first = corresp[0]
        for rename in corresp[1:]:
            relabel[rename] = first

    last_max = 0
    for idx, el in enumerate(relabel):
        diff = el - last_max
        if diff == 1:
            last_max = el
        if diff > 1:
            last_max += 1
            relabel[idx] = last_max


# - [ ] move to tests
def test_uniques():
    unsifted1 = np.array([0, 0, 1, 0, 2, 0, 1, 0, 0, 3, 1, 0, 2, 0, 0, 4])
    unsifted2 = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])

    assert np.array(unique(unsifted1)) == np.array([0, 1, 2, 3, 4])
    assert np.array(unique_nonzero(unsifted1)) == np.array([1, 2, 3, 4])
    assert np.array(unique(unsifted2)) == np.array([0])
    assert np.array(unique_nonzero(unsifted2)) == np.array([])
