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


# TODO: add condition order argument? This may require a large refactoring of
#       the function to allow for 2-step tests (step 1 - within subjects,
#       step 2 - across subjects)
# TODO: move `min_adj_ch` up and add `min_adj`
def permutation_cluster_test_array(data, adjacency, stat_fun=None,
                                   threshold=None, p_threshold=0.05,
                                   paired=False, one_sample=False, tail='both',
                                   n_permutations=1000, n_stat_permutations=0,
                                   progress=True, return_distribution=False,
                                   backend='auto', min_adj_ch=0):
    """Permutation cluster test on array data.

    Parameters
    ----------
    data : np.ndarray | list of np.ndarray
        An array where first two dimensions are ``conditions x observations``
        or list of arrays where each array has observations in the first
        dimension. If the data contains channels it should be in the dimension
        immediately after observations.
    adjacency : 2d boolean array | None
        Array that denotes adjacency between channels (or vertices). If
        ``None`` it is assumed that no channels/vertices are present.
    stat_fun : function | None
        Statistical function to use. It should allow as many arguments as
        conditions and should return one array of computed statistics.
    threshold : float | None
        Cluster entry threshold for the test statistic. If ``None`` (default)
        the ``p_threshold`` argument is used.
    p_threshold : float
        P value threshold to use in cluster entry threshold computation. For
        standard parametric tests (t test, ANOVA) it is computed from
        theoretical test distribution; if ``n_stat_permutations`` is above zero
        the threshold is obtained from percentile of permutation distribution.
    paired : bool
        Whether the permutations should be conducted for paired samples
        scenario (randomization of condition orders within observations).
        Currently the condition orders are randomized even if they are the same
        for all subjects. This argument is also used to automatically pick
        a statistical test if ``stat_fun`` is ``None``.
    one_sample : bool
        Whether the permutations should be conducted for a one sample scenario
        (sign flipping randomization). This argument is also used to
        automatically pick a statistical test if ``stat_fun`` is ``None``.
    tail : str
        Which differences to test. ``'both'`` tests positive and negative
        effects, while ``'pos'`` - only positive.
        NEG is not implemented!
    n_permutations : int
        Number of cluster based permutations to perform. Defaults to ``1000``.
    n_stat_permutations : int
        Whether to compute ``threshold`` using permutations (this is separate
        from cluster-based permutations when the computed thresholds are used).
        If ``n_stat_permutations > 0`` then this many permutations will be used
        to compute statistical cluster-entry thresholds. The threshold is set
        to ``p_threshold`` of the computed permutation distribution.
    progress : bool | str | tqdm progressbar
        Whether to show a progressbar (if boolean) or what kind of progressbar
        to show (``'notebook'`` or ``'text'``). Alternatively a progressbar can
        be passed that will be reset and set to a new maximum.
    return_distribution : bool
        Whether to return the distribution of cluster-based permutations.
        If ``True`` a dictionary of positive and negative cluster statistics
        from permutations is returned.
    backend : str
        Clustering backend to use. Can be ``'auto'``, ``'mne'``, ``'borsar'``
        or ``'numpy'``. Depending on the search space, different backend may be
        optimal. Defaults to ``'auto'`` which selects the backend
        automatically.
    min_adj_ch: int
        Minimum number of adjacent in-cluster channels to retain a point in
        the cluster.

    Returns
    -------
    stat : np.ndarray
        Statistical test results in the search space (same dimensions as
        ``data[0]``, apart from the first one representing observations).
    clusters : list of np.ndarray
        List of clusters. Each cluster is a boolean array of cluster
        membership.
    cluster_p : np.ndarray
        P values for each cluster.
    distribution : dict | None
        Dictionary of cluster statistics from permutations. Only returned if
        ``return_distribution`` is ``True``.
    """

    from .utils import progressbar
    from borsar.cluster.label import _get_cluster_fun, find_clusters

    n_groups = len(data)
    if stat_fun is None:
        stat_fun = _find_stat_fun(n_groups, paired, tail)

    if paired or one_sample:
        n_obs = data[0].shape[0]
        signs_size = tuple([n_obs] + [1] * (data[0].ndim - 1))
    else:
        condition = np.concatenate([np.ones(data[idx].shape[0]) * idx
                                    for idx in range(n_groups)])
        data_unr = np.concatenate(data)

    if one_sample:
        signs = np.array([-1, 1])

    pos_dist = np.zeros(n_permutations)
    if tail == 'both':
        neg_dist = np.zeros(n_permutations)

    # test on non-permuted data
    stat = stat_fun(*data)

    # compute threshold from stat, use permutation distribution if
    # n_stat_permutations > 0
    if n_stat_permutations > 0:
        threshold = _compute_threshold_via_permutations(
            data, paired, tail, stat_fun, p_threshold, n_stat_permutations,
            progress=progress)
    else:
        threshold = _compute_threshold(data, threshold, p_threshold,
                                       paired, one_sample)

    # use 3d clustering
    cluster_fun = _get_cluster_fun(stat, adjacency=adjacency,
                                   backend=backend, min_adj_ch=min_adj_ch)

    clusters, cluster_stats = find_clusters(
        stat, threshold, adjacency=adjacency, cluster_fun=cluster_fun,
        min_adj_ch=min_adj_ch)

    if not clusters:
        return stat, clusters, cluster_stats

    if paired and n_groups > 2:
        orders = [np.arange(n_groups)]
        for _ in range(n_groups - 1):
            orders.append(np.roll(orders[-1], shift=-1))
        data_all = np.stack(data, axis=0)

    pbar = progressbar(progress, total=n_permutations)

    # compute permutations
    for perm in range(n_permutations):
        # permute data / predictors
        if one_sample:
            # one-sample sign-flip
            idx = np.random.random_integers(0, 1, size=signs_size)
            perm_signs = signs[idx]
            perm_data = [data[0] * perm_signs]
        elif paired and n_groups == 2:
            # this is analogous to one-sample sign-flip but with paired data
            # (we could also perform one sample t test on condition differences
            #  with sign-flip in the permutation step)
            idx1 = np.random.random_integers(0, 1, size=signs_size)
            idx2 = 1 - idx1
            perm_data = list()
            perm_data.append(data[0] * idx1 + data[1] * idx2)
            perm_data.append(data[0] * idx2 + data[1] * idx1)
        elif paired and n_groups > 2:
            ord_idx = np.random.randint(0, n_groups, size=n_obs)
            perm_data = data_all.copy()
            for obs_idx in range(n_obs):
                this_order = orders[ord_idx[obs_idx]]
                perm_data[:, obs_idx] = data_all[this_order, obs_idx]
        elif not paired:
            this_order = condition.copy()
            np.random.shuffle(this_order)
            perm_data = [data_unr[this_order == idx]
                         for idx in range(n_groups)]

        perm_stat = stat_fun(*perm_data)

        _, perm_cluster_stats = find_clusters(
            perm_stat, threshold, adjacency=adjacency, cluster_fun=cluster_fun,
            min_adj_ch=min_adj_ch)

        # if any clusters were found - add max statistic
        if len(perm_cluster_stats) > 0:
            max_val = perm_cluster_stats.max()

            if max_val > 0:
                pos_dist[perm] = max_val

            if tail in ['both', 'neg']:
                min_val = perm_cluster_stats.min()
                if min_val < 0:
                    neg_dist[perm] = min_val

        if progressbar:
            pbar.update(1)

    # compute permutation probability
    # TODO - fix, when we want only 'pos' or 'neg' but use for example t test
    cluster_p = np.array([(pos_dist > cluster_stat).mean() if cluster_stat > 0
                          else (neg_dist < cluster_stat).mean()
                          for cluster_stat in cluster_stats])
    if tail == 'both':
        cluster_p *= 2  # because we use two-tail
        cluster_p[cluster_p > 1.] = 1.  # probability has to be <= 1.

    # sort clusters by p value
    cluster_order = np.argsort(cluster_p)
    cluster_p = cluster_p[cluster_order]
    clusters = [clusters[i] for i in cluster_order]

    if return_distribution:
        return stat, clusters, cluster_p, dict(pos=pos_dist, neg=neg_dist)
    else:
        return stat, clusters, cluster_p


def _compute_threshold(data, threshold, p_threshold, paired,
                       one_sample):
    '''Find significance threshold analytically.'''
    if threshold is None:
        from scipy.stats import distributions
        n_groups = len(data)
        n_obs = [len(x) for x in data]

        if n_groups < 3:
            len1 = len(data[0])
            len2 = len(data[1]) if (len(data) > 1 and data[1] is not None) else 0
            df = (len1 - 1 if paired or one_sample else len1 + len2 - 2)
            threshold = distributions.t.ppf(1 - p_threshold / 2., df=df)
        else:
            # ANOVA F
            n_obs = data[0].shape[0] if paired else sum(n_obs)
            dfn = n_groups - 1
            dfd = n_obs - n_groups
            threshold = distributions.f.ppf(1. - p_threshold, dfn, dfd)
    return threshold


def _find_stat_fun(n_groups, paired, tail):
    '''Find relevant stat_fun given ``n_groups``, ``paired`` and ``tail``.'''
    if n_groups > 2 and tail == 'both':
        raise ValueError('Number of compared groups is > 2, but tail is set'
                         ' to "both". If you want to use ANOVA, set tail to'
                         ' "pos".')
    if n_groups > 2 and not tail == 'both':
        if paired:
            # repeated measures ANOVA
            return rm_anova_stat_fun
        else:
            from scipy.stats import f_oneway

            def stat_fun(*args):
                fval, _ = f_oneway(*args)
                return fval
            return stat_fun
    else:
        if paired:
            from scipy.stats import ttest_rel

            def stat_fun(*args):
                tval, _ = ttest_rel(*args)
                return tval
            return stat_fun
        else:
            from scipy.stats import ttest_ind

            def stat_fun(*args):
                tval, _ = ttest_ind(*args, equal_var=False)
                return tval
            return stat_fun


def rm_anova_stat_fun(*args):
    '''Stat fun that does one-way repeated measures ANOVA.'''
    from mne.stats import f_mway_rm

    data = np.stack(args, axis=1)
    n_factors = data.shape[1]

    fval, _ = f_mway_rm(data, factor_levels=[n_factors],
                        return_pvals=False)

    if data.ndim > 3:
        fval = fval.reshape(data.shape[2:])
    return fval


# FIXME: streamline/simplify permutation reshaping and transposing
# FIXME: time and see whether a different solution is better
# FIXME: splitting across jobs could be smarter (chunks of permutations, not
# one by one)
def _compute_threshold_via_permutations(data, paired, tail, stat_fun,
                                        p_threshold=0.05, n_permutations=1000,
                                        progress=True,
                                        return_distribution=False,
                                        n_jobs=1):
    '''
    Compute significance thresholds using permutations.

    Assumes ``n_conditions x n_observations x ...`` data array.
    Note that the permutations are implemented via shuffling of the condition
    labels, not randomization of independent condition orders.
    '''
    from .utils import progressbar

    if paired:
        # concatenate condition dimension if needed
        if isinstance(data, (list, tuple)):
            data = np.stack(data, axis=0)

        dims = np.arange(data.ndim)
        dims[:2] = [1, 0]
        n_cond, n_obs = data.shape[:2]
        data_unr = data.transpose(*dims).reshape(n_cond * n_obs,
                                    *data.shape[2:])

        # compute permutations of the stat
        if n_jobs == 1:
            stats = np.zeros(shape=(n_permutations, *data.shape[2:]))
            pbar = progressbar(progress, total=n_permutations)
            for perm_idx in range(n_permutations):
                stats[perm_idx] = _paired_perm(
                    data_unr, stat_fun, n_cond, n_obs, dims, pbar=pbar
                )
        else:
            from joblib import Parallel, delayed
            stats = Parallel(n_jobs=n_jobs)(
                delayed(_paired_perm)(data_unr, stat_fun, n_cond, n_obs, dims)
                for perm_idx in range(n_permutations)
            )
            stats = np.stack(stats, axis=0)
    else:
        n_cond = len(data)
        condition = np.concatenate([np.ones(data[idx].shape[0]) * idx
                                    for idx in range(n_cond)])
        data_unr = np.concatenate(data)

        if n_jobs == 1:
            stats = np.zeros(shape=(n_permutations, *data[0].shape[1:]))
            pbar = progressbar(progress, total=n_permutations)
            for perm_idx in range(n_permutations):
                stats[perm_idx] = _unpaired_perm(
                    data_unr, stat_fun, condition, n_cond, pbar=pbar
                )
        else:
            from joblib import Parallel, delayed
            stats = Parallel(n_jobs=n_jobs)(
                delayed(_unpaired_perm)(data_unr, stat_fun, condition, n_cond)
                for perm_idx in range(n_permutations)
            )
            stats = np.stack(stats, axis=0)

    # now check threshold
    if tail == 'pos':
        percentile = 100 - p_threshold * 100
        threshold = np.percentile(stats, percentile, axis=0)
    elif tail == 'neg':
        percentile = p_threshold * 100
        threshold = np.percentile(stats, percentile, axis=0)
    elif tail == 'both':
        percentile_neg = p_threshold / 2 * 100
        percentile_pos = 100 - p_threshold / 2 * 100
        threshold = [np.percentile(stats, perc, axis=0)
                     for perc in [percentile_pos, percentile_neg]]
    else:
        raise ValueError(f'Unrecognized tail "{tail}"')

    if not return_distribution:
        return threshold
    else:
        return threshold, stats


def _paired_perm(data_unr, stat_fun, n_cond, n_obs, dims, pbar=None):
    rnd = (np.random.random(size=(n_cond, n_obs))).argsort(axis=0)
    idx = (rnd + np.arange(n_obs)[None, :] * n_cond).T.ravel()
    this_data = data_unr[idx].reshape(
        n_obs, n_cond, *dims[2:]).transpose(*dims[:2])
    stat = stat_fun(*this_data)

    if pbar is not None:
        pbar.update(1)

    return stat


def _unpaired_perm(data_unr, stat_fun, condition, n_cond, pbar=None):
    rnd = condition.copy()
    np.random.shuffle(rnd)
    this_data = [data_unr[rnd == idx] for idx in range(n_cond)]
    stat = stat_fun(*this_data)

    if pbar is not None:
        pbar.update(1)

    return stat
