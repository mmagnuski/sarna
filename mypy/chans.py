import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from .utils import time_range, mne_types, get_chan_pos



def correct_egi_channel_names(eeg):
    # define function correcting channel names:
    def corr_ch_names(name):
        if name.startswith('EEG'):
            if name == 'EEG 065':
                return 'Cz'
            else:
                return 'E' + str(int(name[-3:]))
        else:
            return name
    # change channel names
    eeg.rename_channels(corr_ch_names)


# TODO: generalize to Evoked (maybe Raw...)
def z_score_channels(eeg):
    tps = mne_types()
    from mne.utils import _get_inst_data
    assert isinstance(eeg, tps['epochs'])

    data = _get_inst_data(eeg)
    n_epoch, n_chan, n_sample = data.shape
    data = data.transpose((1, 0, 2)).reshape((n_chan, n_epoch * n_sample))
    eeg._data = zscore(data, axis=1).reshape(
        (n_chan, n_epoch, n_sample)).transpose((1, 0, 2))
    return eeg


def select_channels(chan_vals, N=5, connectivity=None,
                    select_clusters=None):
    '''select N indices (channels) that maximize mean of chan_vals.

    Parameters
    ----------
    chan_vals : numpy array of shape (n_channels, )
        One dimensional array of channel values.
    N : int
        Number of channels to select.
    connectivity : boolean numpy array of shape (n_channels, n_channels)
        Channel adjacency matrix. If passed then channel groups are selected
        with adjacency constraint.
    select_clusters : numpy array of shape (n_channels, )
        Additional values to bias cluster selection. `select_clusters` is used
        only when connectivity was passed and initial channel selection
        returned more than one group of adjacent channels.
    '''
    from . import cluster

    # step 1, select N channels
    n_channels = chan_vals.shape[0]
    presel_chans = chan_vals.argsort()[-N:]

    # if no connectivity - nothing else to do
    if connectivity is None:
        return presel_chans
        # raise TypeError('Connectivity was not specified...')

    # step 2, cluster channels
    if_chose = np.zeros(n_channels, dtype='bool')
    if_chose[presel_chans] = True
    clst, _ = cluster.cluster_1d(if_chose, connectivity)

    # step 3, additional cluster selection after clustering
    if select_clusters is not None:
        get_cluster = np.array([select_clusters[c].mean()
                               for c in clst]).argmax()
    else:
        get_cluster = np.array([chan_vals[c].mean()
                               for c in clst]).argmax()
    clst = clst[get_cluster]

    # step 4, add channels to fill up the cluster up to N chans
    n_cluster_chans = clst.shape[0]
    need_chans = N - n_cluster_chans
    while need_chans > 0:
        clst_neighb = connectivity[clst, :].copy().any(axis=0)
        clst_neighb[clst] = False
        clst_neighb = np.where(clst_neighb)[0]
        take_n_chans = min(need_chans, clst_neighb.shape[0])
        add_chans = chan_vals[clst_neighb].argsort()[-take_n_chans:]
        add_chans = clst_neighb[add_chans]
        clst = np.hstack([clst, add_chans])
        need_chans = N - clst.shape[0]
    return clst


def find_channels(inst, names):
    one_name = False
    if isinstance(names, str):
        one_name = True
    finder = (lambda val: inst.ch_names.index(val)
              if val in inst.ch_names else None)
    return finder(names) if one_name else list(map(finder, names))


def asymmetry_pairs(ch_names, inst=None):
    '''construct asymetry channel pairs based on names.

    Parameters
    ----------
    ch_names : list of str
        List of channel names.
    inst : mne object instance (optional)
        Mne object like mne.Raw or mne.Epochs

    Returns
    -------
    asym_chans_idx: dict of str -> list of int mappings
        Dictionary mapping hemisphere to list of channel indices. Indices are
        with respect to mne object instance ch_names if `inst` was passed,
        otherwise the indices are with respect to `ch_names`
    asym_chans: dict of str -> list of str mappings
        Dictionary mapping hemisphere to list of channel names.
    '''

    frontal_asym = [ch for ch in ch_names if 'z' not in ch]
    labels = ['right', 'left']
    asym_chans = {l: list() for l in labels}

    for ch in frontal_asym:
        chan_base = ch[:-1]
        chan_value = int(ch[-1])

        if (chan_value % 2) == 1:
            asym_chans['left'].append(ch)
            asym_chans['right'].append(chan_base + str(chan_value + 1))

    if inst is not None:
        asym_chans_idx = {k: [inst.ch_names.index(ch) for ch in asym_chans[k]]
                          for k in asym_chans.keys()}
    else:
        asym_chans_idx = {k: [ch_names.index(ch) for ch in asym_chans[k]]
                          for k in asym_chans.keys()}

    return asym_chans_idx, asym_chans


modes = dict(N170=[(0.145, 0.21), 5, 'min'],
             P100=[(0.075, 0.125), 5, 'max'],
             P300=[(0.3, 0.5), 5, 'maxmean'])

# TODO:
# - [ ] simplify peak selection
# - [ ] move different info checks to separate function, possibly even
#       to mypy.viz.Topo
# - [ ] maybe add an option to fix latency from fit to transform
#       (fix_latency=True or fix_peak_pos=True)
# - [ ] simplify the way channel indices and names are handled (some
#       of that can be in separate functions):
#       channel_names, channel_inds, get_channels('names')?
# - [ ] make less erp-peak dependent - support other data types (freq)?
class Peakachu(object):
    '''Find peaks, select channels... FIXME
    
    Parameters
    ----------
    mode: str
        String that describes peak characteristics to look for.
        For example `N170` means that negative peak in 145 - 210 ms
        is looked for. These options can be overriden with `time_window`
        and `select` keyword arguments.
        Available modes are: 'P100', 'N170' and 'P300'. Their
        characteristics (time_window, n_channels and select method) can
        be seen in `mypy.chans.modes`.
    time_window: two-element tuple
        (start, end) of the peak search window in seconds. Make sure that
        the window makes sense for your data - peaks outside of this
        window will not be found.
    n_channels: int
        Number of channels maximizing peak strength that are chosen.
        These channels are averaged during `transform` and the peak is
        found in this averaged signal. Defaults to 5.
    connectivity: boolean 2d array
        Channel adjacency matrix. If passed - only adjacent channels
        maximizing peak strength are chosen.
    
    Example:
    --------
    >> p = Peakachu(mode='N170', n_channels=6)
    >> p.fit(grand_erp)
    >> p.plot_topomap()
    >> amplitudes, latencies = p.transform(epochs)
    '''
    def __init__(self, mode=None, time_window=None, n_channels=None,
                 select=None, connectivity=None):
        # from mne.preprocessing.peak_finder import peak_finder

        self.time_window = time_window
        self.n_channels = n_channels
        self.select = select
        self.connectivity = connectivity
        # self._finder = peak_finder

        # check mode
        if mode is not None:
            assert isinstance(mode, str), 'mode must be a string, got {}.' \
                .format(mode)
            assert mode in modes, 'unrecognized mode, must be one of: ' \
                + ', '.join(modes.keys()) + ', got {}'.format(mode)
            mode_opts = modes[mode]
            var_names = ['time_window', 'n_channels', 'select']
            for var, opt in zip(var_names, mode_opts):
                if getattr(self, var) is None:
                    setattr(self, var, opt)

        # fill None vals with defaults if no mode:
        if mode is None:
            self.n_channels = 5 if self.n_channels is None else n_channels

        # check input
        assert isinstance(self.n_channels, int), 'n_channels must be int, ' \
            'got {}'.format(type(n_channels))
        valid_select = ['min', 'max', 'minmean', 'maxmean']
        assert isinstance(self.select, str), 'select must be string, ' \
            'got {}'.format(type(select))
        assert self.select in valid_select, 'unrecognized select, must be one'\
            ' of: ' + ', '.join(valid_select) + ', got {}'.format(select)

        # check connectivity
        self.connectivity = connectivity
        if connectivity is not None:
            if isinstance(connectivity, str):
                from . import cluster
                self.connectivity = cluster.construct_adjacency_matrix(
                    connectivity)
            elif not isinstance(connectivity, np.ndarray):
                raise TypeError('connectivity must be a str or numpy array'
                                ' got {} instead.'.format(type(connectivity)))

    def fit(self, inst):
        '''
        Fit Peakachu to erp data.
        This leads to selection of channels that maximize peak strength.
        These channels are then used during `transform` to search for peak.
        Different data can be passed during `fit` and `transform` - for example
        `fit` could use condition-average while transform can be used on separate
        conditions.
        '''
        from mne.evoked import Evoked
        from mne.io.pick import _pick_data_channels, pick_info
        assert isinstance(inst, (Evoked, np.ndarray)), 'inst must be either' \
            ' Evoked or numpy array, got {}.'.format(type(inst))

        # deal with bad channels and non-data channels
        picks = _pick_data_channels(inst.info)

        self._info = pick_info(inst.info, picks)
        self._all_ch_names = [inst.ch_names[i] for i in picks]

        # get peaks
        peak_val, peak_ind = self._get_peaks(inst, select=picks)

        # select n_channels
        vals = peak_val if 'max' in self.select else -peak_val
        chan_ind = select_channels(vals, N=self.n_channels,
                        connectivity=self.connectivity)
        self._chan_ind = [picks[i] for i in chan_ind]
        self._chan_names = [inst.ch_names[ch] for ch in self._chan_ind]
        self._peak_vals = peak_val
        return self

    def transform(self, inst, average_channels=True):
        '''
        Extract peaks from given data using information on channels that
        maximize peak strength obtained during `fit`. Returns peak amplitude
        and latency. `average_channels` argument can be used to avoid averaging
        channels. If mne.Epochs is passed the peaks are found for single trials
        (which may not be a good idea unless you have very good SNR).
        '''
        from mne.evoked import Evoked
        tps = mne_types()

        assert isinstance(inst, (Evoked, tps['epochs']))
        select = [inst.ch_names.index(ch) for ch in self._chan_names]
        peak_val, peak_ind = self._get_peaks(inst, select=select)

        if 'mean' in self.select:
            peak_times = np.empty(peak_ind.shape)
            peak_times.fill(np.nan)
        else:
            peak_times = inst.times[self._current_time_range][peak_ind]

        if average_channels:
            peak_val = peak_val.mean(axis=0)
            peak_times = peak_times.mean(axis=0)
        return peak_val, peak_times

    def plot_topomap(self, info=None):
        import matplotlib as mpl
        from .viz import Topo

        original_info = False
        remove_level_0 = False
        if info is None:
            original_info = True
            info = self._info

        # if a different info is passed - compare and
        # fill unused channels with 0
        if not original_info:
            # compare given and original info
            ch_num = len(info['ch_names'])
            vals = np.zeros(ch_num)
            overlapping = list()
            orig_inds = list()
            for ch in info['ch_names']:
                has_chan = ch in self._info['ch_names']
                overlapping.append(has_chan)
                if has_chan:
                    orig_inds.append(self._info['ch_names'].index(ch))
            vals[np.array(overlapping)] = self._peak_vals[orig_inds]
            remove_level_0 = np.sum(overlapping) < ch_num
        else:
            vals = self._peak_vals


        # topoplot
        tp = Topo(vals, info, show=False)

        # highligh channels
        mark_ch_inds = [info['ch_names'].index(ch) for ch in self._chan_names
                        if ch in info['ch_names']]
        tp.mark_channels(mark_ch_inds)

        # make all topography lines solid
        tp.solid_lines()

        if remove_level_0:
            tp.remove_level(0.)

        # final touches
        tp.fig.set_facecolor('white')
        return tp.fig

    def _get_peaks(self, inst, select=False):
        from mne.utils import _get_inst_data

        t_rng = time_range(inst, self.time_window)
        self._current_time_range = t_rng

        data = _get_inst_data(inst)
        data_segment = data[..., t_rng]
        if data.ndim == 3:
            # put channels first
            data_segment = data_segment.transpose((1, 0, 2))
            
        if select is True:
            data_segment = data_segment[self._chan_ind, :]
        elif isinstance(select, (list, np.ndarray)):
            data_segment = data_segment[select, :]
        n_channels = data_segment.shape[0]

        if data_segment.ndim == 1:
            data_segment = data_segment[:, np.newaxis]
        if self.select == 'min':
            peak_ind = data_segment.argmin(axis=-1)
        elif self.select == 'max':
            peak_ind = data_segment.argmax(axis=-1)
        elif 'mean' in self.select:
            peak_ind = None
            peak_val = data_segment.mean(axis=-1)

        if self.select in ['min', 'max']:
            if data_segment.ndim == 2:
                peak_val = data_segment[range(n_channels), peak_ind]
            elif data_segment.ndim == 3:
                peak_val = np.zeros(peak_ind.shape)
                for ep in range(peak_ind.shape[1]):
                    peak_val[:, ep] = data_segment[range(n_channels),
                                                      ep, peak_ind[:, ep]]

        return peak_val, peak_ind

    def plot_erp(self, inst):
        tps = mne_types()
        from mne.viz import plot_compare_evokeds

        picks = [inst.ch_names.index(ch) for ch in self._chan_names]
        if isinstance(inst, tps['epochs']):
            erps = {c: inst[c].average() for c in inst.event_id.keys()}
            fig = plot_compare_evokeds(erps, picks=picks)
        else:
            fig = plot_compare_evokeds(inst, picks=picks)

        fig.set_facecolor('white')
        return fig
