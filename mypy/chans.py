import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from .utils import time_range, mne_types



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



modes = dict(N170=[(0.145, 0.21), 5, 'min'],
             P100=[(0.075, 0.125), 5, 'max'],
             P300=[(0.3, 0.5), 5, 'maxmean'])

# TODO:
# - [ ] make less erp-peak dependent - support other data types (freq)
# fit (add option to fix latency)
# transform # average=True, average_channels?
# channel_names, channel_inds ? get_channels('names')
class Peakachu(object):
    '''Find peaks, select channels...

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
        from mne.evoked import Evoked
        assert isinstance(inst, (Evoked, np.ndarray)), 'inst must be either' \
            ' Evoked or numpy array, got {}.'.format(type(inst))

        self._info = inst.info
        self._all_ch_names = inst.ch_names

        # get peaks
        peak_val, peak_ind = self._get_peaks(inst)

        # select n_channels
        vals = peak_val if 'max' in self.select else -peak_val
        chan_ind = select_channels(vals, N=self.n_channels,
                        connectivity=self.connectivity)
        self._chan_ind = chan_ind
        self._chan_names = [inst.ch_names[ch] for ch in chan_ind]
        self._peak_vals = peak_val
        return self

    def transform(self, inst, average_channels=True):
        from mne.evoked import Evoked
        tps = mne_types()

        assert isinstance(inst, (Evoked, tps['epochs']))
        peak_val, peak_ind = self._get_peaks(inst, select=True)

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
        from mne.viz import plot_topomap

        if info is None:
            info = self._info

        fig, ax = plt.subplots()
        axim, _ = plot_topomap(self._peak_vals, info)

        # TODO: move the code below to mypy
        #       (something like a Topomap object or just mark_topomap_channels)
        # highligh channels
        chans = fig.axes[0].findobj(mpl.patches.Circle)
        for ch in self._chan_ind:
            chans[ch].set_color('white')
            chans[ch].set_radius(0.01)
            chans[ch].set_zorder(4)

        # make all topography lines solid
        chld = fig.axes[0].get_children()
        topo_lines = chld[:6] # change into finding obj elements
        for l in topo_lines:
            l.set_linestyle('-')

        # final touches
        fig = plt.gcf()
        fig.set_facecolor('white')
        return fig

    def _get_peaks(self, inst, select=False):
        from mne.utils import _get_inst_data

        t_rng = time_range(inst, self.time_window)
        self._current_time_range = t_rng

        data = _get_inst_data(inst)
        if data.ndim == 3:
            # channels first
            data_segment = data[:, :, t_rng].transpose((1, 0, 2))
        else:
            data_segment = data[:, t_rng]

        if select:
            data_segment = data_segment[self._chan_ind, :]
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
