import numpy as np
import pandas as pd
from scipy.io import loadmat

from borsar.utils import find_range
from .utils import group, extend_slice


def din_dataframe(eeg):
	'''
	Turns DIN1, DIN2, ... channels of an egi file
	into a dataframe with latency as index and
	DIN numbers as columns. Each row of the dataframe is
	therefore a reconstruction of the LPT digital signal

	Parameters
	----------
	eeg - mne Raw created with `read_raw_egi`

	Returns
	-------
	df : pandas DataFrame
		Dataframe with latency as index and DIN numbers as
		columns. Values of the dataframe are booleans.
		If df.loc[120, 4] is True for example - then at latency
		120 DIN4 is active.
		df.loc[250, :] on the other hand is the full binary re-
		presentation of the LPT signal at latency 250.
	'''

	 # find DIN-channels
	din_chans = [(ch, i) for i, ch in enumerate(eeg.info['ch_names'])
				 if ch.startswith('D')]

	# create a mapping: din channel -> latencies
	lat = dict()
	for ch, i in din_chans:
		digits = "".join([c for c in ch if c.isdigit()])
		lat[int(digits)] = np.where(eeg._data[i,:] > 0)[0]

	# regroup into an array:
	din_types = np.array(sorted(lat.keys()))
	evt = [[[l, t] for l in lat[t]] for t in din_types]
	evt = np.concatenate(evt, axis=0)

	# sort and add fake row to unify loop
	ind = evt[:,0].argsort(axis=0)
	evt = evt[ind,:]
	evt = np.concatenate([evt, evt[-1,:][np.newaxis, :]+100])

	# find events occuring in the same time sample
	current_row, current_latency = 0, evt[0,0]
	start_seq = 0
	latency = list()
	ifdinactive = np.zeros([evt.shape[0], len(din_types)],
		dtype='bool')

	for r in range(1, evt.shape[0]):
		if not evt[r,0] == current_latency:
			din_ind = [np.where(din_types == x)[0][0]
						for x in evt[start_seq:r, 1]]
			ifdinactive[current_row, din_ind] = True
			latency.append(current_latency)

			current_latency = evt[r,0]
			current_row += 1
			start_seq = r

	# return as a dataframe
	return pd.DataFrame(index=latency, columns=din_types,
						data=ifdinactive[:current_row, :])


def get_events_from_din(eeg):
	'''Turn DIN1, DIN2, ... etc. event channels to mne-python event array.

	Parameters
	----------
	eeg : Raw
		Instance of mne-python raw file.

	Returns
	-------
	events : array
		Numpy array of size (n_events, 3). Complies to mne-python
		convention of event arrays.
	'''
	# assert isinstance(mne.Raw) etc?
	df = din_dataframe(eeg)
	n_evnt = df.shape[0]
	events = np.zeros([n_evnt, 3], dtype='int')
	dins = np.tile(df.columns.values, [n_evnt, 1])
	events[:, 2] = np.sum(df.values * dins, axis=1)
	events[:, 0] = df.index.values
	return events


def remove_din_channels(eeg):
	'''Remove channels starting with 'D' like 'DIN2' or 'DI16'.'''
	# pick non-din channels:
	non_din_chans = [ch for ch in eeg.info['ch_names']
						if not ch.startswith('D')
						and not ch == 'STI 014']
	eeg.pick_channels(non_din_chans)


def extend_bads(rej_win, extend, copy=True):
	'''Extend rejection periods by specfied number of samples.'''
	if copy:
		rej_win = rej_win.copy()
	n_samples = rej_win.shape[1]
	bad_inds = np.any(rej_win, axis=1).nonzero()[0]
	for bad in bad_inds:
		slices = group(rej_win[bad, :], return_slice=True)
		for slc in slices:
			slc = extend_slice(slc, extend, n_samples)
			rej_win[bad, slc] = True
	return rej_win


def apply_artifacts_to_tfr(tfr, artif, orig_time, fillval=np.nan):
	'''Fill tfr data with `fillval` according to `artif`.
	Operates in-place.'''
	n_items, n_samples = artif.shape
	bad_inds = np.any(artif, axis=1).nonzero()[0]
	for bad in bad_inds:
		limits = group(artif[bad, :])
		for lim in limits:
			slc = find_range(tfr.times, list(orig_time[lim]))
			tfr.data[bad, :, :, slc] = fillval
	return tfr


# TODO:
# - [ ] adapt this to use numpy structured datasets (?)
def read_set_events(filename, ignore_fields=None):
	'''Open set file, read epoch events and turn them into a dataframe.

	Parameters
	----------
	filename: str
		Name of the set file to read (can be absolute or relative path)
	ignore_fields: list of str | None
		Epoch event fields to ignore (these fields are note included in the df)

	Returns
	-------
	df: pandas.DataFrame
		Events read into dataframe
	'''
	EEG = loadmat(filename, uint16_codec='latin1',
				  struct_as_record=False, squeeze_me=True)['EEG']
	flds = [f for f in dir(EEG.event[0]) if not f.startswith('_')]
	events = EEG.event
	df_dict = dict()
	for f in flds:
		df = df_dict[f] = [ev.__getattribute__(f) for ev in events]
	df = pd.DataFrame(df_dict)

	# reorder columns:
	take_fields = ['epoch', 'type']
	ignore_fields = list() if ignore_fields is None else ignore_fields
	take_fields.extend([col for col in df.columns if not
					   col in take_fields or col in ignore_fields])
	return df.loc[:, take_fields]


def read_rej(fname, sfreq, bad_types=['reject']):
	from mne import Annotations

	rej = pd.read_table(fname)
	rej.loc[:, ['start', 'end']] = rej.loc[:, ['start', 'end']] / sfreq
	onset, duration = rej.start.values, (rej.end - rej.start).values
	description = ['BAD ' + val if val in bad_types else val
				   for val in rej.type.tolist()]
	return Annotations(onset, duration, description)


class AutoMark(object):
	def __init__(self):
		self.window = None
		self.step = None
		self.ranges = None
		self.variances = None

	def calculate(self, raw, window=1., step=0.25, minmax=False, variance=False,
				  reduction='mean', progressbar=True):
		if not minmax and not variance:
			raise Warning('Nothing computed. To compute variance you need to'
						  ' pass `variance=True`, to compute range you need to'
						  ' pass `minmax=True`.')
			return self

		data = raw._data
		window = int(round(window * raw.info['sfreq']))
		step = int(round(step * raw.info['sfreq']))
		self.window = window
		self.step = step
		self.sfreq = raw.info['sfreq']

		n_samples = data.shape[1]
		n_windows = int(np.floor((n_samples - window) / step))
		self.ranges = np.zeros(n_windows) if minmax else None
		self.variances = np.zeros(n_windows) if variance else None

		reduction = dict(mean=np.mean, max=np.max)[reduction]

		if progressbar:
			from tqdm import tqdm_notebook
			pbar = tqdm_notebook(total=n_windows)

		# step through data
		for window_idx in range(n_windows):
			first = window_idx * step
			last = first + window
			data_buffer = data[:, first:last]

			if minmax:
				self.ranges[window_idx] = reduction(
					data_buffer.max(axis=1) - data_buffer.min(axis=1))
			if variance:
				self.variances[window_idx] = reduction(data_buffer.var(axis=1))
			if progressbar:
				pbar.update(1)
		return self

	def reject(self, max_range=23e-5, max_variance=23e-6):
		if self.ranges is not None:
			is_bad_range = self.ranges > max_range
			# turn to annotations
			annot = self._to_annotation(is_bad_range, description='BAD_range')

		if self.variances is not None:
			is_bad_variance = self.variances > max_variance
			# turn to annotations
			if self.ranges is not None:
				annot = annot + self._to_annotation(is_bad_variance,
													description='BAD_variance')
			else:
				annot = self._to_annotation(is_bad_variance,
											description='BAD_variance')
		return annot

	def _to_annotation(self, bad_bool, description='BAD_'):
		import mne
		groups = group(bad_bool)
		if len(groups) > 0:
			onset = groups[:, 0] * (self.step / self.sfreq)
			duration = (self.window / self.sfreq) + np.diff(groups).ravel() * (self.step / self.sfreq)
			return mne.Annotations(onset, duration, [description] * groups.shape[0])
		else:
			return mne.Annotations([], [], [])


# - [ ] develop this function a little better
# - [ ] check what is used by raw.reject_bads() when no inds given
def mark_reject_peak2peak(raw, reject={'eeg': 23e-5}, window_length=1.,
						  label='bad p2p'):
	import mne
	from mne.utils import _reject_data_segments
	_, inds = _reject_data_segments(raw._data, reject, {'eeg': 0.},
									window_length, raw.info, 0.5)

	# turn inds to time, join
	time_segments = np.array(inds) / raw.info['sfreq']
	time_segments = join_segments(time_segments)

	segment_duration = np.diff(time_segments, axis=-1).ravel()
	return mne.Annotations(time_segments[:, 0], segment_duration, label)


# - [ ] maybe move to utils
def join_segments(time_segments):
	from mypy.utils import group

	# check which should be joined
	join_segments = (time_segments[:-1, 1] - time_segments[1:, 0]) == 0
	segment_groups = group(join_segments)
	segment_groups[:, 1] += 1

	# join segments
	final_segments = np.vstack([time_segments[segment_groups[:, 0], 0],
	                            time_segments[segment_groups[:, 1], 1]]).T

	# we missed single-segments, search for them
	prev = 0
	missed_segments = list()
	for row in segment_groups:
		if not row[0] == prev:
			missed_segments.extend(list(range(prev + 1, row[0])))
		prev = row[1]
	if prev < len(time_segments):
		missed_segments.extend(list(range(prev + 1, len(time_segments))))

	# append missed single segments to multi-segments and sort
	final_segments = np.vstack(
		[final_segments, time_segments[missed_segments, :]])
	final_segments = final_segments[final_segments[:, 0].argsort(), :]

	return final_segments


def align_events(events1, events2):
	'''Return indices for each of the two event trains that give best match.'''
	common_events = list(set(events1).intersection(set(events2)))
	ind1, ind2 = [np.where(np.in1d(events, common_events))[0]
		      for events in [events1, events2]]

	len1, len2 = len(ind1), len(ind2)
	if len1 == len2:
		return ind1, ind2
	elif len1 > len2:
		longer, shorter = events1[ind1], events2[ind2]
	else:
		longer, shorter = events2[ind2], events1[ind1]

	# roll shorter along longer
	lng_len, shrt_len = len(longer), len(shorter)
	n_steps = lng_len - shrt_len + 1
	scores = np.zeros(n_steps, dtype='int')
	for offset in range(n_steps):
		scores[offset] = np.sum(longer[offset:offset + shrt_len] == shorter)
	best_offset = scores.argmax()

	if len1 > len2:
		ind1 = ind1[best_offset:best_offset + shrt_len]
	else:
		ind2 = ind2[best_offset:best_offset + shrt_len]
	return ind1, ind2
