import numpy as np
import pandas as pd
from scipy.io import loadmat

from mypy.utils import group, extend_slice, find_range

# TODOs:
# - [ ] create_middle_events should be made more universal


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


def reject_events_in_bad_segments(events, bad_segments, around_event=(-10,10),
								  remove_types=None):
	'''Remove events that coincide with bad segments.

	Parameters
	----------
	events 		 - mne events array
	bad_segments - N by 2 matrix with bad segment info
				   each row is a bad segment, first column is the
				   bad segment onset and the second column is the
				   bad segment offset (in samples)
	around_event - (lowerlim, higherlim) box limits around
				   each event. If a bad segment overlaps
				   with the box, the event is rejected.
				   Box limits are in samples.
				   (-500, 1000) means 500 samples before
				   up to 1000 samples after, but one can
				   create box that does not contain the
				   event like (250, 500)
	remove_types - list or numpy array of event types (int)
				   that should be checked for removal

	Returns
	-------
	events - corrected events array
	'''
	if remove_types is None:
		test_events = np.arange(events.shape[0])
	else:
		test_events = np.vstack([events[:,2] == x for x in remove_types])
		test_events = np.where(np.any(test_events, axis=0))

	ev = events[test_events, 0]
	ev = np.vstack([ev + x for x in around_event]).T
	remove = np.zeros(ev.shape[0], dtype='bool')
	for ii in range(ev.shape[0]):
		if np.any(np.logical_and(ev[ii,0] >= bad_segments[:,0],
			ev[ii,0] < bad_segments[:,1])):
			remove[ii] = True
			continue
		if np.any(np.logical_and(ev[ii,1] > bad_segments[:,0],
			ev[ii,1] <= bad_segments[:,1])):
			remove[ii] = True
			continue
		if np.any(np.logical_and(bad_segments[:,0] >= ev[ii,0],
			bad_segments[:,0] < ev[ii,1])):
			remove[ii] = True
	# remove events
	remove_ind = test_events[remove]
	if remove_ind.shape[0] > 0:
		events = np.delete(events, remove_ind, axis=0)
	return events


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


# - [ ] make sure this is universal
def create_middle_events(events, min_time=14., sfreq=250):
	'''Put raven events in the trial centre,
	deleting events that begin trials shorter than
	min_time.'''
	ep_start = np.logical_or(events[:,2] == 1, events[:,2] == 0)
	ep_end = np.logical_or(events[:,2] == 10, events[:,2] == 11)
	ep_len = events[ep_end, 0] - events[ep_start, 0]
	long_enough = ep_len > int(min_time * sfreq)
	new_events = events[ep_start,:]
	new_events[:,0] += np.round(ep_len / 2).astype('int')
	return new_events[long_enough, :]


# TODO:
# - [ ] adapt this to use numpy structured datasets (?)
# - [ ] currently reads set events, should check epoch field...
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
					 (col in take_fields or col in ignore_fields)])
	return df.loc[:, take_fields]


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
