import numpy as np
import pandas as pd


# TODOs:
# - [ ] correct_egi_channel_names could live in a separate module
# - [ ] create_middle_events should be made more universal


def din_dataframe(eeg):
	'''
	Turns DIN1, DIN2, ... channels of an egi file
	into a dataframe with latency as index and
	DIN numbers as columns. Each row of the dataframe is
	therefore a reconstruction of the LPT digital signal

	arguments
	---------
	eeg - mne Raw created with `read_raw_egi`

	output
	------
	df - dataframe with latency as index and DIN numbers as
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


def din2event(eeg):
	'''Turn DIN1, DIN2, ... etc. event channels
	to mne event array.
	'''
	# assert isinstance(mne.Raw) etc?
	df = din_dataframe(eeg)
	n_evnt = df.shape[0]
	events = np.zeros([n_evnt, 3], dtype='int')
	dins = np.tile(df.columns.values, [n_evnt, 1])
	events[:, 2] = np.sum(df.values * dins, axis=1)
	events[:, 0] = df.index.values 
	eeg.info['events'] = events

	# pick non-din channels:
	non_din_chans = [ch for ch in eeg.info['ch_names']
						if not ch.startswith('D')
						and not ch == 'STI 014']
	eeg.pick_channels(non_din_chans)
	return eeg


def reject_events_in_bad_segments(events, bad_segments, around_event=(-10,10), remove_types=None):
    '''removes events that coincide with bad segments

    parameters
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

    returns
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
