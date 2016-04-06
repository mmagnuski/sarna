import numpy as np
import pandas as pd


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
	evt = np.concatenate(evnts, axis=0)

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
	# assert isinstance(mne.Raw) etc?
	eeg = eeg.copy()
	df = din_dataframe(eeg)
	n_evnt = df.shape[0]
	events = np.zeros([n_evnt, 3])
	dins = np.tile(df.columns.values, [n_evnt, 1])
	events[:, 2] = np.sum(df.values * dins, axis=1)
	events[:, 0] = df.index.values 
	eeg.info['events'] = events

	# pick non-din channels:
	non_din_chans = [ch for ch in eeg.info['ch_names']
						if not ch.startswith('D')]
	eeg.pick_channels(non_din_chans)
	return eeg
