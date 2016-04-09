import os
import numpy as np
import pandas as pd
from matplotlib.color import hex2color 


# TODOs:
# - [ ] finish color function
# - [ ] maybe a sep fun - colors for list input?


# load colors
# -----------
file_path = os.path.split(__file__)[0]
all_colors = dict()
searchlist = ["wiki pl"]

# wiki pl
columns = ['colname', 'hex'] + list('rgb') + ['description', 'addinfo']
wiki_pl_fullpath = os.path.join(file_path, 'colors_pl.txt')
wiki_pl = pd.read_table(wiki_pl_fullpath, names=columns)
all_colors['wiki pl'] = {'names': list(wiki_pl.colname), 
					'rgb': wiki_pl.loc[:, list('rgb')].values}

# seaborn colors
try:
	import seabron.apionly as sns
	all_colors['xkcd'] = reformat_dict(sns.xkcd_rgb.xkcd_rgb)
	all_colors['crayon'] = reformat_dict(sns.crayon.crayon)
	searchlist += ["xkcd", "crayon"]
except ImportError:
	pass


def color(colname, pername=1):
	# for list -> go through each colname and
	#             select one color (best) matching
	# for char -> return (optionally) multiple colors
	# if isinstance(colname, list):
	pass


def fuzzy_match(s1, s2):
	last_char = 0
	for ch in s1:
		last_char = s2.find(ch, last_char)
		if last_char < 0:
			return False
	return True


def fuzzy_match_score(s1, s2):
	last_char = 0
	char_ind = list()
	for ch in s1:
		last_char = s2.find(ch, last_char)
		char_ind.append(last_char)
		if last_char < 0:
			return 0.
	# score
	ind_score = np.arange(len(s2), 0, -1)
	ind_score /= ind_score.sum()
	return ind_score[char_ind].sum()


def reformat_dict(d):
	keys = d.keys()
	names = list()
	rgb = list()
	for k in keys:
		names.append(k)
		rgb.append(hex2color(d[k]))
	return {'names': names, 'rgb': np.array(rgb)}
