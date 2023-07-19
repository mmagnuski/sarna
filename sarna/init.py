# this file is just to easily get most common imports at once
# by writing: `from mypy.init import *

import os
import sys
import os.path as op
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

from scipy.io import loadmat, savemat
from scipy import stats, signal

try:
    import xarray as xr
except ImportError:
    print('Could not import xarray')

# from scipy.stats import distributions as dist
# from scipy import fftpack
# import seaborn as sns?

from .utils import find_index, find_range
