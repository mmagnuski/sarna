# this file is just to easily get most common imports at once
# by writing: `from mypy.init import *`

import os
import os.path as op
from os.path import join
from glob import glob

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

try:
  from showit import image
except ImportError:
  pass
# import seaborn as sns

from mypy import dB, whos, find_index

def pwd():
    return os.getcwd()
cwd = pwd
