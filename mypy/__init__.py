# some imports
import numpy as np

from . import events
from . import proj
from .colors import colors

# import freq
from . import viz


# later move these to utils:
def do_not_warn():
    '''turns off DeprecationWarnings as they can be
    painful in the current jupyter notebook'''
    import warnings
    try:
        from exceptions import DeprecationWarning # py2
    except ImportError:
        global DeprecationWarning
        # from warnings import DeprecationWarning
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*use @default decorator instead.*')
    # warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/IPython/.*')
    # another way, turn only deprecation wornings for a specific package (like mne):
    # warnings.filterwarnings('default', category=DeprecationWarning, module='.*/mypackage/.*')


def rld(pkg):
    '''fast reaload (no need to type much)'''
    import importlib
    importlib.reload(pkg)


# TODO:
# - [ ] should check type and size of the vars (check how mne-python does it)
def whos():
    """Print the local variables in the caller's frame.
    Copied from stack overflow:
    http://stackoverflow.com/questions/6618795/get-locals-from-calling-namespace-in-python"""
    import inspect
    frame = inspect.currentframe()
    try:
        print(frame.f_back.f_locals) # here the locals should be inspected and pretty-printed
    finally:
        del frame


def find_index(vec, vals):
    if not isinstance(vals, list):
        vals = [vals]
    return [np.abs(vec - x).argmin() for x in vals]


def find_range(vec, ranges):
    '''
    Parameters
    ----------
    vec : numpy array
        Vector of sorted values.
    ranges: list of tuples/lists or two-element list/tuple
    '''
    assert isinstance(ranges, (list, tuple))
    assert len(ranges) > 0
    one_in = False
    if not isinstance(ranges[0], (list, tuple)) and len(ranges) == 2:
        one_in = True
        ranges = [ranges]

    slices = list()
    for rng in ranges:
        start, stop = [np.abs(vec - x).argmin() for x in rng]
        slices.append(slice(start, stop + 1)) # including last index
    if one_in:
        slices = slices[0]
    return slices


# join inds
def group(vec):
    in_grp = False
    group_lims = list()
    vec = np.append(vec, False)
    for ii, el in enumerate(vec):
        if in_grp and not el:
            in_grp = False
            group_lims.append([start_ind, ii-1])
        elif not in_grp and el:
            in_grp = True
            start_ind = ii
    return np.array(group_lims)
