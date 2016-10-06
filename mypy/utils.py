import numpy as np


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


def time_range(inst, time_window):
    return find_range(inst.times, time_window)


# join inds
# TODO:
# - [ ] diff mode
# - [ ] option to return slice
def group(vec, diff=False, return_slice=False):
    in_grp = False
    group_lims = list()
    if diff:
        vec = np.append(vec, np.max(vec) + 1)
        vec = np.diff(vec) > 0
    else:
        vec = np.append(vec, False)

    # group
    for ii, el in enumerate(vec):
        if in_grp and not el:
            in_grp = False
            group_lims.append([start_ind, ii-1])
        elif not in_grp and el:
            in_grp = True
            start_ind = ii
    grp = np.array(group_lims)

    # format output
    if diff:
        grp[:, 1] += 1
    if return_slice:
        slc = list()
        for start, stop in grp:
            slc.append(slice(start, stop + 1))
        return slc
    else:
        return grp


def subselect_keys(key, mapping, sep='/'):
    '''select keys with subselection by a separator.
    This code was shared by Dennis Engemann on github.'''

    if isinstance(key, str):
        key = [key]

    mapping = deepcopy(mapping)

    if any(sep in k_i for k_i in mapping.keys()):
        if any(k_e not in mapping for k_e in key):
            # Select a given key if the requested set of
            # '/'-separated types are a subset of the types in that key

            for k in mapping.keys():
                if not all(set(k_i.split('/')).issubset(k.split('/'))
                           for k_i in key):
                    del mapping[k]

            if len(key) == 0:
                raise KeyError('Attempting selection of keys via '
                               'multiple/partial matching, but no '
                               'event matches all criteria.')
    else:
        raise ValueError('Your keys are bad formatted.')
    return mapping
