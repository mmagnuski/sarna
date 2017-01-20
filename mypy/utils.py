import numpy as np


# - [ ] may not be necessary any longer...
def do_not_warn():
    '''turns off DeprecationWarnings as they can be (were)
    painful in the current (older) jupyter notebook'''
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


# - [ ] convenient reloading will require much more work
def rld(pkg):
    '''fast reaload (no need to type much)'''
    import importlib
    importlib.reload(pkg)


# TODO:
# - [ ] should check type and size of the vars (check how mne-python does it)
# - [ ] detect if in notebook/qtconsole and ignore 'Out' and similar vars
# - [ ] display numpy arrays as "5 x 2 int array"
#       lists as "list of strings" etc.
def whos():
    """Print the local variables in the caller's frame.
    Copied from stack overflow:
    http://stackoverflow.com/questions/6618795/get-locals-from-calling-namespace-in-python"""
    import inspect
    frame = inspect.currentframe()
    ignore_vars = []
    ignore_starting = ['__']
    try:
        lcls = frame.f_back.f_locals
        # test if ipython
        ipy_vars = ['In', 'Out', 'get_ipython', '_oh', '_sh']
        in_ipython = all([var in lcls for var in ipy_vars])
        if in_ipython:
            ignore_vars = ipy_vars + ['_']
            ignore_starting += ['_i']
        for name, var in lcls.items():
            if name in ignore_vars:
                continue
            if any([name.startswith(s) for s in ignore_starting]):
                continue
            print(name, type(var), var)
    finally:
        del frame


# - [ ] if np.ndarray try to format output in the right shape
def find_index(vec, vals):
    if not isinstance(vals, (list, tuple, np.ndarray)):
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


def extend_slice(slc, val, maxind):
    start, stop, step = slc.start, slc.stop, slc.step
    if not start == 0:
        start -= val
        if start < 0:
            start = 0
    if not stop == maxind:
        stop += val
        if stop > maxind:
            stop = maxind
    return slice(start, stop, step)


# join inds
# TODO:
# - [ ] docs!
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
    This code was shared by Dennis Engemann on github.

    Parameters
    ----------
    key : string | list of strings
        Keys to subselect with.
    mapping : dict
        Dictionary that is being selected.
    sep : string
        Separator to use in subselection.
    '''

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


# - [ ] more checks for mne type
# - [ ] maybe move to mneutils ?
def get_info(inst):
    from mne.io.meas_info import Info
    if isinstance(inst, Info):
        return inst
    else:
        return inst.info
