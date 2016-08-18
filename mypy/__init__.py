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
