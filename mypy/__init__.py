# some imports
from . import events
from .colors import colors


# later move these to utils:
def do_not_warn():
    '''turns off DeprecationWarnings as they can be
    painful in the current jupyter notebook'''
    import warnings
    try:
        from exceptions import DeprecationWarning # py2
    except ImportError:
        pass # py3
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # warnings.filterwarnings('ignore', category=DeprecationWarning, message='.*use @default decorator instead.*')
    # warnings.filterwarnings('ignore', category=DeprecationWarning, module='.*/IPython/.*')
    # another way, turn only deprecation wornings for a specific package (like mne):
    # warnings.filterwarnings('default', category=DeprecationWarning, module='.*/mypackage/.*')


def rld(pkg):
    '''fast reaload (no need to type much)'''
    import importlib
    importlib.reload(pkg)