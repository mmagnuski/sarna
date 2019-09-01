import os
import os.path as op
from pathlib import Path
import json
from sys import platform


def find_dropbox():
    '''Find dropbox location.

    Returns
    -------
    dropbox_path : str
        Full path to main Dropbox directory.
    '''
    if any([platform == plt for plt in ["linux", "linux2", "darwin"]]):
        config_pth = op.expanduser('~/.dropbox')
    elif platform == "win32":
        appdata = Path(os.getenv('APPDATA')).parent
        config_pth = op.join(appdata, 'Local', 'Dropbox')
    if op.exists(config_pth):
        json_path = op.join(config_pth, 'info.json')
        with open(json_path) as f:
            info = json.load(f)
        return info['personal']['path']
    else:
        raise ValueError('Could not find Dropbox directory.')


def get_valid_path(pth_list):
    '''
    Select the first path that exists on current machine.

    Parameters
    ----------
    pth_list : list of str
        List of paths to check.

    Returns
    -------
    pth : str
        The first path that exists on current machine.
    '''
    for pth in pth_list:
        if os.path.exists(pth):
            return pth
    raise ValueError('could not find valid path')
