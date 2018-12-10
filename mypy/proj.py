import os
import os.path as op
import json
from sys import platform


def find_dropbox():
    if any([platform == plt for plt in ["linux", "linux2", "darwin"]]):
        config_pth = os.path.expanduser('~/.dropbox')
    elif platform == "win32":
        config_pth = op.join(os.getenv('APPDATA')[:-8], 'Local', 'Dropbox')
    if os.path.exists(config_pth):
        with open(op.join(config_pth, 'info.json')) as f:
            info = json.load(f)
        return info['personal']['path']


def get_valid_path(pth_list):
    for pth in pth_list:
        if os.path.exists(pth):
            return pth
    raise ValueError('could not find valid path')
