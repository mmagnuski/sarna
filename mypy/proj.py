import os
import json
import yaml
from sys import platform


def find_dropbox():
    if platform == "linux" or platform == "linux2":
        drp_pth = os.path.expanduser('~/.dropbox')
    # elif platform == "darwin":
    #     # dropbox on OS X?
    elif platform == "win32":
        drp_pth = os.path.join(os.getenv('APPDATA')[:-8], 'Local', 'Dropbox')
    if os.path.exists(drp_pth):
        info_file = os.path.join(drp_pth, 'info.json')
        with open(info_file) as f:
            info = json.load(f)
        return info['personal']['path']


def get_valid_path(pth_list):
    for pth in pth_list:
        if os.path.exists(pth):
            return pth
    raise ValueError('could not find valid path')
