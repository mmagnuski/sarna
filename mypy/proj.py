import os
import json
import yaml

# add Project class - can read yaml project setup info
#                   - governs path etc.
# basic project structure(?):
#   code
#   notebooks
#   data
#   report (/fig)
#

# dropping stuff into global workspace:
# globals()['var'] = "an object"
#
# def insert_into_namespace(name_space, name, value=None):
#     name_space[name] = value
#
# insert_into_namespace(globals(), "var", "an object")


def find_dropbox():
    app = os.getenv('APPDATA')
    path_list = app[3:].split('\\')[:-1]
    path_list.extend(['Local', 'Dropbox'])
    drp_pth = app[:3] + os.path.join(*path_list)
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


def read_paths(fname):
    paths = yaml.load(fname)
