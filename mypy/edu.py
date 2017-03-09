import os
import platform
import importlib

import numpy as np


def test_system():
    '''Print simple system info and some other junk, just to see if
    system has been set up and homeworks are from different machines'''

    modules = ['matplotlib', 'seaborn', 'mne', 'mypy']
    longest_str = max(map(len, modules)) + 8
    txt = '\n{} {}\n{}\n'.format(platform.system(), platform.machine(),
                                 platform.processor())

    # check module presence and versions
    for module in modules:
        txt += '\n{}: '.format(module)
        try:
            mdl = importlib.import_module(module)
            base_txt = '{:>%d}' % (longest_str - len(module))
            txt += base_txt.format(mdl.__version__)
        except ImportError:
            txt += 'BRAK :('
        if module in ('mne', 'mypy'):
            txt += "; instalacja z git'a" if is_git_installed(mdl) \
                else ";  zwykła instalacja"

    # print some random junk
    values = np.random.randint(0, 1001, (2, 3))
    txt += '\n\nTwoje szczęśliwe liczby to:\n{}'.format(values)
    print(txt)


def is_git_installed(module):
    '''simple check for whether module is git-installed - tests for presence
    of .git directory and relevant git subdirectories'''
    sep = os.path.sep
    module_dir = sep.join(module.__file__.split(sep)[:-2])
    has_all_dirs = False
    if '.git' in os.listdir(module_dir):
        subdirs = ['hooks', 'info', 'logs', 'objects', 'refs']
        git_dir_contents = os.listdir(os.path.join(module_dir, '.git'))
        has_all_dirs = all([x in git_dir_contents for x in subdirs])
    return has_all_dirs
