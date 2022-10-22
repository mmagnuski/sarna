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


def find_google_drive(all=True):
    '''Find google drive.

    This function assumes your Google Drive is installed as a separate system
    drive, and that it contains shared folders (then
    ``.shortcut-targets-by-id`` subdirectory is present on the drive).'''
    from string import ascii_uppercase
    drive_letters = ascii_uppercase[5:]

    # this is the directory where linked shared drives are stored
    # but drive letter can be changed
    google_drives = list()
    for drive in drive_letters:
        check_dir = f'{drive}:\.shortcut-targets-by-id'
        if op.exists(check_dir):
            google_drives.append(check_dir)

    if len(google_drives) == 0:
        raise ValueError('Could not find google drive folder.')
    elif not all:
        google_drives = google_drives[0]

    return google_drives


def find_shared_folder(project_name, as_path=False):
    '''Find shared folder on google drive.'''
    if as_path:
        from pathlib import Path

    drive_dirs = find_google_drive(all=True)

    # there may be more than one drive (multiple accounts)
    for drive_dir in drive_dirs:
        subdirs = os.listdir(drive_dir)

        # there can be many linking subdirectories in .shortcut-targets-by-id
        # google drive folder, we find the one containing switchorder directory:
        for subdir in subdirs:
            pth = op.join(drive_dir, subdir)
            if op.isdir(pth):
                contents = os.listdir(pth)
                if project_name in contents:
                    pth = op.join(pth, project_name)
                    pth = Path(pth) if as_path else pth
                    return pth
    raise ValueError(f'Could not find {project_name} folder.')




# TODO - remove_prefix = True removes 'sub-'
def find_subjects(directory=None, pattern='sub-\w+',
                  return_files=False, remove_prefix=True):
    '''Find files / directories matching a pattern.'''
    import re

    subjects = list()
    if return_files:
        files = dict()

    fls = os.listdir(directory)

    for file in fls:
        match = re.match(pattern, file)
        if match is not None:
            subj = file[slice(*match.span())]
            if subj not in subjects:
                subjects.append(subj)
                if return_files:
                    files[subj] = list()
            if return_files:
                files[subj].append(file)

    subjects.sort()
    if remove_prefix:
        subjects = [subj.replace('sub-', '') for subj in subjects]

    if return_files:
        return subjects, files
    else:
        return subjects