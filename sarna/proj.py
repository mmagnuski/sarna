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


def find_onedrive(kind='auto'):
    '''Find OneDrive directory.

    Parameters
    ----------
    kind : str
        One of ``'auto'``, ``'personal'``, ``'business'``. Defaults to
        ``'auto'``. If ``'auto'``, the function will try to find the business
        OneDrive first, and if it fails, it will try to find the personal.

    Returns
    -------
    onedrive_path : str
        Full path to OneDrive directory.
    '''
    if kind == 'auto':
        try:
            return find_onedrive(kind='business')
        except ValueError:
            return find_onedrive(kind='personal')

    var_name = ('OneDriveConsumer' if kind == 'personal'
                else 'OneDriveCommercial')
    onedrive_path = os.getenv(var_name)
    if onedrive_path is None:
        raise ValueError(f'Could not find {kind} OneDrive.')
    return onedrive_path


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


# TODO - actually returns the drive letter + .shortcut-targets-by-id folder
def find_google_drive(all=True):
    '''Find google drive local drive.

    This function assumes your Google Drive is installed as a separate system
    drive, and that it contains shared folders (then
    ``.shortcut-targets-by-id`` subdirectory is present on the drive).

    Parameters
    ----------
    all : bool
        If True, return all google drive directories, otherwise return only the
        first one found. Defaults to ``True``.

    Returns
    -------
    google_drives : list of str | str
        List of google drive directories if ``all=True``, else string with the
        first directory found.
    '''
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


def find_shared_folder(folder_name, as_path=False):
    '''Find shared folder on google drive.

    Parameters
    ----------
    folder_name : str
        Name of the shared directory.
    as_path : bool
        If True, return the path as a ``pathlib.Path`` object. Defaults to
        ``False``.

    Returns
    -------
    shared_folder : str | pathlib.Path
        Path to the shared folder.
    '''
    if as_path:
        from pathlib import Path

    drive_dirs = find_google_drive(all=True)

    # there may be more than one drive (multiple accounts)
    for drive_dir in drive_dirs:
        subdirs = os.listdir(drive_dir)

        # there can be many linking subdirectories in .shortcut-targets-by-id
        # google drive folder, we find the one containing the target directory:
        for subdir in subdirs:
            pth = op.join(drive_dir, subdir)
            if op.isdir(pth):
                contents = os.listdir(pth)
                if folder_name in contents:
                    pth = op.join(pth, folder_name)
                    pth = Path(pth) if as_path else pth
                    return pth
    raise ValueError(f'Could not find {folder_name} folder.')


# TODO: return_files also returns directories (check!)
def find_subjects(directory, pattern='sub-\w+',
                  return_files=False, remove_prefix=True):
    '''Find files / subdirectories matching a pattern in given directory.

    Parameters
    ----------
    directory : str
        Directory to search.
    pattern : str
        Pattern (regular expression) to match.
    return_files : bool
        If True, return all files matching the pattern, otherwise return only
        the matching span (subject id). Defaults to ``False``.
    remove_prefix : bool
        If True, remove the prefix (e.g. 'sub-') from the subject id. Defaults
        to ``True``.

    Returns
    -------
    subjects : list of str
        List of subject ids.
    files : dict
        Dictionary of subject id -> list of files matching the subject id. Only
        returned if ``return_files`` is ``True``.
    '''
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
