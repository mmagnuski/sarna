import os
import platform
import importlib

import numpy as np


def test_system():
    '''Print simple system info and some other junk, just to see if
    system has been set up and homeworks are from different machines.'''

    try:
        import mne
    except ImportError:
        raise ImportError('Nie masz biblioteki `mne`!')
    mne.sys_info()

    modules = ['seaborn', 'borsar', 'sarna']
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
        if module in ('borsar', 'sarna'):
            txt += "; instalacja z git'a" if is_git_installed(mdl) \
                else ";  zwykła instalacja"

    # print some random junk
    values = np.random.randint(0, 1001, (2, 3))
    txt += '\n\nTwoje szczęśliwe liczby to:\n{}'.format(values)
    print(txt)


def is_git_installed(module):
    '''Simple check for whether module is git-installed.

    Tests for the presence of a ``.git`` directory and some other relevant git
    subdirectories.
    '''
    sep = os.path.sep
    module_dir = sep.join(module.__file__.split(sep)[:-2])
    has_all_dirs = False
    if '.git' in os.listdir(module_dir):
        subdirs = ['hooks', 'info', 'logs', 'objects', 'refs']
        git_dir_contents = os.listdir(os.path.join(module_dir, '.git'))
        has_all_dirs = all([x in git_dir_contents for x in subdirs])
    return has_all_dirs


def plot_matrix(matrix, colors=False, bgcolor=None, fmt='.2g'):
    '''Plot 2d matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        2d numpy array to plot.
    colors : bool
        Whether to color the matrix cells according to their values.
    bgcolor : rgb | None
        Cell color if ``colors`` is ``False`` (all cells have uniform color
        then). List of ``[r, g, b]`` values.
    fmt : string, optional
        String formatting code to when writing matrix values.

    Returns
    -------
    ax : matplotlib.Axes
        Axes with the heatmap.
    '''
    import seaborn as sns

    if matrix.ndim == 1:
        matrix = matrix[np.newaxis, :]

    # set default parameters
    longest = max(matrix.shape)
    base_fontsize = 16 if matrix.dtype == 'int' else 14
    fontsize = (base_fontsize if longest <= 8 else
                max(round(base_fontsize - (longest - 8) / 2), 2))
    font_props = {'fontsize': fontsize}
    if not colors:
        font_props['color'] = 'k'
    bgcolor = [[0.9, 0.9, 0.6]] if bgcolor is None else [bgcolor]

    # use seaborn to plot the heatmap
    ax = sns.heatmap(matrix, annot=True, annot_kws=font_props, cbar=colors,
                     linewidths=1, xticklabels=[], yticklabels=[], fmt=fmt)

    # modify seaborn plot to have uniform background color
    if not colors:
        qmesh = ax.get_children()[0]
        color_array = np.ones((25, 4))
        color_array[:, :3] = bgcolor
        qmesh.set_facecolors(color_array)

    # change figure size
    dim1, dim2 = matrix.shape
    dim1, dim2 = min(dim2, 8), min(dim1, 8)
    fig = ax.figure
    fig.set_size_inches((dim1, dim2))

    return ax


def spectral_reconstruction(raw, ch_name='Oz', tmin=5., tmax=None):
    """Simple interface showing how adding sinusoids with frequency and
    phase taken from the fft of the signal leads to reconstructing the
    original signal in the time domain.

    Parameters
    ----------
    raw : mne.Raw
        Mne instance of Raw object. The signal to use in plotting and
        reconstruction.
    ch_name : str
        Name of the channel chosen to visualize.
    tmin : int | float
        Start of the time segment to investigate in seconds.
    tmax : int | float | None
        End of the time segment to investigate in seconds. Default is None
        which calculates ``tmax`` based on ``tmin`` as ``tmin + 2``.
    """
    from functools import partial
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft

    # select data
    ch_index = raw.ch_names.index(ch_name)
    tmax = tmin + 2. if tmin is None else tmax
    sfreq = raw.info['sfreq']
    start, stop = (np.array([tmin, tmax]) * sfreq).astype('int')
    signal = raw._data[ch_index, start:stop]

    # calculate spectrum and freqs
    n_fft = len(signal)
    n_freqs = n_fft // 2 + 1
    freqs = np.arange(n_freqs, dtype=float) * (sfreq / n_fft)
    spectrum = fft(signal)

    fig, ax = plt.subplots(nrows=2)

    time = np.linspace(tmin, tmax, num=len(signal))
    d = dict(fig=fig, upto=0, ax=ax, freqs=freqs, n_freqs=n_freqs,
             spectrum=spectrum, time=time)

    sim, spect = _create_sim(spectrum, 0, n_freqs)
    ax[0].plot(time, signal)
    sim_line = ax[0].plot(time, sim, color='r')[0]

    plot_spect = np.abs(spect[:n_freqs])
    plot_spect[plot_spect == 0] = np.nan
    ax[1].plot(freqs, np.abs(spectrum[:n_freqs]))
    spect_scatter = ax[1].scatter(freqs, plot_spect, color='r')

    d['sim_line'] = sim_line
    d['spect_scatter'] = spect_scatter

    change_lines_prt = partial(_spect_recon_change_lines, dc=d)
    fig.canvas.mpl_connect('key_press_event', change_lines_prt)


def _spect_recon_change_lines(event, dc=None):
    '''Temporary function to update ``spectral_reconstruction`` figure.'''

    upto = dc['upto']
    if event.key == 'up':
        upto += 1
        upto = min([dc['n_freqs'], upto])
    elif event.key == 'down':
        upto -= 1
        upto = max([0, upto])

    simi, spect = _create_sim(dc['spectrum'], upto, dc['n_freqs'])
    dc['sim_line'].set_data(dc['time'], simi)
    dc['spect_scatter'].remove()
    plot_spect = np.abs(spect[:dc['n_freqs']])
    plot_spect[plot_spect == 0] = np.nan
    dc['spect_scatter'] = dc['ax'][1].scatter(dc['freqs'], plot_spect,
                                              color='r')
    dc['upto'] = upto
    dc['fig'].canvas.draw()


def _create_sim(spectrum, upto, mx):
    from scipy.fftpack import ifft

    upto = min([max([0, upto]), mx])
    spect = np.zeros(len(spectrum), dtype='complex')
    all_inds = np.argsort(np.abs(spectrum[:mx]))[::-1]

    # TODO: vectorize
    for ind in all_inds[:upto]:
        spect[ind] = spectrum[ind]
        spect[-ind] = spectrum[-ind]

    return ifft(spect), spect
