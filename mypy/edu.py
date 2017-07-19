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


def change_lines(event, dc=None):
    '''temporary function to update figure'''

    upto = dc['upto']
    if event.key == 'up':
        upto += 1
        upto = min([250, upto])
    elif event.key == 'down':
        upto -= 1
        upto = max([0, upto])

    simi, spect = create_sim(dc['spectrum'], upto)
    dc['sim_line'].set_data(dc['x'], simi)
    dc['spect_scatter'].remove()
    plot_spect = np.abs(spect[:251])
    plot_spect[plot_spect == 0] = np.nan
    dc['spect_scatter'] = dc['ax'][1].scatter(dc['freqs'], plot_spect,
                                              color='r')
    dc['upto'] = upto
    dc['fig'].canvas.draw()


def create_sim(spectrum, upto):
    from scipy.fftpack import ifft

    upto = min([max([0, upto]), 250])
    spect = np.zeros(len(spectrum), dtype='complex')
    all_inds = np.argsort(np.abs(spectrum[:251]))[::-1]

    for ind in all_inds[:upto]:
        spect[ind] = spectrum[ind]
        spect[-ind] = spectrum[-ind]

    return ifft(spect), spect


def spectral_reconstruction(raw, ch_name='Oz', start_t=5.):
    from functools import partial
    import matplotlib.pyplot as plt
    from scipy.fftpack import fft

    # select data
    ch_index = raw.ch_names.index(ch_name)
    start, stop = (np.array([start_t, start_t + 2.]) *
                   raw.info['sfreq']).astype('int')
    signal = raw._data[ch_index, start:stop]

    # calculate spectrum and freqs
    n_fft = len(signal)
    sfreq = raw.info['sfreq']
    freqs = np.arange(n_fft // 2 + 1, dtype=float) * (sfreq / n_fft)
    spectrum = fft(signal)

    fig, ax = plt.subplots(nrows=2)

    upto = 0
    x = np.linspace(0, len(signal) / sfreq, num=len(signal))
    d = dict(upto=upto, ax=ax, freqs=freqs, spectrum=spectrum, x=x)

    sim, spect = create_sim(spectrum, 0)
    ax[0].plot(x, signal)
    sim_line = ax[0].plot(x, sim, color='r')[0]

    plot_spect = np.abs(spect[:251])
    plot_spect[plot_spect == 0] = np.nan
    ax[1].plot(freqs, np.abs(spectrum[:251]))
    spect_scatter = ax[1].scatter(freqs, plot_spect, color='r')

    d['sim_line'] = sim_line
    d['spect_scatter'] = spect_scatter
    d['fig'] = fig

    change_lines_prt = partial(change_lines, dc=d)
    fig.canvas.mpl_connect('key_press_event', change_lines_prt)
