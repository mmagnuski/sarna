import numpy as np
import matplotlib.pyplot as plt
from .viz import Topo


# - [ ] lasso selection from mne SelectFromCollection
# - [ ] better support for time-like dim when matrix is freq-freq
# - [ ] add pyqt (+pyqtgraph) backend
class MultiDimView(object):
    def __init__(self, data, axislist=None):
        self.data = data
        if isinstance(axislist, list):
            assert len(axislist) == len(data.shape)
            for i, ax in enumerate(axislist):
                assert len(ax) == data.shape[i]
        self.axlist = axislist
        self.fig, self.ax = plt.subplots()
        self.chan_ind = 0
        self.epoch_ind = 0

        self.launch()
        cid = self.fig.canvas.mpl_connect('key_press_event', self.onclick)

    def launch(self):
        # plt.ion()
        data_to_show = self.get_slice()
        self.im = self.ax.imshow(data_to_show, cmap='hot', interpolation='none',
            origin='lower')

        # set x ticks
        xtck = [int(x) for x in self.im.axes.get_xticks()[1:-1]]
        lfreq = self.axlist[1][xtck]
        self.im.axes.set_xticks(xtck)
        self.im.axes.set_xticklabels(lfreq)

        # set y ticks
        ytck = [int(x) for x in self.im.axes.get_yticks()[1:-1]]
        hfreq = self.axlist[0][ytck]
        self.im.axes.set_yticks(ytck)
        self.im.axes.set_yticklabels(hfreq);

        self.ax.set_title('{}'.format(self.axlist[2][self.chan_ind]))
        self.fig.show()

    def get_slice(self):
        if self.data.ndim == 3:
            data_to_show = self.data[:, :, self.chan_ind]
        elif self.data.ndim == 4:
            data_to_show = self.data[:, :, self.chan_ind, self.epoch_ind]
        return data_to_show

    def refresh(self):
        self.ax.set_title('{}'.format(self.axlist[2][self.chan_ind]))
        self.im.set_data(self.get_slice())
        self.fig.canvas.draw()

    def onclick(self, event):
        if event.key == 'up':
            self.chan_ind += 1
            self.chan_ind = min(self.chan_ind, self.data.shape[2]-1)
            self.refresh()
        elif event.key == 'down':
            self.chan_ind -= 1
            self.chan_ind = max(0, self.chan_ind)
            self.refresh()
        elif event.key == 'left':
            self.epoch_ind -= 1
            self.chan_ind = max(0, self.epoch_ind)
            self.refresh()
        elif event.key == 'right':
            self.epoch_ind += 1
            self.chan_ind = min(self.epoch_ind, self.data.shape[3]-1)
            self.refresh()


class SpectrumPlot(object):
    def __init__(self, psd, freq, info):
        from mne.viz.utils import SelectFromCollection

        self.trial = 0
        self.freq = freq
        self.psd = psd
        self.info = info
        self.ch_names = info['ch_names']
        self.ch_point_size = 36

        self.xpos = 0.
        self.is_mouse_pressed = False

        self.n_trials, self.n_channels, self.n_freqs = psd.shape

        # box selection
        self.freq_box_selection = None
        self.freq_window = slice(0, len(freq))
        self.selected_channels = range(self.n_channels)

        # plot setup
        self.fig = plt.figure()
        self.ax = list()
        self.ax.append(self.fig.add_axes([0.05, 0.1, 0.3, 0.8]))
        self.ax.append(self.fig.add_axes([0.4, 0.05, 0.45, 0.9]))
        self.fig.suptitle('trial {}'.format(self.trial))

        # average psd within freq_window
        topo_data = psd[self.trial, :, self.freq_window].mean(axis=-1)
        self.topo = Topo(topo_data, self.info, axes=self.ax[0])

        # modify topo channels style
        self.topo.chans.set_sizes(
            [self.ch_point_size] * self.topo.chans.get_offsets().shape[0])
        self.topo.chans.set_facecolor('k')

        # add lasso selection
        self.lasso_selection = SelectFromCollection(
            self.topo.axis, self.topo.chans, np.array(self.ch_names))

        # psd plot
        self.psd_lines = self.ax[1].plot(
            self.freq, self.psd[0].mean(axis=0))[0]
        self.ax[1].set_xlabel('Frequency (Hz)')
        self.ax[1].set_ylabel('Power (AU)')

        # ‘key_press_event’	KeyEvent - key is pressed
        # ‘key_release_event’	KeyEvent - key is released
        # ‘motion_notify_event’	MouseEvent - mouse motion

        self.fig.canvas.mpl_connect('lasso_event', self.on_lasso)
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('motion_notify_event',
                                    self.on_mouse_movement)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)


    def on_lasso(self):
        self.selected_channels = [self.ch_names.index(ch) for ch in
                                  self.lasso_selection.selection]
        this_psd = self.psd[self.trial, self.selected_channels].mean(axis=0)
        vmin, vmax = this_psd.min(), this_psd.max()
        y_rng = vmax - vmin
        self.psd_lines.set_ydata(this_psd)
        old_ylim = self.ax[1].get_ylim()
        new_ylim = (min(vmin - 0.05 * y_rng, old_ylim[0]),
                    max(vmax + 0.05 * y_rng, old_ylim[1]))
        self.ax[1].set_ylim(new_ylim)
        self.fig.canvas.draw()

    def on_press(self, event):
        if event.inaxes != self.ax[1]: return
        if self.freq_box_selection is not None:
            self.freq_box_selection.remove()

        self.is_mouse_pressed = True
        self.xpos = event.xdata
        ylim = self.ax[1].get_ylim()
        y_height = ylim[1] - ylim[0]
        smallstep = np.diff(self.freq[[0, 1]]) / 4.
        self.freq_box_selection = plt.bar(
            left=self.xpos - smallstep, bottom=ylim[0], width=smallstep * 2,
            height=y_height, alpha=0.3)[0]
        self.fig.canvas.draw()

    def on_mouse_movement(self, event):
        if not self.is_mouse_pressed: return
        if event.inaxes != self.ax[1]: return

        current_xpos = event.xdata
        ylim = self.ax[1].get_ylim()
        left, bottom = min(current_xpos, self.xpos), ylim[0]


        height = ylim[1] - ylim[0]
        width = np.abs(current_xpos - self.xpos)
        self.freq_box_selection.set_bounds(left, bottom, width, height)

        self.fig.canvas.draw()

    def rescale_box_selection(self):
        ylim = self.ax[1].get_ylim()
        y_height = ylim[1] - ylim[0]
        smallstep = np.diff(self.freq[[0, 1]]) / 2.

        self.freq_box_selection.set_bounds(
            self.freq[self.freq_window.start] - smallstep, ylim[0],
            np.diff(self.freq[self.freq_window][[0, -1]]) + smallstep * 2,
                    y_height)
        self.freq_box_selection.set_alpha(0.5)


    def on_release(self, event):
        from mne.viz.utils import SelectFromCollection

        if event.inaxes != self.ax[1]: return
        self.is_mouse_pressed = False

        # correct box position
        current_xpos = event.xdata
        lfreq, hfreq = (min(current_xpos, self.xpos),
                        max(current_xpos, self.xpos))
        self.freq_window = find_range(self.freq, [lfreq, hfreq])

        self.rescale_box_selection()

        # update topo
        topo_data = self.psd[self.trial, :, self.freq_window].mean(axis=-1)
        self.ax[0].clear()
        self.topo = Topo(topo_data, self.info, axes=self.ax[0])

        # modify topo channels style
        self.topo.chans.set_sizes([self.ch_point_size] *
                                  self.topo.chans.get_offsets().shape[0])
        self.topo.chans.set_facecolor('k')

        # add lasso selection
        self.lasso_selection.disconnect()
        self.lasso_selection = SelectFromCollection(
            self.topo.axis, self.topo.chans, np.array(self.ch_names))
        # self.fig.canvas.mpl_connect('lasso_event', self.on_lasso)

        self.fig.canvas.draw()

    def on_key(self, event):
        from mne.viz.utils import SelectFromCollection
        refresh = False
        if event.key == 'right' and self.trial < self.n_trials:
            self.trial += 1
            refresh = True
        if event.key == 'left' and self.trial > 0:
            self.trial -= 1
            refresh = True

        if refresh:
            # update topo
            topo_data = self.psd[self.trial, :, self.freq_window].mean(axis=-1)
            self.ax[0].clear()
            self.topo = Topo(topo_data, self.info, axes=self.ax[0])

            # modify topo channels style
            self.topo.chans.set_sizes(
                [self.ch_point_size] * self.topo.chans.get_offsets().shape[0])
            self.topo.chans.set_facecolor('k')

            # add lasso selection
            self.lasso_selection.disconnect()
            self.lasso_selection = SelectFromCollection(
                self.topo.axis, self.topo.chans, np.array(self.ch_names))

            # refresh psd
            this_psd = self.psd[self.trial, self.selected_channels].mean(axis=0)
            vmin, vmax = this_psd.min(), this_psd.max()
            y_rng = vmax - vmin
            self.psd_lines.set_ydata(this_psd)
            old_ylim = self.ax[1].get_ylim()
            new_ylim = (min(vmin - 0.05 * y_rng, old_ylim[0]),
                        max(vmax + 0.05 * y_rng, old_ylim[1]))
            self.ax[1].set_ylim(new_ylim)

            # refresh psd box ylims
            self.rescale_box_selection()

            # set overall title to indicate current trial
            self.fig.suptitle('trial {}'.format(self.trial))

            self.fig.canvas.draw()
