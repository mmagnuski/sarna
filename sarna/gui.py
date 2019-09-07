
import numpy as np
import matplotlib.pyplot as plt
from .viz import Topo
from borsar.channels import find_channels


# TODOs:
# MultiDimView will need review and changes:
# - [ ] consider merging with TFR_GUI
# - [ ] add colorbar
# - [ ] cickable (blockable) color bar (??)
# SignalPlotter:
# - [ ] design object API (similar to fastplot)
# - [ ] continuous and epoched signal support


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

        if psd.ndim == 2:
            # no epochs, only channels x frequencies, add epochs dim
            psd = psd[np.newaxis, :]

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


# TODOs:
# - [ ] add colorbars
# - [ ] allow for auto-scaling vmin, vmax for given topo / image
# - [ ] add x and y labels and ticklabels
class TFR_GUI(object):
    '''
    Class for interactive plotting of image-like (for example TFR or PAC)
    mutli-channel results.

    Parameters
    ----------
    data : numpy array
        Array of shape (channels, dim1, dim2), for example (channels,
        frequency, time).
    info : mne.Info
        Info object containing channel position information.
    x_axis : numpy array
        Array of coordinates for the x axis.
    y_axis : numpy array
        Array of coordinates for the y axis.
    x_label : str
        Label for the x axis.
    y_label : str
        Label for the y axis.

    Returns
    -------
    TFR_GUI : TFR_GUI object
        The object containing handles to axes and visual components such as the
        topomap object (``borsar.viz.Topo``) or the heatmap image.
    '''
    def __init__(self, data, info, x_axis=None, y_axis=None, x_label=None,
                 y_label=None):
        import mne
        from mne.viz.utils import SelectFromCollection
        from matplotlib.widgets import RectangleSelector

        img_y = 0.05 if x_label is None else 0.12
        img_h = 0.9 if x_label is None else 0.83

        # create figure and panels
        self.fig = plt.figure(figsize=(10, 5))
        self.topo_ax = self.fig.add_axes([0.05, 0.1, 0.3, 0.8])
        self.image_ax = self.fig.add_axes([0.5, img_y, 0.45, img_h])

        self.data = data
        self.info = info
        ch_names = info['ch_names']

        # init topo
        # ---------
        # average values for channels
        topo_data = data.mean(axis=(1, 2))

        # TODO change vmin and vmax for TFR vs PAC
        vmin = 0
        vmax = data.max()
        self.topo = Topo(topo_data, info, axes=self.topo_ax,
                         vmin=vmin, vmax=vmax)

        # oznaczamy dodatkowymi 'groszkami' pozycję kanałów
        marks = self.topo_ax.scatter(*self.topo.chan_pos.T, marker='o', s=10,
                                     c='k')

        # init image
        # ----------
        avg_img = data.mean(axis=0)
        self.image_ax.imshow(avg_img, aspect='auto', origin='lower', vmin=vmin,
                             vmax=vmax)

        # labels and ticks
        if x_axis is not None:
            _set_img_labels(self.image_ax, x_axis, dim='x')
        if y_axis is not None:
            _set_img_labels(self.image_ax, y_axis, dim='y')
        if x_label:
            self.image_ax.set_xlabel(x_label)
        if y_label:
            self.image_ax.set_ylabel(y_label)

        # interfaces
        # ----------

        # box-selection
        self.box_select = RectangleSelector(self.image_ax, self.on_boxselect,
                                            drawtype='box', useblit=True,
                                            button=[1], interactive=True)

        # lasso selection
        self.lasso_selection = SelectFromCollection(
            self.topo_ax, marks, np.array(ch_names), alpha_other=0.1)

        # podpinamy te funkcje pod odpowiednie wydarzenia
        self.fig.canvas.mpl_connect('key_press_event', self.box_select)
        self.fig.canvas.mpl_connect('lasso_event', self.on_lasso)

    def on_boxselect(self, eclick, erelease):
        ext = self.box_select.extents

        # FIXME: change so that we look for closest limits, not round
        ext = np.array(_special_rounding(ext[:2]) + _special_rounding(ext[2:]))

        bad_idx = ext < 0
        if (bad_idx).any():
            ext[bad_idx] = 0

        # TODO could also check edge limits...

        # pierwszy wymiar adresujemy: ext[2]:ext[3] + 1
        dim1 = slice(ext[2], ext[3] + 1)

        # drugi wymiar adresujemy: ext[0]:ext[1] + 1
        dim2 = slice(ext[0], ext[1] + 1)

        topo_data = self.data[:, dim1, dim2].mean(axis=(1, 2))
        self.topo.update(topo_data)

        ext = ext + np.array([-0.5, 0.5, -0.5, 0.5])
        self.box_select.extents = ext.tolist() # tolist() may not be needed

        # wymuszamy redraw (may not be needed)
        self.fig.canvas.draw()

    def on_lasso(self):
        ch_idx = find_channels(self.info, self.lasso_selection.selection)
        avg_img = self.data[ch_idx].mean(axis=0)

        # TODO - change imshow to image update
        self.image_ax.images[0].set_data(avg_img)

        # may not be needed:
        # self.box_select.update()


def _special_rounding(vals):
    from decimal import localcontext, Decimal, ROUND_HALF_UP, ROUND_HALF_DOWN
    out = list()
    with localcontext() as ctxt:
        ctxt.rounding = ROUND_HALF_UP
        this_val = Decimal(vals[0]).to_integral_value()
        out.append(int(this_val))

    with localcontext() as ctxt:
        ctxt.rounding = ROUND_HALF_DOWN
        this_val = Decimal(vals[1]).to_integral_value()
        out.append(int(this_val))

    return out


def _set_img_labels(ax, labels, dim='x'):
    if dim == 'x':
        idx = ax.get_xticks().astype('int')
    elif dim == 'y':
        idx = ax.get_yticks().astype('int')

    in_limits = (idx >= 0) & (idx < len(labels))
    idx[~in_limits] = 0
    use_labels = labels[idx]
    use_labels[~in_limits] = np.nan

    if dim == 'x':
        ax.set_xticklabels(use_labels)
    elif dim == 'y':
        ax.set_yticklabels(use_labels)
