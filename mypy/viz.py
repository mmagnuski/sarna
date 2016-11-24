import numpy as np
import matplotlib.pyplot as plt

# TODOs:
# MultiDimView:
# - [ ] add colorbar
# - [ ] add topo view
# - [ ] clickable topo view
# - [ ] cickable (blockable) color bar
# - [ ] add window select


def get_spatial_colors(inst):
    from mne.viz.evoked import _rgb
    info = get_info(inst)

    # this should be get_channel_pos or sth like this
    locs3d = np.array([info['chs'][i]['loc'][:3] \
                       for i in range(len(info['ch_names']))])
    x, y, z = locs3d.T
    return _rgb(info, x, y, z)


def masked_image(img, mask, alpha=0.75, mask_color=(0.5, 0.5, 0.5),
                 axis=None, **imshow_kwargs):
    defaults = {'interpolation': 'none', 'origin': 'lower'}
    defaults.update(imshow_kwargs)

    if axis is None:
        fig, axis = plt.subplots()

    # plot images
    main_img = axis.imshow(img, **defaults)
    mask_img = add_image_mask(mask, alpha=alpha, mask_color=mask_color,
                              axis=axis, **defaults)
    return main_img, mask_img


def add_image_mask(mask, alpha=0.75, mask_color=(0.5, 0.5, 0.5),
                   axis=None, **imshow_kwargs):
    if axis is None:
        axis = plt.gca()

    # create RGBA mask:
    mask_img = np.array(list(mask_color) + [0.]).reshape((1, 1, 4))
    mask_img = np.tile(mask_img, list(mask.shape) + [1])

    # set alpha
    mask_img[np.logical_not(mask), -1] = alpha

    # plot images
    return axis.imshow(mask_img, **imshow_kwargs)


# TODO
# - [ ] lasso selection from mne SelectFromCollection
# - [ ] better support for time-like dim when matrix is freq-freq
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


def set_3d_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().

    modified from:
    http://stackoverflow.com/questions/13685386/matplotlib-equal-unit-length-with-equal-aspect-ratio-z-axis-is-not-equal-to
    '''

    x_lim, y_lim, z_lim = ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()

    def get_range(lim):
        return lim[1] - lim[0], np.mean(lim)
    x_range, x_mean = get_range(x_lim)
    y_range, y_mean = get_range(y_lim)
    z_range, z_mean = get_range(z_lim)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_mean - plot_radius, x_mean + plot_radius])
    ax.set_ylim3d([y_mean - plot_radius, y_mean + plot_radius])
    ax.set_zlim3d([z_mean - plot_radius, z_mean + plot_radius])


class Topo(object):
    def __init__(self, values, info):
        from mne.viz.topomap import plot_topomap
        import matplotlib as mpl

        self.info = info
        self.values = values

        # plot topomap
        im, lines = plot_topomap(values, info)

        self.fig = im.figure
        self.img = im
        self.lines = lines
        self.marks = list()
        self.chans = im.axes.findobj(mpl.patches.Circle)
        self.chan_pos = np.array([ch.center for ch in self.chans])

    def remove_level(self, lvl):
        if not isinstance(lvl, list):
            lvl = [lvl]
        for l in lvl:
            remove_lines = np.where(self.lines.levels == l)[0]
            for rem_ln in remove_lines:
                self.lines.collections[rem_ln].remove()
            for pop_ln in np.flipud(np.sort(remove_lines)):
                self.lines.collections.pop(pop_ln)

    def solid_lines(self):
        for ln in self.lines.collections:
            ln.set_linestyle('-')

    def mark_channels(self, chans, **marker_params):
        default_marker = dict(marker='o', markerfacecolor='w',
                              markeredgecolor='k', linewidth=0, markersize=4)
        for k in marker_params.keys():
            default_marker[k] = marker_params[k]

        # mark
        marks = self.img.axes.plot(self.chan_pos[chans, 0],
                                   self.chan_pos[chans, 1], **default_marker)
        self.marks.append(marks)

# other experiments:
# ------------------
# # getting pos of chan mask:
# add_chans = list()
# test_lines = im.axes.findobj(mpl.lines.Line2D)
# for l in test_lines:
#     if l.get_marker() is not 'None':
#         add_chans.append(l)
#
# x, y = add_chans[0].get_data()
#
# from mne.viz.topomap import _prepare_topo_plot
# _, pos, _, _, _ = _prepare_topo_plot(eeg, 'eeg', layout=None)
# pos


# ClusterViewer
# on init:
# inst -> info
# fig, sel = eeg.plot_sensors(kind='select')
#
# def pacplot(ch_ind=None, fig=fig):
#     if ch_ind is None:
#         ch_ind = [eeg.ch_names.index(ch) for ch in fig.lasso.selection]
#     im = t_effect[ch_ind, :, :].mean(axis=0).T
#     mask = cluster_id[ch_ind, :, :].mean(axis=0).T > 0.5
#     fig, ax = plt.subplots()
#     masked_image(im, mask, origin='lower', vmin=-5, vmax=5)
#
#     ax.set_xticks(f_l)
#     ax.set_xticklabels(f_low[f_l])
#     ax.set_yticks(f_h)
#     ax.set_yticklabels(f_high[f_h])
#     ax.figure.canvas.draw()
#
# def on_pick(event, fig=fig):
#     if event.mouseevent.key == 'control' and fig.lasso is not None:
#          for ind in event.ind:
#              fig.lasso.select_one(event.ind)
#
#          return
#     pacplot(ch_ind=event.ind)
#
# fig.canvas.mpl_connect('pick_event', on_pick)
# fig.canvas.mpl_connect('lasso_event', pacplot)
