import numpy as np
import matplotlib.pyplot as plt
from mypy.utils import get_info, find_index

# TODOs:
# MultiDimView:
# - [ ] add colorbar
# - [ ] add topo view
# - [ ] clickable topo view
# - [ ] cickable (blockable) color bar
# - [ ] add window select
# SignalPlotter:
# - [ ] design object API (similar to fastplot)
# - [ ] continuous and epoched signal support


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


# - [ ] should check for image extent, origin etc.
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


# TODO:
# [ ] add docs
# - [ ] add support for vectors of topographies
# - [ ] check out psychic.scalpplot.plot_scalp for slightly different topo plots
#       https://github.com/wmvanvliet/psychic/blob/master/psychic/scalpplot.py
class Topo(object):
    '''High-level object that allows for convenient topographic plotting.
    * FIXME *

    Example
    -------
    topo = Topo(values, info, axis=ax)
    topo.remove_levels(0)
    topo.solid_lines()
    topo.set_linewidth(1.5)
    topo.mark_channels([4, 5, 6], markerfacecolor='r', markersize=12)
    '''

    def __init__(self, values, info, **kwargs):
        from mne.viz.topomap import plot_topomap
        import matplotlib as mpl

        self.info = info
        self.values = values

        has_axis = 'axis' in kwargs.keys()
        if has_axis:
            self.axis = kwargs['axis']
            plt.sca(self.axis)

        # plot using mne's topomap
        im, lines = plot_topomap(values, info, **kwargs)

        self.fig = im.figure
        if not has_axis:
            self.axis = im.axes
        self.img = im
        self.lines = lines
        self.marks = list()
        self.chans = im.axes.findobj(mpl.patches.Circle)
        self.chan_pos = np.array([ch.center for ch in self.chans])

    def remove_levels(self, lvl):
        if not isinstance(lvl, list):
            lvl = [lvl]
        for l in lvl:
            remove_lines = np.where(self.lines.levels == l)[0]
            for rem_ln in remove_lines:
                self.lines.collections[rem_ln].remove()
            for pop_ln in np.flipud(np.sort(remove_lines)):
                self.lines.collections.pop(pop_ln)

    def solid_lines(self):
        self.set_linestyle('-')

    def set_linestyle(self, *args, **kwargs):
        for ln in self.lines.collections:
            ln.set_linestyle(*args, **kwargs)

    def set_linewidth(self, lw):
        for ln in self.lines.collections:
            ln.set_linewidths(lw)

    def mark_channels(self, chans, **marker_params):
        default_marker = dict(marker='o', markerfacecolor='w',
                              markeredgecolor='k', linewidth=0, markersize=8)
        for k in marker_params.keys():
            default_marker[k] = marker_params[k]

        # mark
        marks = self.axis.plot(self.chan_pos[chans, 0],
                                   self.chan_pos[chans, 1], **default_marker)
        self.marks.append(marks)

# # for Topo, setting channel props:
# for ch in ch_ind:
#     self.chans[ch].set_color('white')
#     self.chans[ch].set_radius(0.01)
#     self.chans[ch].set_zorder(4)



# TODOs:
# - [ ] fix issue with shapes that touch matrix edge
# - [ ] let it work for all the shapes (not only the first one completed)
# - [ ] docstring
# - [ ] cluster mode (returns a list or dict mapping cluster ids to list of
#       cluster contours) - so that each cluster can be marked by a different
#       color.
# - [ ] one convolution for all clusters
def create_cluster_contour(mask):
    from scipy.ndimage import convolve #, label

    mask_int = mask.astype('int')
    kernels = {'upper': [[-1], [1], [0]],
               'lower': [[0], [1], [-1]],
               'left': [[-1, 1, 0]],
               'right': [[0, 1, -1]]}
    kernels = {k: np.array(v) for k, v in kernels.items()}
    lines = {k: (convolve(mask_int, v[::-1, ::-1]) == 1).astype('int')
             for k, v in kernels.items()}

    search_order = {'upper': ['right', 'left', 'upper'],
                    'right': ['lower', 'upper', 'right'],
                    'lower': ['left', 'right', 'lower'],
                    'left': ['upper', 'lower', 'left']}
    movement_direction = {'upper': [0, 1], 'right': [1, 0],
                          'lower': [0, -1], 'left': [-1, 0]}
    search_modifiers = {'upper_left': [-1, 1], 'right_upper': [1, 1],
                        'lower_right': [1, -1], 'left_lower': [-1, -1]}
    finish_modifiers = {'upper': [-0.5, 0.5], 'right': [0.5, 0.5],
                        'lower': [0.5, -0.5], 'left': [-0.5, -0.5]}

    # current index - upmost upper line
    current_index = np.array([x[0] for x in np.where(lines['upper'])])
    closed_shape = False
    current_edge = 'upper'
    edge_points = [tuple(current_index + [-0.5, -0.5])]
    direction = movement_direction[current_edge]

    while not closed_shape:
        ind = tuple(current_index)
        new_edge = None

        # check the next edge
        for edge in search_order[current_edge]:
            modifier = '_'.join([current_edge, edge])
            has_modifier = modifier in search_modifiers
            if has_modifier:
                modifier_value = search_modifiers[modifier]
                test_ind = tuple(current_index + modifier_value)
            else:
                test_ind = ind

            if lines[edge][test_ind] == 1:
                new_edge = edge
                lines[current_edge][ind] = -1
                break
            elif lines[edge][test_ind] == -1: # -1 means 'visited'
                closed_shape = True
                new_edge = 'finish'
                lines[current_edge][ind] = -1
                break

        if not new_edge == current_edge:
            edge_points.append(tuple(
                current_index + finish_modifiers[current_edge]))
            direction = modifier_value if has_modifier else [0, 0]
            current_edge = new_edge
        else:
            direction = movement_direction[current_edge]

        current_index += direction
    return edge_points


def highlight(x_values, which_highligh, kind='patch', color=None,
              alpha=0.3, axis=None, level=0.04, height=0.03):
    '''Highlight ranges along x axis.

    Parameters
    ----------
    x_values : numpy array
        Values specifying x axis points along which which_highligh operates.
    which_highligh : numpy array of bool
        Boolean values - each entry specifies whether corresponding value for
        x_values belongs to highlighted ranges.
    '''
    from matplotlib.patches import Rectangle
    from mypy.utils import group

    if color is None:
        color = 'orange' if kind == 'patch' else 'k'
    axis = plt.gca() if axis is None else axis

    ylims = axis.get_ylim()
    y_rng = np.diff(ylims)
    hlf_dist = np.diff(x_values).mean() / 2
    grp = group(which_highligh, return_slice=True)
    for slc in grp:
        this_x = x_values[slc]
        start = this_x[0] - hlf_dist
        length = np.diff(this_x[[0, -1]]) + hlf_dist * 2
        ptch = Rectangle((start, ylims[0]), length, y_rng, lw=0,
                         facecolor=color, alpha=alpha)
        axis.add_patch(ptch)


def plot_topomap_raw(raw, times=None):
    '''plot_topomap for raw mne objects

    Parameters
    ----------
    raw : mne Raw object
        mne Raw object instance
    times : list of ints or floats (or just int/flot)
        Times to plot topomaps for.

    returns
    -------
    fig : matplotlib figure
        Figure handle
    '''
    import mne
    import matplotlib.pyplot as plt

    if times is None:
        raise TypeError('times must be a list of real values.')
    elif not isinstance(times, list):
        times = [times]

    # ADD a check for channel pos

    # pick only data channels (currently only eeg)
    picks = mne.pick_types(raw.info, eeg=True, meg=False)
    info = mne.pick_info(raw.info, sel=picks)

    # find relevant time samples
    time_samples = find_index(raw.times, times)

    # pick only data channels (currently only eeg)
    picks = mne.pick_types(raw.info, eeg=True, meg=False)
    info = mne.pick_info(raw.info, sel=picks)

    # find relevant time samples and select data
    time_samples = np.array(find_index(raw.times, times))
    data_slices = raw._data[picks[:, np.newaxis], time_samples[np.newaxis, :]] - \
        raw._data[picks, :].mean(axis=1)[:, np.newaxis] # remove DC by default

    fig, axes = plt.subplots(ncols=len(times), squeeze=False)
    minmax = np.abs([data_slices.min(), data_slices.max()]).max()
    for i, ax in enumerate(axes.ravel()):
        mne.viz.plot_topomap(data_slices[:, i], info, axes=ax,
                             vmin=-minmax, vmax=minmax)
        ax.set_title('{} s'.format(times[i]))
    return fig
