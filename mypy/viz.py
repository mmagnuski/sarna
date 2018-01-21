import numpy as np
import matplotlib.pyplot as plt
from mypy.utils import get_info, find_index, find_range

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


def masked_image(img, mask=None, alpha=0.75, mask_color=(0.5, 0.5, 0.5),
                 axis=None, **imshow_kwargs):
    defaults = {'interpolation': 'none', 'origin': 'lower'}
    defaults.update(imshow_kwargs)

    if axis is None:
        fig, axis = plt.subplots()

    # plot images
    main_img = axis.imshow(img, **defaults)
    if mask is not None:
        mask_img = add_image_mask(mask, alpha=alpha, mask_color=mask_color,
                                  axis=axis, **defaults)
        return main_img, mask_img
    else:
        return main_img


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


# TODO# check in plot_sensors how button_press_event select relevant chan labels

# SpectralPlot - can work from here and make it a little more universal

# this pacplot can be used with partial to couple with some figure
def pacplot(ch_ind=None, fig=None):
    if ch_ind is None:
        ch_ind = [eeg.ch_names.index(ch) for ch in fig.lasso.selection]
    im = t_ef[ch_ind, :, :].mean(axis=0).T
    mask = np.abs(im) > 2.
    fig, ax = plt.subplots()
    masked_image(im, mask, origin='lower', vmin=-3, vmax=3)

# this on_pick can be used with partial to couple with some figure
def on_pick(event, fig=None):
    if event.mouseevent.key == 'control' and fig.lasso is not None:
         for ind in event.ind:
             fig.lasso.select_one(event.ind)

         return
    pacplot(ch_ind=event.ind)

# fig.canvas.mpl_connect('pick_event', on_pick)
# fig.canvas.mpl_connect('lasso_event', pacplot)


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
# - [ ] check why remove_levels didn't work
# - [ ] check out psychic.scalpplot.plot_scalp for slightly different topo plots
#       https://github.com/wmvanvliet/psychic/blob/master/psychic/scalpplot.py
#       (there is also eelbrain)
class Topo(object):
    '''High-level object that allows for convenient topographic plotting.

    Parameters
    ----------
    values : numpy array
        Values to topographically plot.
    info : mne Info instance
        Info object containing channel positions.
    **kwargs : any additional keyword arguments
        Additional keyword arguments are passed to mne.viz.plot_topomap

    Returns
    -------
    topo : mypy.viz.Topo instance
        Topo object that exposes various useful methods like `remove_levels`
        or `mark_channels`.

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

        # check channel positions - some (older?) versions use scatter so
        # the channels are marked with `mpl.patches.Circle` but at other times
        # `mpl.collections.PathCollection` is being used.
        circles = im.axes.findobj(mpl.patches.Circle)
        if len(circles) == 0:
            # look for PathCollection
            path_collection = im.axes.findobj(mpl.collections.PathCollection)
            if len(path_collection) > 0:
                self.chans = path_collection[0]
                self.chan_pos = self.chans.get_offsets()
            else:
                raise RuntimeError('Could not find matplotlib objects '
                                   'representing channels. Looked for '
                                   '`matplotlib.patches.Circle` and '
                                   '`matplotlib.collections.PathCollection`.')
        else:
            self.chans = circles
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
        '''Highlight specified channels with markers.

        Parameters
        ----------
        chans : numpy array
            Channels to highlight. Integer array with channel indices or
            boolean array of shape (n_channels, ).
        **kwargs
            Any additional keyword arguments are passed as arguments to
            plt.plot. It is useful for defining marker properties like
            `markerfacecolor` or `markersize`.
        '''
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


def color_limits(data):
    vmax = np.abs([np.nanmin(data), np.nanmax(data)]).max()
    return -vmax, vmax


# - [ ] enhance Topo with that functionality
def selected_Topo(values, info, indices, replace='zero', **kawrgs):
    # if a different info is passed - compare and
    # fill unused channels with 0
    ch_num = len(info['ch_names'])

    if replace == 'zero':
        vals = np.zeros(ch_num)
    elif replace == 'min':
        vals = np.ones(ch_num) * min(values)
    elif replace == 'max':
        vals = np.ones(ch_num) * max(values)

    vals[indices] = values

    # topoplot
    tp = Topo(vals, info, show=False, **kawrgs)

    # make all topography lines solid
    tp.solid_lines()
    tp.remove_levels(0.)

    # final touches
    tp.fig.set_facecolor('white')

    return tp


# TODOs:
# create_contour:
# - [x] let it work for all the shapes (not only the first one completed)
# - [x] fix issue with shapes that touch matrix edge
# - [x] add function that corrects for image extent
# - [ ] rename to create_contour
# - [ ] docstring
#
# separate cluter_contour?:
# - [ ] cluster mode (returns a list or dict mapping cluster ids to list of
#       cluster contours) - so that each cluster can be marked by a different
#       color.
# - [ ] one convolution for all clusters
def create_cluster_contour(mask, extent=None):
    from scipy.ndimage import correlate

    orig_mask_shape = mask.shape
    mask_int = np.pad(mask.astype('int'), ((1, 1), (1, 1)), 'constant')
    kernels = {'upper': np.array([[-1], [1], [0]]),
               'lower': np.array([[0], [1], [-1]]),
               'left': np.array([[-1, 1, 0]]),
               'right': np.array([[0, 1, -1]])}
    lines = {k: (correlate(mask_int, v) == 1).astype('int')
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
    upper_lines = np.where(lines['upper'])
    outlines = list()

    while len(upper_lines[0]) > 0:
        current_index = np.array([x[0] for x in upper_lines])
        closed_shape = False
        current_edge = 'upper'
        edge_points = [tuple(current_index + [-0.5, -0.5])]
        direction = movement_direction[current_edge]

        while not closed_shape:
            new_edge = None
            ind = tuple(current_index)

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
        # TODO: this should be done at runtime
        x = np.array([l[1] for l in edge_points])
        y = np.array([l[0] for l in edge_points])
        outlines.append([x, y])
        upper_lines = np.where(lines['upper'] > 0)
    _correct_all_outlines(outlines, orig_mask_shape, extent=extent)
    return outlines


def _correct_all_outlines(outlines, orig_mask_shape, extent=None):
    if extent is not None:
        orig_ext = [-0.5, orig_mask_shape[1] - 0.5,
                    -0.5, orig_mask_shape[0] - 0.5]
        orig_ranges = [orig_ext[1] - orig_ext[0],
                       orig_ext[3] - orig_ext[2]]
        ext_ranges = [extent[1] - extent[0],
                       extent[3] - extent[2]]
        scales = [ext_ranges[0] / orig_ranges[0],
                  ext_ranges[1] / orig_ranges[1]]

    def find_successive(vec):
        vec = vec.astype('int')
        two_consec = np.where((vec[:-1] + vec[1:]) == 2)[0]
        return two_consec

    for current_outlines in outlines:
        x_lim = (0, orig_mask_shape[1])
        y_lim = (0, orig_mask_shape[0])

        x_above = current_outlines[0] > x_lim[1]
        x_below = current_outlines[0] < x_lim[0]
        y_above = current_outlines[1] > y_lim[1]
        y_below = current_outlines[1] < y_lim[0]

        x_ind, y_ind = list(), list()
        for x in [x_above, x_below]:
            x_ind.append(find_successive(x))
        for y in [y_above, y_below]:
            y_ind.append(find_successive(y))

        all_ind = np.concatenate(x_ind + y_ind)

        if len(all_ind) > 0:
            current_outlines[1] = np.insert(current_outlines[1],
                                            all_ind + 1, np.nan)
            current_outlines[0] = np.insert(current_outlines[0],
                                            all_ind + 1, np.nan)
        # compensate for padding
        current_outlines[0] = current_outlines[0] - 1.
        current_outlines[1] = current_outlines[1] - 1.

        if extent is not None:
            current_outlines[0] = ((current_outlines[0] + 0.5) * scales[0]
                                   + extent[0])
            current_outlines[1] = ((current_outlines[1] + 0.5) * scales[1]
                                   + extent[2])


# TODO - [ ] consider moving selection out to some simple interface
#            with .__init__ and .next()
#      - [ ] or maybe just use np.random.choice
#      - [ ] change zoom to size
#      - [ ] add 'auto' zoom
def imscatter(x, y, images, ax=None, zoom=1, selection='random'):
    '''
    Plot images as scatter points. Puppy scatter, anyone?

    modified version of this stack overflow answer:
    https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points'''
    if ax is None:
        ax = plt.gca()

    if isinstance(images, str): images = [images]
    if isinstance(images, list) and isinstance(images[0], str):
        images = [plt.imread(image) for image in images]

    if not isinstance(zoom, list):
        zoom = [zoom] * len(images)

    im = [OffsetImage(im, zoom=zm) for im, zm in zip(images, zoom)]
    x, y = np.atleast_1d(x, y)

    artists = []
    img_idx = list(range(len(im)))
    sel_idx = img_idx.copy()

    for idx, (x0, y0) in enumerate(zip(x, y)):
        # refill sel_idx if empty
        if 'replace' not in selection and len(sel_idx) < 1:
            sel_idx = img_idx.copy()

        # select image index
        if 'random' in selection:
            take = sample(range(len(sel_idx)), 1)[0]
        elif 'replace' in selection:
            take = idx % len(im)
        else:
            take = 0

        if 'replace' not in selection:
            im_idx = sel_idx.pop(take)
        else:
            im_idx = sel_idx[take]

        # add image to axis
        ab = AnnotationBbox(im[im_idx], (x0, y0),
                            xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))

    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale() # this may not be needed
    return artists


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


# - [ ] test a little and change the API and options
def significance_bar(start, end, height, displaystring, lw=0.1,
                     markersize=7, boxpad=-1.2, fontsize=14, color='k'):
    from matplotlib.markers import TICKDOWN
    # draw a line with downticks at the ends
    plt.plot([start, end], [height] * 1, '-', color=color, lw=lw,
             marker=TICKDOWN, markeredgewidth=linewidth, markersize=markersize)
    # draw the text with a bounding box covering up the line
    bbox_dict = dict(facecolor='0.', edgecolor='none',
                     boxstyle='Square,pad=' + str(boxpad))
    plt.text(-1.4 * (start + end), height, displaystring, ha='center',
             va='center', bbox=bbox_dict, size=fontsize)

    # another way:
    # x0, x1 = 1, 2
    # y, h, col = tips['total_bill'].max() + 1, 1, 'k'
    # plt.plot([x0, x0, x1, x1], [y, y+h, y+h, y], lw=0.4, c=col)
    # plt.text((x0+x1)*.4, y+h, "ns", ha='center', va='bottom', color=col)


# - [ ] cover some classical cases:
#       * time-chan
#       * time-freq
#       * chan-freq
def plot_cluster_heatmap(values, mask=None, axis=None, x_axis=None,
                         y_axis=None, outlines=False, colorbar=True,
                         line_kwargs=dict(), ch_names=None, freq=None):
    n_channels = values.shape[0]
    if x_axis is None and freq is not None:
        x_axis = freq

    heatmap(values, mask=mask, axis=axis, x_axis=x_axis, y_axis=y_axis,
            outlines=outlines, colorbar=True, line_kwargs=dict())

    if ch_names is not None:
        plt.yticks(np.arange(len(ch_names)) + 0.5, ch_names);
        for tick in plt.gca().yaxis.get_major_ticks():
            tick.label.set_fontsize(8)


# - [ ] cmap support
# - [ ] multiple masks, multiple alpha, multiple outline_colors
def heatmap(array, mask=None, axis=None, x_axis=None, y_axis=None,
            outlines=False, colorbar=True, line_kwargs=dict()):
    vmin, vmax = color_limits(array)
    n_rows, n_cols = array.shape

    x_axis = np.arange(n_cols) if x_axis is None else x_axis
    y_axis = np.arange(n_rows) if y_axis is None else y_axis

    # set extents
    x_step = np.diff(x_axis)[0]
    y_step = np.diff(y_axis)[0]
    ext = [*(x_axis[[0, -1]] + [-x_step / 2, x_step / 2]),
           *(y_axis[[0, -1]] + [-y_step / 2, y_step / 2])]


    out = masked_image(array, mask=mask, vmin=vmin, vmax=vmax,
                       cmap='RdBu_r', aspect='auto', extent=ext,
                       interpolation='nearest', origin='lower',
                       axis=axis)
    if mask is None:
        axis = out
    else:
        axis = out[0]

    # add outlines if necessary
    if outlines:
        if 'color' not in line_kwargs.keys():
            line_kwargs['color'] = 'w'
        outlines = create_cluster_contour(mask, extent=ext)
        for x_line, y_line in outlines:
            plt.plot(x_line, y_line, **line_kwargs)

    # plt.xlabel('Frequency', fontsize=14)
    # plt.ylabel('Channels', fontsize=14)
    # plt.title('{}'.format(format_pvalue(pval[cluster_id])))

    if colorbar:
        cbar = plt.colorbar(axis)
        # cbar.set_label('t values')
        return axis, cbar
    else:
        return axis


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
    data_slices = raw._data[picks[:, np.newaxis], time_samples[np.newaxis, :]]
    #- raw._data[picks, :].mean(axis=1)[:, np.newaxis] # remove DC by default

    fig, axes = plt.subplots(ncols=len(times), squeeze=False)
    minmax = np.abs([data_slices.min(), data_slices.max()]).max()
    for i, ax in enumerate(axes.ravel()):
        mne.viz.plot_topomap(data_slices[:, i], info, axes=ax,
                             vmin=-minmax, vmax=minmax)
        ax.set_title('{} s'.format(times[i]))
    return fig
