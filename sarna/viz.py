import numpy as np
import matplotlib.pyplot as plt

from borsar.viz import Topo, heatmap, color_limits, add_colorbar_to_axis
from borsar.utils import find_range, find_index, get_info
from borsar.channels import get_ch_pos

from .utils import group


def get_spatial_colors(inst):
    '''Get mne-style spatial colors for given mne object instance.'''
    from mne.viz.evoked import _rgb
    x, y, z = get_ch_pos(inst).T
    return _rgb(x, y, z)


def get_color_cycle():
    return plt.rcParams['axes.prop_cycle'].by_key()['color']


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


# # for Topo, setting channel props:
# for ch in ch_ind:
#     self.chans[ch].set_color('white')
#     self.chans[ch].set_radius(0.01)
#     self.chans[ch].set_zorder(4)


# TODO - [ ] ! missing imports !: OffsetImage, sample and AnnotationBbox
# #    - [ ] consider moving selection out to some simple interface
#            with .__init__ and .next()?
#      - [ ] or maybe just use np.random.choice
#      - [ ] change zoom to size
#      - [ ] add 'auto' zoom / size
def imscatter(x, y, images, ax=None, zoom=1, selection='random'):
    '''
    Plot images as scatter points. Puppy scatter, anyone?

    modified version of this stack overflow answer:
    https://stackoverflow.com/questions/22566284/matplotlib-how-to-plot-images-instead-of-points

    FIXME : add docs
    '''
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


def highlight(x_values=None, highlight=None, color=None, alpha=1., bottom_bar=False,
              bar_color='black', bottom_extend=True, ax=None):
    '''Highlight ranges along x axis.

    Parameters
    ----------
    x_values : numpy array
        Values specifying x axis points along which highlight operates.
    highlight : slice | list of slices | numpy array | list of numpy arrays
        Slice or boolean numpy array defining which values in ``x_values``
        should be highlighted. If list of slices or numpy arrays is provided
        each list element will be used to create a separate highlight.
    color : str | list | numpy array, optional
        Highlight patch color in format understood by matplotlib. The default
        is ``'orange'``.
    alpha : float
        Highlight patch transparency. ``1.`` (opaque) by default.
    bottom_bar : bool
        Whether to place an opaque highlight bar at the bottom of the figure.
    bar_color : str | list | numpy array, optional
        Bottom bar color in format understood by matplotlib. The default
        is ``'black'``.
    bottom_extend : bool
        Whether to extend the bottom of the axis before adding the bottom bar.
    ax : matplotlib Axes | None
        Highlight on an already present axis. Default is ``None`` which creates
        a new figure with one axis.

    Returns
    -------
    patches : list of matplotlib.patches.Patch
        List of highlight patches. If ``bottom_bar`` is ``True`` then each
        element of the list is a tuple of two patches: the first is the
        highlight patch and the second is the bottom bar patch.
    '''

    ax = plt.gca() if ax is None else ax
    x_values = _handle_x_values(ax, x_values)

    ylims = ax.get_ylim()
    y_rng = np.diff(ylims)[0]

    grp = _check_highlight_var(highlight)

    patch_low = ylims[0]
    if bottom_bar:
        bar_h = y_rng * 0.05
        bar_low = (ylims[0] - bar_h / 2 if bottom_extend
                   else ylims[0] + bar_h / 2)
        patch_low = bar_low + bar_h / 2

    patches = highlight_bar(
        x_values, grp, level=patch_low, height=None, color=color,
        alpha=alpha, ax=ax
    )

    if bottom_bar:
        bottom_patches = highlight_bar(
            x_values, grp, level=bar_low, height=bar_h,
            color=bar_color, alpha=1., ax=ax
        )
        patches = [(main, bottom) for main, bottom in
                    zip(patches, bottom_patches)]

    if bottom_bar and bottom_extend:
        ax.set_ylim((ylims[0] - bar_h, ylims[1]))

    return patches


def _check_highlight_var(highlight):
    if isinstance(highlight, list):
        if all([isinstance(x, slice) for x in highlight]):
            grp = highlight
        elif all([isinstance(x, np.ndarray) for x in highlight]):
            grp = [group(x, return_slice=True)[0] for x in highlight]
    elif isinstance(highlight, np.ndarray) and highlight.dtype == 'bool':
        grp = group(highlight, return_slice=True)
    elif isinstance(highlight, slice):
        grp = [highlight]
    else:
        raise TypeError('highlight must be slice, list of slices, '
                        'numpy boolean array or list of numpy boolean arrays')

    return grp


def _get_x_values_from_axis(ax):
    ax_lines = ax.findobj(plt.Line2D)

    if len(ax_lines) > 0:
        x_values = ax_lines[0].get_xdata()
    else:
        x_values = None

    return x_values


def _handle_x_values(ax, x_values):
    if x_values is None:
        x_values = _get_x_values_from_axis(ax)

    if x_values is None:
        raise ValueError('x_values must be provided if axis does not contain '
                         'any lines')

    return x_values


def highlight_bar(x_values=None, highlight=None, level=None, height=None,
                  color=None, alpha=1., ax=None):
    '''Highlight ranges along x axis.

    Parameters
    ----------
    x_values : numpy array
        Values specifying x axis points along which highlight operates.
    highlight : slice | list of slices | numpy array | list of numpy arrays
        Slice or boolean numpy array defining which values in ``x_values``
        should be highlighted. If list of slices or numpy arrays is provided
        each list element will be used to create a separate highlight.
    level : float | None
        Level at which to place the highlight. If ``None`` then the highlight
        will be placed at the bottom of the axis.
    height : float | None
        Height of the highlight. If ``None`` then the highlight will extend to
        the top of the axis.
    color : str | list | numpy array, optional
        Highlight patch color in format understood by matplotlib. The default
        is ``'orange'``.
    alpha : float
        Highlight patch transparency. 0.3 by default.
    ax : matplotlib Axes | None
        Highlight on an already present axis. Default is ``None`` which creates
        a new figure with one axis.

    Returns
    -------
    patches : list of matplotlib.patches.Patch
        List of highlight patches.
    '''
    from matplotlib.patches import Rectangle

    x_values = _handle_x_values(ax, x_values)

    # prepare path args
    color = [0.95] * 3 if color is None else color
    args = dict(lw=0, facecolor=color, alpha=alpha)
    if alpha == 1.:
        args['zorder'] = 0

    ax = plt.gca() if ax is None else ax
    grp = _check_highlight_var(highlight)

    # TODO: handle level and height in % of axis
    ylims = ax.get_ylim()
    y_rng = np.diff(ylims)[0]
    x_half_step = np.diff(x_values).mean() / 2

    patches = list()
    level = ylims[0] if level is None else level
    height = y_rng - (ylims[0] - level) if height is None else height

    for slc in grp:
        this_x = x_values[slc]
        start = this_x[0] - x_half_step
        length = np.diff(this_x[[0, -1]])[0] + x_half_step * 2

        patch = Rectangle((start, level), length, height, **args)
        ax.add_patch(patch)
        patches.append(patch)

    return patches


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


def plot_topomap_raw(raw, times=None):
    '''``plot_topomap`` for mne ``Raw`` objects.

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


# TODO - n_axes as total int and then optimal layout is chosen?
# FIXME - change space to be [x0, y0, w, h]
# ``h_dist`` and ``w_dist`` auto?
def prepare_equal_axes(fig, n_axes, space=[0.02, 0.98, 0.02, 0.98],
                       w_dist=0.025, h_dist=0.025):
    '''Prepare equal axes spanning given figure space.

    Very useful for creating a grid for topographic maps.

    Parameters
    ----------
    fig : matplotlib figure
        Figure to create grid of axes in.
    n_axes : tuple of ints
        Number of axes in each dimension.
    space : list of floats
        Space to use for axes. Should be in the form of
        ``[left, right, bottom, top]``.
    w_dist : float
        Distance between axes in the horizontal (width) direction.
    h_dist : float
        Distance between axes in the vertical (height) direction.

    Returns
    -------
    axes : numpy array of axes
        Array of axes handles.
    '''

    # transforms
    trans = fig.transFigure
    trans_inv = fig.transFigure.inverted()

    axes_space = np.array(space).reshape((2, 2)).T
    axes_space_disp_units = trans.transform(axes_space)
    space_h = np.diff(axes_space_disp_units[:, 1])[0]
    space_w = np.diff(axes_space_disp_units[:, 0])[0]

    w_dist, h_dist = trans.transform([w_dist, h_dist])

    h = (space_h - (h_dist * (n_axes[0] - 1))) / n_axes[0]
    w = (space_w - (w_dist * (n_axes[1] - 1))) / n_axes[1]

    # if too much width or height space of each axes - increase spacing
    # FIXME, ADD: other spacing options (for example align to top)
    if w > h:
        w_diff = w - h
        additional_w_dist = w_diff * n_axes[1] / (n_axes[1] - 1)
        w_dist += additional_w_dist
        w = h
    elif h > w:
        h_diff = h - w
        additional_h_dist = h_diff * n_axes[0] / (n_axes[0] - 1)
        h_dist += additional_h_dist
        h = w

    # start creating axes from bottom left corner,
    # then flipud the axis matrix
    axes = list()
    w_fig, h_fig = trans_inv.transform([w, h])
    for row_idx in range(n_axes[0]):
        row_axes = list()
        y0 = axes_space_disp_units[0, 1] + (h + h_dist) * row_idx
        _, y0 = trans_inv.transform([0, y0])
        for col_idx in range(n_axes[1]):
            x0 = axes_space_disp_units[0, 0] + (w + w_dist) * col_idx
            x0, _ = trans_inv.transform([x0, 0])
            ax = fig.add_axes([x0, y0, w_fig, h_fig])
            row_axes.append(ax)
        axes.append(row_axes)

    axes = np.flipud(np.array(axes))
    return axes


def equal_axes_grid(fig, n_axes, space=[0.02, 0.98, 0.02, 0.98],
                    w_dist=0.025, h_dist=0.025):
    '''Prepare equal axes spanning given figure space.

    Very useful for creating a grid for topographic maps.

    Parameters
    ----------
    fig : matplotlib figure
        Figure to create grid of axes in.
    n_axes : tuple of ints
        Number of axes in each dimension.
    space : list of floats
        Space to use for axes. Should be in the form of
        ``[left, right, bottom, top]``.
    w_dist : float
        Distance between axes in the horizontal (width) direction.
    h_dist : float
        Distance between axes in the vertical (height) direction.

    Returns
    -------
    axes : numpy array of axes
        Array of axes handles.
    '''
    return prepare_equal_axes(fig, n_axes, space, w_dist, h_dist)


# CONSIDER: plot small images instead of rectangles and scale their alpha by
# data density, idea similar to this matplotlib example:
# https://matplotlib.org/3.4.3/gallery/lines_bars_and_markers/gradient_bar.html
# https://stackoverflow.com/questions/42063542/mathplotlib-draw-triangle-with-gradient-fill
def glassplot(x=None, y=None, data=None, x_width=0.2, zorder=4,
              alpha=0.3, linewidth=2.5, ax=None):
    '''Plot transparent patches marking mean and standard error.

    Parameters
    ----------
    x : str
        Categories to plot on the x axis. Should be a valid column name of the
        ``data`` DataFrame.
    y : str
        Values to plot on the y axis. Should be a valid column name of the
        ``data`` DataFrame.
    data : DataFrame
        DataFrame with the data to plot.
    x_width : float
        Width of each of the patches as a fraction of the x axis distance
        between adjacent categories.
    zorder : int
        Z-order of the lines marking the average. The patches are drawn one
        z order below.
    alpha : float
        Transparency of the patches.
    linewidth : float
        Width of the lines marking the average.
    ax : matplotlib axes
        Axes to plot on. If None, a new figure is created.

    Returns
    -------
    ax : matplotlib axes
        Axes with the plot.
    '''
    import matplotlib as mpl
    from scipy.stats import sem
    import seaborn as sns

    assert data is not None
    assert x is not None
    assert y is not None

    categories = data.loc[:, x].unique()
    assert len(categories) < data.shape[0]

    new_ax = False
    colors = sns.utils.get_color_cycle()
    if ax is None:
        new_ax = True
        ax = plt.gca()
    else:
        swarms = ax.findobj(mpl.collections.PathCollection)
        if len(swarms) > 0:
            assert len(swarms) == len(categories)
            colors = [swarm.get_facecolor()[0] for swarm in swarms]
            colors = [color[:3] for color in colors]  # ignore alpha

    x_ticks = np.arange(len(categories))
    # TODO - if axis is passed, check x labels (order)
    # x_pos = ax.get_xticks()
    # x_lab = [x.get_text() for x in ax.get_xticklabels()]

    means = data.groupby(x).mean()
    width = np.diff(x_ticks)[0] * x_width

    if new_ax:
        ylm = [np.inf, -np.inf]

    for idx, (this_label, this_x) in enumerate(zip(categories, x_ticks)):
        # plot mean
        this_mean = means.loc[this_label, y]
        ax.plot([this_x - width, this_x + width], [this_mean, this_mean],
                color=colors[idx], lw=linewidth, zorder=zorder)

        # add CI (currently standard error of the mean)
        msk = data.loc[:, x] == this_label
        data_sel = data.loc[msk, y]
        this_sem = sem(data_sel.values)
        rct = plt.Rectangle((this_x - width, this_mean - this_sem),
                            width * 2, this_sem * 2, zorder=zorder - 1,
                            facecolor=colors[idx], alpha=alpha)
        ax.add_patch(rct)

    if new_ax:
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(categories)

    return ax


def connect_swarms(x=None, y=None, data=None, ax=None, color=None,
                   alpha=None):
    # based on the solution from stack overflow:'
    # https://stackoverflow.com/a/63171175/2225833
    import matplotlib as mpl

    assert data is not None
    assert ax is not None
    x = x if x is not None else ax.get_xlabel()
    y = y if y is not None else ax.get_ylabel()
    color = 'black' if color is None else color
    alpha = 0.1 if alpha is None else color

    swarms = ax.findobj(mpl.collections.PathCollection)
    dot_pos = [dots.get_offsets() for dots in swarms]

    categories = data.loc[:, x].unique()
    # categories = [label.get_text() for label in ax.get_xticklabels()]

    assert len(swarms) == len(categories)

    # before plotting, we need to sort so that the data points
    # correspond to each other as they did in "set1" and "set2"
    sortings = list()

    for cat in categories:
        msk = data.loc[:, x] == cat
        data_sel = data.loc[msk, y].values
        sortings.append(data_sel.argsort())

    prev_pos = dot_pos[0]
    lines = list()
    for ix in range(1, len(swarms)):
        # revert "ascending sort" using argsort() indices,
        # and then sort into order corresponding with set1
        this_pos = dot_pos[ix][sortings[ix].argsort()][sortings[0]]

        x_pos = np.stack([prev_pos[:, 0], this_pos[:, 0]], axis=0)
        y_pos = np.stack([prev_pos[:, 1], this_pos[:, 1]], axis=0)
        lines.append(ax.plot(x_pos, y_pos, color=color, alpha=alpha))

        prev_pos = this_pos

    return lines


# layout functions
# ----------------

# - [ ] TODO: add wrt (with respect to) and align (top/bottom etc)
#       so that change_position(axs[:3], xby=0.1, wrt=axs[3:])
#       would position the first 3 axes 0.1 above the next 3 axes
def change_axis_position(axs, x=None, y=None, xby=None, yby=None):
    '''Change axis position.

    Parameters
    ----------
    axs : list of axes
        Axes to change position of.
    x : float
        New x position of axes.
    y : float
        New y position of axes.
    xby : float
        Change x position by this value.
    yby : float
        Change y position by this value.
    '''
    pos = [ax.get_position().bounds for ax in axs]

    for ax, ps in zip(axs, pos):
        ps = list(ps)
        if x is not None:
            ps[0] = x
        if y is not None:
            ps[1] = y
        if xby is not None:
            ps[0] += xby
        if yby is not None:
            ps[1] += yby
        ax.set_position(ps)


def _get_pos_dist(axs):
    positions = [axs.get_position().bounds for ax in axs]

    distances = list()
    distances.append(positions[0][0])
    for ix in range(len(positions) - 1):
        pos = positions[ix]
        next_pos = positions[ix + 1]
        dist = next_pos[0] - (pos[0] + pos[2])
        distances.append(dist)

    # TODO: not clear why this is necessary...
    last_pos = positions[-1]
    distances.append(1 - (last_pos[0] + last_pos[2]))

    return positions, distances


def _align_along_x(axs, left=0, right=0):
    positions, distances = _get_pos_dist(axs)
    distances[-1] += left - right
    avg_dist = np.array(distances[1:]).mean()

    last = distances[0] - left
    for ax, pos in zip(axs, positions):
        pos = list(pos)
        pos[0] = last
        ax.set_position(pos)
        last += avg_dist + pos[2]


def rescale_axis(axs, x=None, y=None, xto='center', yto='center'):
    '''Shrink axes in x and/or y dimension by specific value / percent.
    Shrinking by a negative value extends the axis.

    Parameters
    ----------
    axs : list-like of axes
        Matplotlib axes to shrink.
    x : int | float | str
        Value to change the axis width by. Can also be a percent string
        for example ``'-25%'`` (which shrinks by 25% of current width).
    xto : str
        FIXME
    y : int | float | str
        Value to change the axis height by. Can also be a percent string
        for example ``'-25%'`` (which shrinks by 25% of current height).
    yto : str
        FIXME
    '''
    if x is None and y is None:
        # nothing to do
        return

    positions = [ax.get_position().bounds for ax in axs]
    scale_x, x_perc = _parse_perc(x) if x is not None else (None, None)
    scale_y, y_perc = _parse_perc(y) if y is not None else (None, None)

    for ax, pos in zip(axs, positions):
        if x is not None:
            pos = _scale_ax(pos, 0, scale_x, x_perc, xto)
        if y is not None:
            pos = _scale_ax(pos, 1, scale_y, y_perc, yto)
        ax.set_position(pos)


# TODO - refactor with borsar % parsing?
def _parse_perc(val):
    is_perc = False
    if isinstance(val, str) and '%' in val:
        val = float(val.replace('%', '')) / 100
        is_perc = True
    return val, is_perc


def _scale_ax(pos, ix, scale, is_perc, align_to):
    '''ix = 0 for x, 1 for y'''
    width = pos[2 + ix]
    change = scale * width if is_perc else scale
    new_width = width + change
    new_pos = (pos[0 + ix] if align_to == 'left' else
               pos[0 + ix] - change if align_to == 'right' else
               pos[0 + ix] - change / 2)

    pos = list(pos)
    pos[0 + ix] = new_pos
    pos[2 + ix] = new_width

    return pos


def _unify_lims(axs, xlim=False, ylim=False):
    lims = {'x': [np.inf, -np.inf], 'y': [np.inf, -np.inf]}
    for ax in axs:
        if xlim:
            xlm = ax.get_xlim()
            if xlm[0] < lims['x'][0]:
                lims['x'][0] = xlm[0]
            if xlm[1] > lims['x'][1]:
                lims['x'][1] = xlm[1]
        if ylim:
            ylm = ax.get_ylim()
            if ylm[0] < lims['y'][0]:
                lims['y'][0] = ylm[0]
            if ylm[1] > lims['y'][1]:
                lims['y'][1] = ylm[1]
    for ax in axs:
        if xlim:
            ax.set_xlim(lims['x'])
        if ylim:
            ax.set_ylim(lims['y'])


def _align_x_center(ax, source):
    pos = [x.get_position().bounds for x in [ax, source]]
    mids = [p[0] + p[2] * 0.5 for p in pos]
    mid_diff = mids[1] - mids[0]
    if not mid_diff == 0:
        axpos = list(pos[0])
        axpos[0] += mid_diff
        ax.set_position(axpos)