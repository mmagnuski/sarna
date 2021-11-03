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


# - [ ] support list/tuple of slices for which_highlight?
# - [ ] `level` and `height` are unused but should allow for highlight that
#       takes only a fraction of the axis
#       kind='patch', level=0.04, height=0.03
def highlight(x_values, highlight, color=None, alpha=0.3, bottom_bar=False,
              bottom_extend=True, axis=None):
    '''Highlight ranges along x axis.

    Parameters
    ----------
    x_values : numpy array
        Values specifying x axis points along which which_highlight operates.
    highlight : slice | numpy array
        Slice or boolean numpy array defining which values in ``x_values``
        should be highlighted.
    color : str | list | numpy array, optional
        Color in format understood by matplotlib. The default is 'orange'.
    alpha : float
        Highlight patch transparency. 0.3 by default.
    bottom_bar : bool
        Whether to place a highlight bar at the bottom of the figure.
    axis : matplotlib Axes | None
        Highlight on an already present axis. Default is ``None`` which creates
        a new figure with one axis.
    '''
    from matplotlib.patches import Rectangle

    color = 'orange' if color is None else color
    axis = plt.gca() if axis is None else axis

    ylims = axis.get_ylim()
    y_rng = np.diff(ylims)[0]
    hlf_dist = np.diff(x_values).mean() / 2

    if isinstance(highlight, (np.ndarray, list)):
        grp = group(highlight, return_slice=True)
    elif isinstance(highlight, slice):
        grp = [highlight]

    args = dict(lw=0, facecolor=color, alpha=alpha)
    if alpha == 1.:
        args['zorder'] = 0

    for slc in grp:
        this_x = x_values[slc]
        start = this_x[0] - hlf_dist
        length = np.diff(this_x[[0, -1]])[0] + hlf_dist * 2
        ptch = Rectangle((start, ylims[0]), length, y_rng, **args)
        axis.add_patch(ptch)

        if bottom_bar:
            bar_h = y_rng * 0.05
            ptch = Rectangle((start, ylims[0] - bar_h / 2), length, bar_h, lw=0,
                             facecolor='k', alpha=1.)
            axis.add_patch(ptch)

    if bottom_bar:
        axis.set_ylim((ylims[0] - bar_h * 1, ylims[1]))


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


# - [ ] remove and add heatmap options to borsar.Cluster.plot()
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


def prepare_equal_axes(fig, n_axes, space=[0.02, 0.98, 0.02, 0.98],
                       w_dist=0.025, h_dist=0.025):
    '''Prepare equal axes spanning given figure space. FIXME docs'''

    # transforms
    trans = fig.transFigure
    trans_inv = fig.transFigure.inverted()

    # FIXME - change space to be [x0, y0, w, h]
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
