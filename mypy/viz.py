import numpy as np
import matplotlib.pyplot as plt

# TODOs:
# MultiDimView:
# - [ ] add colorbar
# - [ ] add topo view
# - [ ] clickable topo view
# - [ ] cickable (blockable) color bar
# - [ ] add window select


def masked_image(img, mask, alpha=0.75, mask_color=(0.5, 0.5, 0.5),
    **imshow_kwargs):
    defaults = {'interpolation': 'none', 'origin': 'lower'}
    defaults.update(imshow_kwargs)

    # create RGBA mask:
    mask_img = np.array(list(mask_color) + [0.]).reshape([1,1,4])
    mask_img = np.tile(mask_img, list(img.shape) + [1])

    # set alpha
    mask_img[np.logical_not(mask), -1] = alpha

    # plot images
    plt.imshow(img, **defaults)
    plt.imshow(mask_img, **defaults)


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
