import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from sarna.viz import highlight


def compare_box_and_slice(x, box, slc):
    '''Function used to compare slice limits and box range of a rectangle.'''
    halfsample = np.diff(x).mean() / 2
    correct_limits = x[slc][[0, -1]] + [-halfsample, halfsample]

    bbox_limits = box.get_bbox().get_points()
    bbox_x_limits = bbox_limits[:, 0]
    print(bbox_x_limits)
    print(correct_limits)

    return np.allclose(correct_limits, bbox_x_limits)


def test_highlight():
    x = np.arange(0, 10, step=0.05)
    n_times = len(x)

    y = np.random.random(n_times)

    # simple usage
    # ------------
    line = plt.plot(x, y)
    highlight(x, slice(10, 40))

    ax = line[0].axes
    rectangles = ax.findobj(Rectangle)
    assert len(rectangles) == 2
    plt.close(ax.figure)

    # two slices, setting color and alpha
    # -----------------------------------
    line = plt.plot(x, y)
    use_alpha, use_color = 0.5, [0.75] * 3
    slices = [slice(10, 40), slice(60, 105)]
    highlight(x, slices, alpha=use_alpha, color=use_color)

    ax = line[0].axes
    rectangles = ax.findobj(Rectangle)
    assert len(rectangles) == 3

    # check box color and box alpha
    rgba = rectangles[0].get_facecolor()
    assert (rgba[:3] == np.array(use_color)).all()
    assert rgba[-1] == use_alpha

    # compare slices and rectangles:
    for box, slc in zip(rectangles, slices):
        assert compare_box_and_slice(x, box, slc)
    plt.close(ax.figure)

    # two slices, using bottom_bar
    # ----------------------------
    line = plt.plot(x, y)

    slices = [slice(10, 40), slice(60, 105)]
    highlight(x, slices, bottom_bar=True)

    ax = line[0].axes
    rectangles = ax.findobj(Rectangle)
    assert len(rectangles) == 5

    for idx, col in enumerate([0.95, 0, 0.95, 0]):
        rect_color = rectangles[idx].get_facecolor()[:3]
        assert (rect_color == np.array([col] * 3)).all()
    plt.close(ax.figure)
