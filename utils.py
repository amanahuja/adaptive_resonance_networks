"""
Utilities
used by various ART modules
"""

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np


def preprocess_data(data):
    """
    Convert to numpy array
    Convert to 1s and 0s

    """
    # Get useful information from first row
    if data[0] is not None:
        irow = data[0]

        # get size
        idat_size = len(irow)

        # get unique characters
        chars = False
        while not chars:
            chars = get_unique_chars(irow, reverse=True)
        char1, char2 = chars

    outdata = []
    idat = np.zeros(idat_size, dtype=bool)

    # Convert to boolean using the chars identified
    for irow in data:
        assert len(irow) == idat_size, "data row lengths not consistent"
        idat = [x == char1 for x in irow]
        # note: idat is a list of bools
        idat = list(np.array(idat).astype(int))
        outdata.append(idat)

    outdata = np.array(outdata)
    return outdata.astype(int)


def get_unique_chars(irow, reverse=False):
    """
    Get unique characters in data
    Helper function
    ----
    reverse:   bool
        Reverses order of the two chars returned
    """
    chars = Counter(irow)
    if len(chars) > 2:
        raise Exception("Data is not binary")
    elif len(chars) < 2:
        # first row doesn't contain both chars
        return False, False

    # Reorder here?
    if reverse:
        char2, char1 = chars.keys()
    else:
        char1, char2 = chars.keys()

    return char1, char2


def display_single_png(raw, ax=None, pred=None):
    """raw data is from 10x10 png
    """
    raw = raw.reshape(10, 10)

    if ax is None:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(1, 1))
        display_canvas = True
    else:
        # ax provided
        display_canvas = False
    ax.imshow(raw, cmap='Greys',  interpolation='nearest')
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if pred is not None:
        ax.set_title("prediction: {}".format(pred))

    if display_canvas:
        plt.show()
    return ax


def display_all_png(data):

    # Construct canvas
    # n_axes = data.shape[0]
    nrows = int(round(data.shape[0] / 5)) + 1
    ncols = 5
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5, 5))

    if not isinstance(axes, np.ndarray):
        axes = [axes]

    # plot each in data
    for idx, data_row in enumerate(data):
        canvas_row = idx / 5
        canvas_col = idx % 5

        assert isinstance(data_row, np.ndarray)
        display_single_png(data_row, axes[canvas_row][canvas_col])

    for idx in range(data.shape[0], nrows*ncols):
        # extra axes in canvas. Clean these.
        canvas_row = idx / 5
        canvas_col = idx % 5
        ax = axes[canvas_row][canvas_col]
        ax.set_xticklabels([])
        ax.set_yticklabels([])
