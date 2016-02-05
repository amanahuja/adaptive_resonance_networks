"""
Utilities
used by various ART modules
"""

from collections import Counter
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
