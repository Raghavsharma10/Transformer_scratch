def _get_valid_indices(shape, ix0, ix1, iy0, iy1):
    """Give array shape and desired indices, return indices that are
    correctly bounded by the shape."""
    ymax, xmax = shape

    if ix0 < 0:
        ix0 = 0
    if ix1 > xmax:
        ix1 = xmax
    if iy0 < 0:
        iy0 = 0
    if iy1 > ymax:
        iy1 = ymax

    if iy1 <= iy0 or ix1 <= ix0:
        raise IndexError(
            'array[{0}:{1},{2}:{3}] is invalid'.format(iy0, iy1, ix0, ix1))

    return list(map(int, [ix0, ix1, iy0, iy1]))