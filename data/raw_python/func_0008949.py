def is_edge(obj, shape):
    """
    Check if a 2d object is on the edge of the array.

    Parameters
    ----------
    obj : tuple(slice, slice)
        Pair of slices (e.g. from scipy.ndimage.measurements.find_objects)
    shape : tuple(int, int)
        Array shape.

    Returns
    -------
    b : boolean
        True if the object touches any edge of the array, else False.
    """

    if obj[0].start == 0: return True
    if obj[1].start == 0: return True
    if obj[0].stop == shape[0]: return True
    if obj[1].stop == shape[1]: return True
    return False