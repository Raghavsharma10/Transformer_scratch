def size(array):
    """
    Return a human-readable description of the number of bytes required
    to store the data of the given array.

    For example::

        >>> array.nbytes
        14000000
        >> biggus.size(array)
        '13.35 MiB'

    Parameters
    ----------
    array : array-like object
        The array object must provide an `nbytes` property.

    Returns
    -------
    out : str
        The Array representing the requested mean.

    """
    nbytes = array.nbytes
    if nbytes < (1 << 10):
        size = '{} B'.format(nbytes)
    elif nbytes < (1 << 20):
        size = '{:.02f} KiB'.format(nbytes / (1 << 10))
    elif nbytes < (1 << 30):
        size = '{:.02f} MiB'.format(nbytes / (1 << 20))
    elif nbytes < (1 << 40):
        size = '{:.02f} GiB'.format(nbytes / (1 << 30))
    else:
        size = '{:.02f} TiB'.format(nbytes / (1 << 40))
    return size