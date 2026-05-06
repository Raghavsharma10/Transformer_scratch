def process_vlen(data_header, array):
    """Process vlen coming back from NCStream v2.

    This takes the array of values and slices into an object array, with entries containing
    the appropriate pieces of the original array. Sizes are controlled by the passed in
    `data_header`.

    Parameters
    ----------
    data_header : Header
    array : :class:`numpy.ndarray`

    Returns
    -------
    ndarray
        object array containing sub-sequences from the original primitive array

    """
    source = iter(array)
    return np.array([np.fromiter(itertools.islice(source, size), dtype=array.dtype)
                     for size in data_header.vlens])