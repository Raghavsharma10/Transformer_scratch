def _groups_of(length, total_length):
    """
    Return an iterator of tuples for slicing, in 'length' chunks.

    Parameters
    ----------
    length : int
        Length of each chunk.
    total_length : int
        Length of the object we are slicing

    Returns
    -------
    iterable of tuples
        Values defining a slice range resulting in length 'length'.

    """
    indices = tuple(range(0, total_length, length)) + (None, )
    return _pairwise(indices)