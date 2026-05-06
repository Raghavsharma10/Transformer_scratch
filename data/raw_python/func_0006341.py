def get_ranges_from_array(arr, append_last=True):
    '''Takes an array and calculates ranges [start, stop[. The last range end is none to keep the same length.

    Parameters
    ----------
    arr : array like
    append_last: bool
        If True, append item with a pair of last array item and None.

    Returns
    -------
    numpy.array
        The array formed by pairs of values by the given array.

    Example
    -------
    >>> a = np.array((1,2,3,4))
    >>> get_ranges_from_array(a, append_last=True)
    array([[1, 2],
           [2, 3],
           [3, 4],
           [4, None]])
    >>> get_ranges_from_array(a, append_last=False)
    array([[1, 2],
           [2, 3],
           [3, 4]])
    '''
    right = arr[1:]
    if append_last:
        left = arr[:]
        right = np.append(right, None)
    else:
        left = arr[:-1]
    return np.column_stack((left, right))