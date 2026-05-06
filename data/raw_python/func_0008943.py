def sortrows(a, i=0, index_out=False, recurse=True):
    """ Sorts array "a" by columns i

    Parameters
    ------------
    a : np.ndarray
        array to be sorted
    i : int (optional)
        column to be sorted by, taken as 0 by default
    index_out : bool (optional)
        return the index I such that a(I) = sortrows(a,i). Default = False
    recurse : bool (optional)
        recursively sort by each of the columns. i.e.
        once column i is sort, we sort the smallest column number
        etc. True by default.

    Returns
    --------
    a : np.ndarray
        The array 'a' sorted in descending order by column i
    I : np.ndarray (optional)
        The index such that a[I, :] = sortrows(a, i). Only return if
        index_out = True

    Examples
    ---------
    >>> a = array([[1,2],[3,1],[2,3]])
    >>> b = sortrows(a,0)
    >>> b
    array([[1, 2],
           [2, 3],
           [3, 1]])
    c, I = sortrows(a,1,True)

    >>> c
    array([[3, 1],
           [1, 2],
           [2, 3]])
    >>> I
    array([1, 0, 2])
    >>> a[I,:] - c
    array([[0, 0],
           [0, 0],
           [0, 0]])
    """
    I = np.argsort(a[:, i])
    a = a[I, :]
    # We recursively call sortrows to make sure it is sorted best by every
    # column
    if recurse & (len(a[0]) > i + 1):
        for b in np.unique(a[:, i]):
            ids = a[:, i] == b
            colids = range(i) + range(i+1, len(a[0]))
            a[np.ix_(ids, colids)], I2 = sortrows(a[np.ix_(ids, colids)],
                                                  0, True, True)
            I[ids] = I[np.nonzero(ids)[0][I2]]

    if index_out:
        return a, I
    else:
        return a