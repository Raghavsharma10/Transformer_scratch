def localizedMaximum(img, thresh=0, min_increase=0, max_length=0, dtype=bool):
    '''
    Returns the local maximum of a given 2d array


    thresh -> if given, ignore all values below that value

    max_length -> limit length between value has to vary  > min_increase

    >>> a = np.array([[0,1,2,3,2,1,0], \
                      [0,1,2,2,3,1,0], \
                      [0,1,1,2,2,3,0], \
                      [0,1,1,2,1,1,0],  \
                      [0,0,0,1,1,0,0]])

    >>> print localizedMaximum(a, dtype=int)
    [[0 1 1 1 0 1 0]
     [0 0 0 0 1 0 0]
     [0 0 0 1 0 1 0]
     [0 0 1 1 0 1 0]
     [0 0 0 1 0 0 0]]
    '''
    # because numba cannot create arrays:
    out = np.zeros(shape=img.shape, dtype=dtype)
    # first iterate all rows:
    _calc(img, out, thresh, min_increase, max_length)
    # that all columns:
    _calc(img.T, out.T, thresh, min_increase, max_length)
    return out