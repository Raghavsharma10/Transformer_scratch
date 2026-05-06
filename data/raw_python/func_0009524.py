def edgesFromBoolImg(arr, dtype=None):
    '''
    takes a binary image (usually a mask)
    and returns the edges of the object inside
    '''
    out = np.zeros_like(arr, dtype=dtype)
    _calc(arr, out)
    _calc(arr.T, out.T)
    return out