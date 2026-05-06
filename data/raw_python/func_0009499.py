def closestDirectDistance(arr, ksize=30, dtype=np.uint16):
    '''
    return an array with contains the closest distance to the next positive
    value given in arr  within a given kernel size
    '''

    out = np.zeros_like(arr, dtype=dtype)
    _calc(out, arr, ksize)
    return out