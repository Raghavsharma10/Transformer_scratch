def coarseMaximum(arr, shape):
    '''
    return an array of [shape]
    where every cell equals the localised maximum of the given array [arr]
    at the same (scalled) position
    '''
    ss0, ss1 = shape
    s0, s1 = arr.shape

    pos0 = linspace2(0, s0, ss0, dtype=int)
    pos1 = linspace2(0, s1, ss1, dtype=int)

    k0 = pos0[0]
    k1 = pos1[0]

    out = np.empty(shape, dtype=arr.dtype)
    _calc(arr, out, pos0, pos1, k0, k1, ss0, ss1)
    return out