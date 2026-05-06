def GetChunk(time, breakpoints, b, mask=[]):
    '''
    Returns the indices corresponding to a given light curve chunk.

    :param int b: The index of the chunk to return

    '''

    M = np.delete(np.arange(len(time)), mask, axis=0)
    if b > 0:
        res = M[(M > breakpoints[b - 1]) & (M <= breakpoints[b])]
    else:
        res = M[M <= breakpoints[b]]
    return res