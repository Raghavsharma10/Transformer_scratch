def circDiff(length, ary1, ary2):
    """calculate the circular difference between two paired arrays.
    This function will return the difference between pairs of numbers; however
    the difference that is output will be minimal in the sense that if we
    assume an array with length = 4: [0, 1, 2, 3], the difference between
    0 and 3 will not be 3, but 1 (i.e. circular difference)"""
    x = np.arange(length)
    mod = length % 2
    if mod == 0:
        temp = np.ones(length)
        temp[length/2:] = -1
    else:
        x = x - np.floor(length/2)
        temp = np.copy(x)
        temp[np.less(x, 0)] = 1
        temp[np.greater(x, 0)] = -1
    x = np.cumsum(temp)

    diagDiffmat = np.empty((length, length))
    for idx in np.arange(length):
        x = np.roll(x, 1)
        diagDiffmat[idx, :] = x
    # return diagDiffmat[ary1][ary2]
    flat = diagDiffmat.flatten()
    ind = ary1*diagDiffmat.shape[0] + ary2
    ind = ind.astype('int')
    return flat[ind]