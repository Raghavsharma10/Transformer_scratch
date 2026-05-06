def Downbin(x, newsize, axis=0, operation='mean'):
    '''
    Downbins an array to a smaller size.

    :param array_like x: The array to down-bin
    :param int newsize: The new size of the axis along which to down-bin
    :param int axis: The axis to operate on. Default 0
    :param str operation: The operation to perform when down-binning. \
           Default `mean`
    '''

    assert newsize < x.shape[axis], \
        "The new size of the array must be smaller than the current size."
    oldsize = x.shape[axis]
    newshape = list(x.shape)
    newshape[axis] = newsize
    newshape.insert(axis + 1, oldsize // newsize)
    trim = oldsize % newsize
    if trim:
        xtrim = x[:-trim]
    else:
        xtrim = x

    if operation == 'mean':
        xbin = np.nanmean(xtrim.reshape(newshape), axis=axis + 1)
    elif operation == 'sum':
        xbin = np.nansum(xtrim.reshape(newshape), axis=axis + 1)
    elif operation == 'quadsum':
        xbin = np.sqrt(np.nansum(xtrim.reshape(newshape) ** 2, axis=axis + 1))
    elif operation == 'median':
        xbin = np.nanmedian(xtrim.reshape(newshape), axis=axis + 1)
    else:
        raise ValueError("`operation` must be either `mean`, " +
                         "`sum`, `quadsum`, or `median`.")

    return xbin