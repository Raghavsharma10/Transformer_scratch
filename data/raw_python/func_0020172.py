def Interpolate(time, mask, y):
    '''
    Masks certain elements in the array `y` and linearly
    interpolates over them, returning an array `y'` of the
    same length.

    :param array_like time: The time array
    :param array_like mask: The indices to be interpolated over
    :param array_like y: The dependent array

    '''

    # Ensure `y` doesn't get modified in place
    yy = np.array(y)
    t_ = np.delete(time, mask)
    y_ = np.delete(y, mask, axis=0)
    if len(yy.shape) == 1:
        yy[mask] = np.interp(time[mask], t_, y_)
    elif len(yy.shape) == 2:
        for n in range(yy.shape[1]):
            yy[mask, n] = np.interp(time[mask], t_, y_[:, n])
    else:
        raise Exception("Array ``y`` must be either 1- or 2-d.")
    return yy