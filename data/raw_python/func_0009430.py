def _validI(x, y, weights):
    '''
    return indices that have enough data points and are not erroneous 
    '''
    # density filter:
    i = np.logical_and(np.isfinite(y), weights > np.median(weights))
    # filter outliers:
    try:
        grad = np.abs(np.gradient(y[i]))
        max_gradient = 4 * np.median(grad)
        i[i][grad > max_gradient] = False
    except (IndexError, ValueError):
        pass
    return i