def ascolumn(x, dtype = None):
    '''Convert ``x`` into a ``column``-type ``numpy.ndarray``.'''
    x = asarray(x, dtype)
    return x if len(x.shape) >= 2 else x.reshape(len(x),1)