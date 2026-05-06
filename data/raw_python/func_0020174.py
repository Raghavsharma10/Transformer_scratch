def Smooth(x, window_len=100, window='hanning'):
    '''
    Smooth data by convolving on a given timescale.

    :param ndarray x: The data array
    :param int window_len: The size of the smoothing window. Default `100`
    :param str window: The window type. Default `hanning`


    '''

    if window_len == 0:
        return np.zeros_like(x)
    s = np.r_[2 * x[0] - x[window_len - 1::-1],
              x, 2 * x[-1] - x[-1:-window_len:-1]]
    if window == 'flat':
        w = np.ones(window_len, 'd')
    else:
        w = eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='same')
    return y[window_len:-window_len + 1]