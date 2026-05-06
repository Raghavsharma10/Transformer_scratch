def _capture(f, t, t0, factor):
    '''
    capture signal and return its standard deviation
    #TODO: more detail
    '''
    n_per_sec = len(t) / t[-1]

    # len of one split:
    n = int(t0 * factor * n_per_sec)
    s = len(f) // n
    m = s * n
    f = f[:m]
    ff = np.split(f, s)
    m = np.mean(ff, axis=1)

    return np.std(m)