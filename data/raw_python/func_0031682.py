def decimate(x, q=10, n=4, k=0.8, filterfun=ss.cheby1):
    """
    scipy.signal.decimate like downsampling using filtfilt instead of lfilter,
    and filter coeffs from butterworth or chebyshev type 1.


    Parameters
    ----------
    x : numpy.ndarray
        Array to be downsampled along last axis.
    q : int 
        Downsampling factor.
    n : int
        Filter order.
    k : float
        Aliasing filter critical frequency Wn will be set as Wn=k/q.
    filterfun : function
        `scipy.signal.filter_design.cheby1` or
        `scipy.signal.filter_design.butter` function

    Returns
    -------
    numpy.ndarray
        Array of downsampled signal.
              
    """
    if not isinstance(q, int):
        raise TypeError("q must be an integer")

    if n is None:
        n = 1

    if filterfun == ss.butter:
        b, a = filterfun(n, k / q)
    elif filterfun == ss.cheby1:
        b, a = filterfun(n, 0.05, k / q)
    else:
        raise Exception('only ss.butter or ss.cheby1 supported')

    try:
        y = ss.filtfilt(b, a, x)
    except: # Multidim array can only be processed at once for scipy >= 0.9.0
        y = []
        for data in x:
            y.append(ss.filtfilt(b, a, data))
        y = np.array(y)

    try:
        return y[:, ::q]
    except:
        return y[::q]