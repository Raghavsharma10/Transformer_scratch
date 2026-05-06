def movav(y, Dx, dx):
    """
    Moving average rectangular window filter:
    calculate average of signal y by using sliding rectangular
    window of size Dx using binsize dx
    
    
    Parameters
    ----------
    y : numpy.ndarray
        Signal
    Dx : float
        Window length of filter.
    dx : float
        Bin size of signal sampling.
                
    
    Returns
    -------
    numpy.ndarray
        Filtered signal.
    
    """
    if Dx <= dx:
        return y
    else:
        ly = len(y)
        r = np.zeros(ly)
        n = np.int(np.round((Dx / dx)))
        r[0:np.int(n / 2.)] = 1.0 / n
        r[-np.int(n / 2.)::] = 1.0 / n
        R = np.fft.fft(r)
        Y = np.fft.fft(y)
        yf = np.fft.ifft(Y * R)
        return yf