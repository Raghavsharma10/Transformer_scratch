def calculate_fft(data, tbin):
    """
    Function to calculate the Fourier transform of data.
    
    
    Parameters
    ----------
    data : numpy.ndarray
        1D or 2D array containing time series.
    tbin : float
        Bin size of time series (in ms).
    
    
    Returns
    -------
    freqs : numpy.ndarray
        Frequency axis of signal in Fourier space.         
    fft : numpy.ndarray
        Signal in Fourier space.
        
    """
    if len(np.shape(data)) > 1:
        n = len(data[0])
        return np.fft.fftfreq(n, tbin * 1e-3), np.fft.fft(data, axis=1)
    else:
        n = len(data)
        return np.fft.fftfreq(n, tbin * 1e-3), np.fft.fft(data)