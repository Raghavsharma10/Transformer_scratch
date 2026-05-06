def fft_freqs(n_fft, fs):
    """Return frequencies for DFT

    Parameters
    ----------
    n_fft : int
        Number of points in the FFT.
    fs : float
        The sampling rate.
    """
    return np.arange(0, (n_fft // 2 + 1)) / float(n_fft) * float(fs)