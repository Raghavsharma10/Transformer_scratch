def psd(t, data, window=None, n_band_average=1):
    """Compute one-sided power spectral density, subtracting mean automatically.

    Parameters
    ----------
    t : Time array
    data : Time series data
    window : {None, "Hanning"}
    n_band_average : Number of samples over which to band average

    Returns
    -------
    f : Frequency array
    psd : Spectral density array
    """
    dt = t[1] - t[0]
    N = len(data)
    data = data - np.mean(data)
    if window == "Hanning":
        data = data*np.hanning(N)
    f = np.fft.fftfreq(N, dt)
    y = np.fft.fft(data)
    f = f[0:N/2]
    psd = (2*dt/N)*abs(y)**2
    psd = np.real(psd[0:N/2])
    if n_band_average > 1:
        f_raw, s_raw = f*1, psd*1
        f = np.zeros(len(f_raw)//n_band_average)
        psd = np.zeros(len(f_raw)//n_band_average)
        for n in range(len(f_raw)//n_band_average):
            f[n] = np.mean(f_raw[n*n_band_average:(n+1)*n_band_average])
            psd[n] = np.mean(s_raw[n*n_band_average:(n+1)*n_band_average])
    return f, psd