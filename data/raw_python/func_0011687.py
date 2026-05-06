def _plot_cwt(ts, coefs, freqs, tsize=1024, fsize=512):
    """Plot time resolved power spectral density from cwt results
    Args:
      ts: the original Timeseries
      coefs:  continuous wavelet transform coefficients as calculated by cwt()
      freqs: list of frequencies (in Hz) corresponding to coefs.
      tsize, fsize: size of the plot (time axis and frequency axis, in pixels)
    """
    import matplotlib.style
    import matplotlib as mpl
    mpl.style.use('classic')
    import matplotlib.pyplot as plt
    from scipy import interpolate
    channels = ts.shape[1]
    fig = plt.figure()
    for i in range(channels):
        rect = (0.1, 0.85*(channels - i - 1)/channels + 0.1, 
                0.8, 0.85/channels)
        ax = fig.add_axes(rect)
        logpowers = np.log((coefs[:, :, i] * coefs[:, :, i].conj()).real)
        tmin, tmax = ts.tspan[0], ts.tspan[-1]
        fmin, fmax = freqs[0], freqs[-1]
        tgrid, fgrid = np.mgrid[tmin:tmax:tsize*1j, fmin:fmax:fsize*1j]
        gd = interpolate.interpn((ts.tspan, freqs), logpowers, 
                                 (tgrid, fgrid)).T
        ax.imshow(gd, cmap='gnuplot2', aspect='auto', origin='lower',
                   extent=(tmin, tmax, fmin, fmax))
        ax.set_ylabel('freq (Hz)')
    fig.axes[0].set_title(u'log(power spectral density)')
    fig.axes[channels - 1].set_xlabel('time (s)')
    fig.show()