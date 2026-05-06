def phase_histogram(dts, times=None, nbins=30, colormap=mpl.cm.Blues):
    """Plot a polar histogram of a phase variable's probability distribution
    Args:
      dts: DistTimeseries with axis 2 ranging over separate instances of an
        oscillator (time series values are assumed to represent an angle)
      times (float or sequence of floats): The target times at which 
        to plot the distribution
      nbins (int): number of histogram bins
      colormap
    """
    if times is None:
        times = np.linspace(dts.tspan[0], dts.tspan[-1], num=4)
    elif isinstance(times, numbers.Number):
        times = np.array([times], dtype=np.float64)
    indices = distob.gather(dts.tspan.searchsorted(times))
    if indices[-1] == len(dts.tspan):
        indices[-1] -= 1
    nplots = len(indices)
    fig = plt.figure()
    n = np.zeros((nbins, nplots))
    for i in range(nplots):
        index = indices[i]
        time = dts.tspan[index]
        phases = distob.gather(dts.mod2pi()[index, 0, :])
        ax = fig.add_subplot(1, nplots, i + 1, projection='polar')
        n[:,i], bins, patches = ax.hist(phases, nbins, (-np.pi, np.pi), 
                                        density=True, histtype='bar')
        ax.set_title('time = %d s' % time)
        ax.set_xticklabels(['0', r'$\frac{\pi}{4}$', r'$\frac{\pi}{2}$', 
                            r'$\frac{3\pi}{4}$', r'$\pi$', r'$\frac{-3\pi}{4}$',
                            r'$\frac{-\pi}{2}$', r'$\frac{-\pi}{4}$'])
    nmin, nmax = n.min(), n.max()
    #TODO should make a custom colormap instead of reducing color dynamic range:
    norm = mpl.colors.Normalize(1.2*nmin - 0.2*nmax, 
                                0.6*nmin + 0.4*nmax, clip=True)
    for i in range(nplots):
        ax = fig.get_axes()[i]
        ax.set_ylim(0, nmax)
        for this_n, thispatch in zip(n[:,i], ax.patches):
            color = colormap(norm(this_n))
            thispatch.set_facecolor(color)
            thispatch.set_edgecolor(color)
    fig.show()