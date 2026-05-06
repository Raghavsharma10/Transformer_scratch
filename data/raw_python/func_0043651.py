def plot_lightcurve(name, lightcurve, period, data,
                    output='.', legend=False, sanitize_latex=False,
                    color=True, n_phases=100,
                    err_const=0.005,
                    **kwargs):
    """plot_lightcurve(name, lightcurve, period, data, output='.', legend=False, color=True, n_phases=100, err_const=0.005, **kwargs)

    Save a plot of the given *lightcurve* to directory *output*.

    **Parameters**

    name : str
        Name of the star. Used in filename and plot title.
    lightcurve : array-like, shape = [n_samples]
        Fitted lightcurve.
    period : number
        Period to phase time by.
    data : array-like, shape = [n_samples, 2] or [n_samples, 3]
        Photometry array containing columns *time*, *magnitude*, and
        (optional) *error*. *time* should be unphased.
    output : str, optional
        Directory to save plot to (default '.').
    legend : boolean, optional
        Whether or not to display legend on plot (default False).
    color : boolean, optional
        Whether or not to display color in plot (default True).
    n_phases : integer, optional
        Number of phase points in fit (default 100).
    err_const : number, optional
        Constant to use in absence of error (default 0.005).

    **Returns**

    None
    """
    phases = numpy.linspace(0, 1, n_phases, endpoint=False)
    ax = plt.gca()
    ax.invert_yaxis()
    plt.xlim(0,2)

    # Plot points used
    phase, mag, *err = get_signal(data).T

    error = err[0] if err else mag*err_const

    inliers = plt.errorbar(numpy.hstack((phase,1+phase)),
                           numpy.hstack((mag, mag)),
                           yerr=numpy.hstack((error, error)),
                           ls='None',
                           ms=.01, mew=.01, capsize=0)

    # Plot outliers rejected
    phase, mag, *err = get_noise(data).T

    error = err[0] if err else mag*err_const

    outliers = plt.errorbar(numpy.hstack((phase,1+phase)),
                            numpy.hstack((mag, mag)),
                            yerr=numpy.hstack((error, error)),
                            ls='None', marker='o' if color else 'x',
                            ms=.01 if color else 4,
                            mew=.01 if color else 1,
                            capsize=0 if color else 1)

    # Plot the fitted light curve
    signal, = plt.plot(numpy.hstack((phases,1+phases)),
                       numpy.hstack((lightcurve, lightcurve)),
                       linewidth=1)

    if legend:
        plt.legend([signal, inliers, outliers],
                   ["Light Curve", "Inliers", "Outliers"],
                   loc='best')

    plt.xlabel('Phase ({0:0.7} day period)'.format(period))
    plt.ylabel('Magnitude')

    plt.title(utils.sanitize_latex(name) if sanitize_latex else name)
    plt.tight_layout(pad=0.1)
    make_sure_path_exists(output)
    plt.savefig(path.join(output, name))
    plt.clf()