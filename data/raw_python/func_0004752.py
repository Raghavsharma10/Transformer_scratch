def fit_gaussian(x, y, yerr, p0):
    """ Fit a Gaussian to the data """
    try:
        popt, pcov = curve_fit(gaussian, x, y, sigma=yerr, p0=p0, absolute_sigma=True)
    except RuntimeError:
        return [0],[0]
    return popt, pcov