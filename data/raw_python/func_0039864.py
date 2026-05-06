def fit_doublegauss_samples(samples,**kwargs):
    """Fits a two-sided Gaussian to a set of samples.

    Calculates 0.16, 0.5, and 0.84 quantiles and passes these to
    `fit_doublegauss` for fitting.

    Parameters
    ----------
    samples : array-like
        Samples to which to fit the Gaussian.

    kwargs
        Keyword arguments passed to `fit_doublegauss`.
    """
    sorted_samples = np.sort(samples)
    N = len(samples)
    med = sorted_samples[N/2]
    siglo = med - sorted_samples[int(0.16*N)]
    sighi = sorted_samples[int(0.84*N)] - med
    return fit_doublegauss(med,siglo,sighi,median=True,**kwargs)