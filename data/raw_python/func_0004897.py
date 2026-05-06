def LogNormal(x, a, mu, sigma):
    """PDF of a log-normal distribution

    Inputs:
    -------
        ``x``: independent variable
        ``a``: amplitude
        ``mu``: center parameter
        ``sigma``: width parameter

    Formula:
    --------
        ``a/ (2*pi*sigma^2*x^2)^0.5 * exp(-(log(x)-mu)^2/(2*sigma^2))
    """
    return a / np.sqrt(2 * np.pi * sigma ** 2 * x ** 2) *\
        np.exp(-(np.log(x) - mu) ** 2 / (2 * sigma ** 2))