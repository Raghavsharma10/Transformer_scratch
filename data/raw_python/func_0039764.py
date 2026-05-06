def autocorr_coeff(x, t, tau1, tau2):
    """Calculate the autocorrelation coefficient."""
    return corr_coeff(x, x, t, tau1, tau2)