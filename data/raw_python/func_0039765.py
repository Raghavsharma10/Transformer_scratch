def integral_scale(u, t, tau1=0.0, tau2=1.0):
    """Calculate the integral scale of a time series by integrating up to
    the first zero crossing.
    """
    tau, rho = autocorr_coeff(u, t, tau1, tau2)
    zero_cross_ind = np.where(np.diff(np.sign(rho)))[0][0]
    int_scale = np.trapz(rho[:zero_cross_ind], tau[:zero_cross_ind])
    return int_scale