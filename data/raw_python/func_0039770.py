def calc_exp_uncertainty(n, std, combined_unc, sys_unc, rel_unc=0.25,
                         confidence=0.95, mean=True):
    """Calculate expanded uncertainty.

    Parameters
    ----------
    n : Number of independent samples
    std : Sample standard deviation
    sys_unc : Systematic uncertainty (b in Coleman and Steele)
    rel_unc : Relative uncertainty of each systematic error source (guess 0.25)
    confidence : Confidence interval (0, 1)
    mean : bool whether or not the quantity is a mean value

    Returns
    -------
    exp_unc : Expanded uncertainty
    nu_x : Degrees of freedom
    """
    s_x = std
    if mean:
        s_x /= np.sqrt(n)
    nu_s_x = n - 1
    b = sys_unc
    nu_b = 0.5*rel_unc**(-2)
    nu_x = ((s_x**2 + b**2)**2)/(s_x**4/nu_s_x + b**4/nu_b)
    t = scipy.stats.t.interval(alpha=0.95, df=nu_x)[-1]
    exp_unc = t*combined_unc
    return exp_unc, nu_x