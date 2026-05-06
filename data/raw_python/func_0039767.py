def calc_multi_exp_unc(sys_unc, n, mean, std, dof, confidence=0.95):
    """Calculate expanded uncertainty using values from multiple runs.

    Note that this function assumes the statistic is a mean value, therefore
    the combined standard deviation is divided by `sqrt(N)`.

    Parameters
    ----------
    sys_unc : numpy array of systematic uncertainties
    n : numpy array of numbers of samples per set
    std : numpy array of sample standard deviations
    dof : numpy array of degrees of freedom
    confidence : Confidence interval for t-statistic
    """
    sys_unc = sys_unc.mean()
    std_combined = combine_std(n, mean, std)
    std_combined /= np.sqrt(n.sum())
    std_unc_combined = np.sqrt(std_combined**2 + sys_unc**2)
    dof = dof.sum()
    t_combined = scipy.stats.t.interval(alpha=confidence, df=dof)[-1]
    exp_unc_combined = t_combined*std_unc_combined
    return exp_unc_combined