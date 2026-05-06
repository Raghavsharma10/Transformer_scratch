def combine_std(n, mean, std):
    """Compute combined standard deviation for subsets.

    See https://stats.stackexchange.com/questions/43159/\
    how-to-calculate-pooled-variance-of-two-groups-given-known-group-variances-\
    mean for derivation.

    Parameters
    ----------
    n : numpy array of sample sizes
    mean : numpy array of sample means
    std : numpy array of sample standard deviations
    """
    # Calculate weighted mean
    mean_tot = np.sum(n*mean)/np.sum(n)
    var_tot = np.sum(n*(std**2 + mean**2))/np.sum(n) - mean_tot**2
    return np.sqrt(var_tot)