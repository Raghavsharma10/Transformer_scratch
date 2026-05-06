def __log_density_single(x, mean, covar):
    """ This is just a test function to calculate 
        the normal density at x given mean and covariance matrix.

        Note: this function is not efficient, so
             _log_multivariate_density is recommended for use.
    """
    n_dim = mean.shape[0]

    dx = x - mean
    covar_inv = scipy.linalg.inv(covar)
    covar_det = scipy.linalg.det(covar)

    den = np.dot(np.dot(dx.T, covar_inv), dx) + n_dim*np.log(2*np.pi) + np.log(covar_det)

    return(-1/2 * den)