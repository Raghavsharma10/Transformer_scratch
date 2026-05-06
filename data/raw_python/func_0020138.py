def GetCovariance(kernel, kernel_params, time, errors):
    '''
    Returns the covariance matrix for a given light curve
    segment.

    :param array_like kernel_params: A list of kernel parameters \
          (white noise amplitude, red noise amplitude, and red noise timescale)
    :param array_like time: The time array (*N*)
    :param array_like errors: The data error array (*N*)

    :returns: The covariance matrix :py:obj:`K` (*N*,*N*)

    '''

    # NOTE: We purposefully compute the covariance matrix
    # *without* the GP white noise term
    K = np.diag(errors ** 2)
    K += GP(kernel, kernel_params, white=False).get_matrix(time)
    return K