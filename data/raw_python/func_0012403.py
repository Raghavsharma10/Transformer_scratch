def rmse_and_unc(values_array, true_values):
    r"""Calculate the root meet squared error and its numerical uncertainty.

    With a reasonably large number of values in values_list the uncertainty
    on sq_errors should be approximately normal (from the central limit
    theorem).
    Uncertainties are calculated via error propagation: if :math:`\sigma`
    is the error on :math:`X` then the error on :math:`\sqrt{X}`
    is :math:`\frac{\sigma}{2 \sqrt{X}}`.

    Parameters
    ----------
    values_array: 2d numpy array
        Array of results: each row corresponds to a different estimate of the
        quantities considered.
    true_values: 1d numpy array
        Correct values for the quantities considered.

    Returns
    -------
    rmse: 1d numpy array
        Root-mean-squared-error for each quantity.
    rmse_unc: 1d numpy array
        Numerical uncertainties on each element of rmse.
    """
    assert true_values.shape == (values_array.shape[1],)
    errors = values_array - true_values[np.newaxis, :]
    sq_errors = errors ** 2
    sq_errors_mean = np.mean(sq_errors, axis=0)
    sq_errors_mean_unc = (np.std(sq_errors, axis=0, ddof=1) /
                          np.sqrt(sq_errors.shape[0]))
    rmse = np.sqrt(sq_errors_mean)
    rmse_unc = 0.5 * (1 / rmse) * sq_errors_mean_unc
    return rmse, rmse_unc