def ddspmt(t, peak_delay=6, under_delay=16, peak_disp=1, under_disp=1,
           p_u_ratio=6):
    """ SPM canonical HRF dispersion derivative, values for time values `t`

    Parameters
    ----------
    t : array-like
        vector of times at which to sample HRF

    Returns
    -------
    hrf : array
        vector length ``len(t)`` of samples from HRF at times `t`

    Notes
    -----
    [1] This is the canonical HRF dispersion derivative function as used in SPM
    [2] It is the numerical difference between the HRF sampled at time `t`, and
    values at `t` for another HRF shape with a small change in the peak
    dispersion parameter (``peak_disp`` in func:`spm_hrf_compat`).

    References:
    -----
    [1] http://nipy.org/
    [2] https://github.com/fabianp/hrf_estimation
    """

    _spm_dd_func = partial(spmt, peak_delay=peak_delay,
                           under_delay=under_delay,
                           under_disp=under_disp, p_u_ratio=p_u_ratio,
                           peak_disp=1.01)

    return (spmt(t) - _spm_dd_func(t)) / 0.01