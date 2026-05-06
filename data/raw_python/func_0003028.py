def dspmt(t, peak_delay=6, under_delay=16, peak_disp=1, under_disp=1,
          p_u_ratio=6):
    """ SPM canonical HRF derivative, HRF derivative values for time values `t`

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
    [1] This is the canonical HRF derivative function as used in SPM.
    [2] It is the numerical difference of the HRF sampled at time `t` minus the
    values sampled at time `t` -1

    References:
    -----
    [1] http://nipy.org/
    [2] https://github.com/fabianp/hrf_estimation
    """
    t = np.asarray(t)
    aryRsp1 = spmt(t, peak_delay=peak_delay, under_delay=under_delay,
                   peak_disp=peak_disp, under_disp=under_disp,
                   p_u_ratio=p_u_ratio)
    aryRsp2 = spmt(t-1, peak_delay=peak_delay, under_delay=under_delay,
                   peak_disp=peak_disp, under_disp=under_disp,
                   p_u_ratio=p_u_ratio)

    return aryRsp1 - aryRsp2