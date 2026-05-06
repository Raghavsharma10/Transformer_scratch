def spmt(t, peak_delay=6, under_delay=16, peak_disp=1, under_disp=1,
         p_u_ratio=6):
    """Normalized SPM HRF function from sum of two gamma PDFs

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
    [1] This is the canonical HRF function as used in SPM. It
    has the following defaults:
        - delay of response (relative to onset) : 6s
        - delay of undershoot (relative to onset) : 16s
        - dispersion of response : 1s
        - dispersion of undershoot : 1s
        - ratio of response to undershoot : 6s
        - onset : 0s
        - length of kernel : 32s

    References:
    -----
    [1] http://nipy.org/
    [2] https://github.com/fabianp/hrf_estimation
    """
    return spm_hrf_compat(t, peak_delay=peak_delay, under_delay=under_delay,
                          peak_disp=peak_disp, under_disp=under_disp,
                          p_u_ratio=p_u_ratio, normalize=True)