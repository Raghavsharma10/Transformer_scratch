def combine_with_wd_noise(f_n, amp_n, f_n_wd, amp_n_wd):
    """Combine noise with wd noise.

    Combines noise and white dwarf background noise based on greater
    amplitude value at each noise curve step.

    Args:
        f_n (float array): Frequencies of noise curve.
        amp_n (float array): Amplitude values of noise curve.
        f_n_wd (float array): Frequencies of wd noise.
        amp_n_wd (float array): Amplitude values of wd noise.

    Returns:
        (tuple of float arrays): Amplitude values of combined noise curve.

    """

    # interpolate wd noise
    amp_n_wd_interp = interpolate.interp1d(f_n_wd, amp_n_wd, bounds_error=False, fill_value=1e-30)

    # find points of wd noise amplitude at noise curve frequencies
    amp_n_wd = amp_n_wd_interp(f_n)

    # keep the greater value at each frequency
    amp_n = amp_n*(amp_n >= amp_n_wd) + amp_n_wd*(amp_n < amp_n_wd)
    return f_n, amp_n