def get_eff_gain(base_std, base_std_unc, meth_std, meth_std_unc, adjust=1):
    r"""Calculates efficiency gain for a new method compared to a base method.
    Given the variation in repeated calculations' results using the two
    methods, the efficiency gain is:

    .. math::

        \mathrm{efficiency\,gain}
        =
        \frac{\mathrm{Var[base\,method]}}{\mathrm{Var[new\,method]}}

    The uncertainty on the efficiency gain is also calculated.

    See the dynamic nested sampling paper (Higson et al. 2019) for more
    details.

    Parameters
    ----------
    base_std: 1d numpy array
    base_std_unc: 1d numpy array
        Uncertainties on base_std.
    meth_std: 1d numpy array
    meth_std_unc: 1d numpy array
        Uncertainties on base_std.

    Returns
    -------
    gain: 1d numpy array
    gain_unc: 1d numpy array
        Uncertainties on gain.
    """
    ratio = base_std / meth_std
    ratio_unc = array_ratio_std(
        base_std, base_std_unc, meth_std, meth_std_unc)
    gain = ratio ** 2
    gain_unc = 2 * ratio * ratio_unc
    gain *= adjust
    gain_unc *= adjust
    return gain, gain_unc