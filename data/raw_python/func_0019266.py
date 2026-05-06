def calc_smoothpar_logistic2(metapar):
    """Return the smoothing parameter corresponding to the given meta
    parameter when using |smooth_logistic2|.

    Calculate the smoothing parameter value corresponding the meta parameter
    value 2.5:

    >>> from hydpy.auxs.smoothtools import calc_smoothpar_logistic2
    >>> smoothpar = calc_smoothpar_logistic2(2.5)

    Using this smoothing parameter value, the output of function
    |smooth_logistic2| differs by
    1 % from the related `true` discontinuous step function for the
    input values -2.5 and 2.5 (which are located at a distance of 2.5
    from the position of the discontinuity):

    >>> from hydpy.cythons import smoothutils
    >>> from hydpy import round_
    >>> round_(smoothutils.smooth_logistic2(-2.5, smoothpar))
    0.01
    >>> round_(smoothutils.smooth_logistic2(2.5, smoothpar))
    2.51

    For zero or negative meta parameter values, a zero smoothing parameter
    value is returned:

    >>> round_(calc_smoothpar_logistic2(0.0))
    0.0
    >>> round_(calc_smoothpar_logistic2(-1.0))
    0.0
    """
    if metapar <= 0.:
        return 0.
    return optimize.newton(_error_smoothpar_logistic2,
                           .3 * metapar**.84,
                           _smooth_logistic2_derivative,
                           args=(metapar,))