def findpeak(x, y, dy=None, position=None, hwhm=None, baseline=None, amplitude=None, curve='Lorentz'):
    """Find a (positive) peak in the dataset.
    
    This function is deprecated, please consider using findpeak_single() instead.

    Inputs:
        x, y, dy: abscissa, ordinate and the error of the ordinate (can be None)
        position, hwhm, baseline, amplitude: first guesses for the named parameters
        curve: 'Gauss' or 'Lorentz' (default)
        
    Outputs:
        peak position, error of peak position, hwhm, error of hwhm, baseline,
            error of baseline, amplitude, error of amplitude.

    Notes:
        A Gauss or a Lorentz curve is fitted, depending on the value of 'curve'.
        
    """
    warnings.warn('Function findpeak() is deprecated, please use findpeak_single() instead.', DeprecationWarning)
    pos, hwhm, baseline, ampl = findpeak_single(x, y, dy, position, hwhm, baseline, amplitude, curve)
    return pos.val, pos.err, hwhm.val, hwhm.err, baseline.val, baseline.err, ampl.val, ampl.err