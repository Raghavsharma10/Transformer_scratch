def reps_to_intensity(reps, slope=-4.8, constant=97.5, quadratic=True):
    """A function mapping from repetitions in the range 1 to 12
    to intensities in the range 0 to 100.

    Parameters
    ----------
    reps
        The number of repetitions to map to the intensity range.
    slope
        Slope for the linear function.
    constant
        Constant for the linear function
    quadratic
        If 'True', add a slight quadratic offset.


    Returns
    -------
    intensity
        An intensity value in the range from 0 to 100.


    Examples
    -------
    >>> reps_to_intensity(5, slope = -5, constant = 100, quadratic = False)
    80
    
    >>> reps_to_intensity(8, slope = -5, constant = 100, quadratic = True)
    67.45
    
    >>> reps_to_intensity(8, slope = -5, constant = 100, quadratic = False)
    65
    """
    intensity = constant + slope * (reps - 1)
    if quadratic:
        return intensity + 0.05 * (reps - 1) ** 2
    else:
        return intensity