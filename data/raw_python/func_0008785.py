def elliptical_gaussian(x, y, amp, xo, yo, sx, sy, theta):
    """
    Generate a model 2d Gaussian with the given parameters.
    Evaluate this model at the given locations x,y.

    Parameters
    ----------
    x, y : numeric or array-like
        locations at which to evaluate the gaussian
    amp : float
        Peak value.
    xo, yo : float
        Center of the gaussian.
    sx, sy : float
        major/minor axes in sigmas
    theta : float
        position angle (degrees) CCW from x-axis

    Returns
    -------
    data : numeric or array-like
        Gaussian function evaluated at the x,y locations.
    """
    try:
        sint, cost = math.sin(np.radians(theta)), math.cos(np.radians(theta))
    except ValueError as e:
        if 'math domain error' in e.args:
            sint, cost = np.nan, np.nan
    xxo = x - xo
    yyo = y - yo
    exp = (xxo * cost + yyo * sint) ** 2 / sx ** 2 \
          + (xxo * sint - yyo * cost) ** 2 / sy ** 2
    exp *= -1. / 2
    return amp * np.exp(exp)