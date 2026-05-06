def Cmatrix(x, y, sx, sy, theta):
    """
    Construct a correlation matrix corresponding to the data.
    The matrix assumes a gaussian correlation function.

    Parameters
    ----------
    x, y : array-like
        locations at which to evaluate the correlation matirx
    sx, sy : float
        major/minor axes of the gaussian correlation function (sigmas)

    theta : float
        position angle of the gaussian correlation function (degrees)

    Returns
    -------
    data : array-like
        The C-matrix.
    """
    C = np.vstack([elliptical_gaussian(x, y, 1, i, j, sx, sy, theta) for i, j in zip(x, y)])
    return C