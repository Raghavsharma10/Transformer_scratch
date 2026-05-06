def validate_totalflux(totalflux):
    """Check integrated flux for invalid values.

    Parameters
    ----------
    totalflux : float
        Integrated flux.

    Raises
    ------
    synphot.exceptions.SynphotError
        Input is zero, negative, or not a number.

    """
    if totalflux <= 0.0:
        raise exceptions.SynphotError('Integrated flux is <= 0')
    elif np.isnan(totalflux):
        raise exceptions.SynphotError('Integrated flux is NaN')
    elif np.isinf(totalflux):
        raise exceptions.SynphotError('Integrated flux is infinite')