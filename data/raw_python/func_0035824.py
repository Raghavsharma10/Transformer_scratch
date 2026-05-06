def planetType(temperature, mass, radius):
    """ Returns the planet type as 'temperatureType massType'
    """

    if mass is not np.nan:
        sizeType = planetMassType(mass)
    elif radius is not np.nan:
        sizeType = planetRadiusType(radius)
    else:
        return None

    return '{0} {1}'.format(planetTempType(temperature), sizeType)