def planetRadiusType(radius):
    """ Returns the planet radiustype given the mass and using planetAssumptions['radiusType']
    """

    if radius is np.nan:
        return None

    for radiusLimit, radiusType in planetAssumptions['radiusType']:

        if radius < radiusLimit:
            return radiusType