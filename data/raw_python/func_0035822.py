def planetMassType(mass):
    """ Returns the planet masstype given the mass and using planetAssumptions['massType']
    """

    if mass is np.nan:
        return None

    for massLimit, massType in planetAssumptions['massType']:

        if mass < massLimit:
            return massType