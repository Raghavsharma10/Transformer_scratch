def almutem(sign, lon):
    """ Returns the almutem for a given
    sign and longitude.

    """
    planets = const.LIST_SEVEN_PLANETS
    res = [None, 0]
    for ID in planets:
        sc = score(ID, sign, lon)
        if sc > res[1]:
            res = [ID, sc]
    return res[0]