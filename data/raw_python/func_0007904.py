def getHouses(jd, lat, lon, hsys):
    """ Returns lists of houses and angles. """
    houses, angles = swe.sweHouses(jd, lat, lon, hsys)
    for house in houses:
        _signInfo(house)
    for angle in angles:
        _signInfo(angle)
    return (houses, angles)