def getFixedStar(ID, date):
    """ Returns a fixed star from the ephemeris. """
    star = eph.getFixedStar(ID, date.jd)
    return FixedStar.fromDict(star)