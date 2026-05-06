def getFixedStar(ID, jd):
    """ Returns a fixed star. """
    star = swe.sweFixedStar(ID, jd)
    _signInfo(star)
    return star