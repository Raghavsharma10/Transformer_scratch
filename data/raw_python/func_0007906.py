def nextSunrise(jd, lat, lon):
    """ Returns the JD of the next sunrise. """
    return swe.sweNextTransit(const.SUN, jd, lat, lon, 'RISE')