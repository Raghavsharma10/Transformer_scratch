def nextSunset(jd, lat, lon):
    """ Returns the JD of the next sunset. """
    return swe.sweNextTransit(const.SUN, jd, lat, lon, 'SET')