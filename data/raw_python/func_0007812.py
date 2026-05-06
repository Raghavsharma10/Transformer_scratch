def nextSunrise(date, pos):
    """ Returns the date of the next sunrise. """
    jd = eph.nextSunrise(date.jd, pos.lat, pos.lon)
    return Datetime.fromJD(jd, date.utcoffset)