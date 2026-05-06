def nextSolarReturn(date, lon):
    """ Returns the next date when sun is at longitude 'lon'. """
    jd = eph.nextSolarReturn(date.jd, lon)
    return Datetime.fromJD(jd, date.utcoffset)