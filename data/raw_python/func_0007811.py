def prevSolarReturn(date, lon):
    """ Returns the previous date when sun is at longitude 'lon'. """
    jd = eph.prevSolarReturn(date.jd, lon)
    return Datetime.fromJD(jd, date.utcoffset)