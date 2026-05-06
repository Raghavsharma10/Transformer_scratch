def getObject(ID, date, pos):
    """ Returns an ephemeris object. """
    obj = eph.getObject(ID, date.jd, pos.lat, pos.lon)
    return Object.fromDict(obj)