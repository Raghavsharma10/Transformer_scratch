def sweObjectLon(obj, jd):
    """ Returns the longitude of an object. """
    sweObj = SWE_OBJECTS[obj]
    sweList = swisseph.calc_ut(jd, sweObj)
    return sweList[0]