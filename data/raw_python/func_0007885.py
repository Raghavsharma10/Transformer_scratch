def sweObject(obj, jd):
    """ Returns an object from the Ephemeris. """
    sweObj = SWE_OBJECTS[obj]
    sweList = swisseph.calc_ut(jd, sweObj)
    return {
        'id': obj,
        'lon': sweList[0],
        'lat': sweList[1],
        'lonspeed': sweList[3],
        'latspeed': sweList[4]
    }