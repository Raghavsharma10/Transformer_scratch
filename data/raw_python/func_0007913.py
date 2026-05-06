def nextStationJD(ID, jd):
    """ Finds the aproximate julian date of the
    next station of a planet.

    """
    speed = swe.sweObject(ID, jd)['lonspeed']
    for i in range(2000):
        nextjd = jd + i / 2
        nextspeed = swe.sweObject(ID, nextjd)['lonspeed']
        if speed * nextspeed <= 0:
            return nextjd
    return None