def syzygyJD(jd):
    """ Finds the latest new or full moon and
    returns the julian date of that event. 
    
    """
    sun = swe.sweObjectLon(const.SUN, jd)
    moon = swe.sweObjectLon(const.MOON, jd)
    dist = angle.distance(sun, moon)
    
    # Offset represents the Syzygy type. 
    # Zero is conjunction and 180 is opposition.
    offset = 180 if (dist >= 180) else 0
    while abs(dist) > MAX_ERROR:
        jd = jd - dist / 13.1833  # Moon mean daily motion
        sun = swe.sweObjectLon(const.SUN, jd)
        moon = swe.sweObjectLon(const.MOON, jd)
        dist = angle.closestdistance(sun - offset, moon)
    return jd