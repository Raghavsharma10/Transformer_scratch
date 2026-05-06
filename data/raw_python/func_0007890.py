def sweFixedStar(star, jd):
    """ Returns a fixed star from the Ephemeris. """
    sweList = swisseph.fixstar_ut(star, jd)
    mag = swisseph.fixstar_mag(star)
    return {
        'id': star, 
        'mag': mag,
        'lon': sweList[0],
        'lat': sweList[1]
    }