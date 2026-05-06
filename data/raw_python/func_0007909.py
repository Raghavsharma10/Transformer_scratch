def pfLon(jd, lat, lon):
    """ Returns the ecliptic longitude of Pars Fortuna.
    It considers diurnal or nocturnal conditions.
    
    """
    sun = swe.sweObjectLon(const.SUN, jd)
    moon = swe.sweObjectLon(const.MOON, jd)
    asc = swe.sweHousesLon(jd, lat, lon,
                           const.HOUSES_DEFAULT)[1][0]
    
    if isDiurnal(jd, lat, lon):
        return angle.norm(asc + moon - sun)
    else:
        return angle.norm(asc + sun - moon)