def isDiurnal(jd, lat, lon):
    """ Returns true if the sun is above the horizon
    of a given date and location. 
    
    """
    sun = swe.sweObject(const.SUN, jd)
    mc = swe.sweHousesLon(jd, lat, lon, 
                          const.HOUSES_DEFAULT)[1][1]
    ra, decl = utils.eqCoords(sun['lon'], sun['lat'])
    mcRA, _ = utils.eqCoords(mc, 0.0)
    return utils.isAboveHorizon(ra, decl, mcRA, lat)