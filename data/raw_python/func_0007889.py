def sweHousesLon(jd, lat, lon, hsys):
    """ Returns lists with house and angle longitudes. """
    hsys = SWE_HOUSESYS[hsys]
    hlist, ascmc = swisseph.houses(jd, lat, lon, hsys)
    angles = [
        ascmc[0],
        ascmc[1],
        angle.norm(ascmc[0] + 180), 
        angle.norm(ascmc[1] + 180)
    ]
    return (hlist, angles)