def sweHouses(jd, lat, lon, hsys):
    """ Returns lists of houses and angles. """
    hsys = SWE_HOUSESYS[hsys]
    hlist, ascmc = swisseph.houses(jd, lat, lon, hsys)
    # Add first house to the end of 'hlist' so that we
    # can compute house sizes with an iterator 
    hlist += (hlist[0],)
    houses = [
        {
            'id': const.LIST_HOUSES[i],
            'lon': hlist[i], 
            'size': angle.distance(hlist[i], hlist[i+1])
        } for i in range(12)
    ]
    angles = [
        {'id': const.ASC, 'lon': ascmc[0]}, 
        {'id': const.MC, 'lon': ascmc[1]},
        {'id': const.DESC, 'lon': angle.norm(ascmc[0] + 180)},
        {'id': const.IC, 'lon': angle.norm(ascmc[1] + 180)}
    ]
    return (houses, angles)