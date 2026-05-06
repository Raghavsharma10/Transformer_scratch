def getHouses(date, pos, hsys):
    """ Returns the lists of houses and angles.
    
    Since houses and angles are computed at the
    same time, this function should be fast.
    
    """
    houses, angles = eph.getHouses(date.jd, pos.lat, pos.lon, hsys)
    hList = [House.fromDict(house) for house in houses]
    aList = [GenericObject.fromDict(angle) for angle in angles]
    return (HouseList(hList), GenericList(aList))