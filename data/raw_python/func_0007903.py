def getObject(ID, jd, lat, lon):
    """ Returns an object for a specific date and 
    location.
    
    """
    if ID == const.SOUTH_NODE:
        obj = swe.sweObject(const.NORTH_NODE, jd)
        obj.update({
            'id': const.SOUTH_NODE,
            'lon': angle.norm(obj['lon'] + 180)
        })
    elif ID == const.PARS_FORTUNA:
        pflon = tools.pfLon(jd, lat, lon)
        obj = {
            'id': ID,
            'lon': pflon,
            'lat': 0,
            'lonspeed': 0,
            'latspeed': 0
        }
    elif ID == const.SYZYGY:
        szjd = tools.syzygyJD(jd)
        obj = swe.sweObject(const.MOON, szjd)
        obj['id'] = const.SYZYGY
    else:
        obj = swe.sweObject(ID, jd)
    
    _signInfo(obj)
    return obj