def _orbList(obj1, obj2, aspList):
    """ Returns a list with the orb and angular
    distances from obj1 to obj2, considering a
    list of possible aspects. 
    
    """
    sep = angle.closestdistance(obj1.lon, obj2.lon)
    absSep = abs(sep)
    return [
        {
            'type': asp,
            'orb': abs(absSep - asp),
            'separation': sep,
        } for asp in aspList
    ]