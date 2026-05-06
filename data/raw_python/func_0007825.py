def orientality(obj, sun):
    """ Returns if an object is oriental or 
    occidental to the sun. 
    
    """
    dist = angle.distance(sun.lon, obj.lon)
    return OCCIDENTAL if dist < 180 else ORIENTAL