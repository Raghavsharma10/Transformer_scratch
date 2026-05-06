def isAboveHorizon(ra, decl, mcRA, lat):
    """ Returns if an object's 'ra' and 'decl' 
    is above the horizon at a specific latitude, 
    given the MC's right ascension.
    
    """
    # This function checks if the equatorial distance from 
    # the object to the MC is within its diurnal semi-arc.
    
    dArc, _ = dnarcs(decl, lat)
    dist = abs(angle.closestdistance(mcRA, ra))
    return dist <= dArc/2.0 + 0.0003