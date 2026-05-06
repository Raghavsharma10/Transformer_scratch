def _aspectDict(obj1, obj2, aspList):
    """ Returns the properties of the aspect of 
    obj1 to obj2, considering a list of possible
    aspects.
    
    This function makes the following assumptions:
    - Syzygy does not start aspects but receives 
      any aspect.
    - Pars Fortuna and Moon Nodes only starts 
      conjunctions but receive any aspect.
    - All other objects can start and receive
      any aspect.
      
    Note: this function returns the aspect
    even if it is not within the orb of obj1
    (but is within the orb of obj2).
    
    """
    # Ignore aspects from same and Syzygy
    if obj1 == obj2 or obj1.id == const.SYZYGY:
        return None
    
    orbs = _orbList(obj1, obj2, aspList)
    for aspDict in orbs:
        asp = aspDict['type']
        orb = aspDict['orb']  
        
        # Check if aspect is within orb
        if asp in const.MAJOR_ASPECTS:
            # Ignore major aspects out of orb
            if obj1.orb() < orb and obj2.orb() < orb:
                continue
        else:
            # Ignore minor aspects out of max orb
            if MAX_MINOR_ASP_ORB < orb:
                continue
            
        # Only conjunctions for Pars Fortuna and Nodes
        if obj1.id in [const.PARS_FORTUNA, 
                       const.NORTH_NODE, 
                       const.SOUTH_NODE] and \
                asp != const.CONJUNCTION:
            continue
        
        # We have a valid aspect within orb
        return aspDict

    return None