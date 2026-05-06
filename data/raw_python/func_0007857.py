def arc(pRA, pDecl, sRA, sDecl, mcRA, lat):
    """ Returns the arc of direction between a Promissor 
    and Significator. It uses the generic proportional 
    semi-arc method.
    
    """
    pDArc, pNArc = utils.dnarcs(pDecl, lat)
    sDArc, sNArc = utils.dnarcs(sDecl, lat)
    
    # Select meridian and arcs to be used
    # Default is MC and Diurnal arcs
    mdRA = mcRA
    sArc = sDArc
    pArc = pDArc
    if not utils.isAboveHorizon(sRA, sDecl, mcRA, lat):
        # Use IC and Nocturnal arcs
        mdRA = angle.norm(mcRA + 180)
        sArc = sNArc
        pArc = pNArc
        
    # Promissor and Significator distance to meridian
    pDist = angle.closestdistance(mdRA, pRA)
    sDist = angle.closestdistance(mdRA, sRA)
    
    # Promissor should be after significator (in degrees)
    if pDist < sDist:
        pDist += 360
        
    # Meridian distances proportional to respective semi-arcs
    sPropDist = sDist / (sArc / 2.0)
    pPropDist = pDist / (pArc / 2.0)
    
    # The arc is how much of the promissor's semi-arc is
    # needed to reach the significator
    return (pPropDist - sPropDist) * (pArc / 2.0)