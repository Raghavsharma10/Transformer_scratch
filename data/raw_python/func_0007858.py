def getArc(prom, sig, mc, pos, zerolat):
    """ Returns the arc of direction between a promissor
    and a significator. Arguments are also the MC, the
    geoposition and zerolat to assume zero ecliptical 
    latitudes.
    
    ZeroLat true => inZodiaco, false => inMundo
    
    """
    pRA, pDecl = prom.eqCoords(zerolat)
    sRa, sDecl = sig.eqCoords(zerolat)
    mcRa, mcDecl = mc.eqCoords()
    return arc(pRA, pDecl, sRa, sDecl, mcRa, pos.lat)