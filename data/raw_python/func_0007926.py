def getAspect(obj1, obj2, aspList):
    """ Returns an Aspect object for the aspect between two
    objects considering a list of possible aspect types.
    
    """
    ap = _getActivePassive(obj1, obj2)
    aspDict = _aspectDict(ap['active'], ap['passive'], aspList)
    if not aspDict:
        aspDict = {
            'type': const.NO_ASPECT,
            'orb': 0,
            'separation': 0,
        } 
    aspProp = _aspectProperties(ap['active'], ap['passive'], aspDict)
    return Aspect(aspProp)