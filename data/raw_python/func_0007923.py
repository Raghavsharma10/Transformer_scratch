def aspectType(obj1, obj2, aspList):
    """ Returns the aspect type between objects considering
    a list of possible aspect types.
    
    """
    ap = _getActivePassive(obj1, obj2)
    aspDict = _aspectDict(ap['active'], ap['passive'], aspList)
    return aspDict['type'] if aspDict else const.NO_ASPECT