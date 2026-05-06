def isAspecting(obj1, obj2, aspList):
    """ Returns if obj1 aspects obj2 within its orb,
    considering a list of possible aspect types. 
    
    """
    aspDict = _aspectDict(obj1, obj2, aspList)
    if aspDict:
        return aspDict['orb'] < obj1.orb()
    return False