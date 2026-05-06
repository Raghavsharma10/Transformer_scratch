def hasAspect(obj1, obj2, aspList):
    """ Returns if there is an aspect between objects 
    considering a list of possible aspect types.
    
    """
    aspType = aspectType(obj1, obj2, aspList)
    return aspType != const.NO_ASPECT