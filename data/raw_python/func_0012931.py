def getfieldnamesendswith(idfobject, endswith):
    """get the filednames for the idfobject based on endswith"""
    objls = idfobject.objls
    tmp = [name for name in objls if name.endswith(endswith)]
    if tmp == []:
        pass
    return [name for name in objls if name.endswith(endswith)]