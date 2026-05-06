def getObjectList(IDs, date, pos):
    """ Returns a list of objects. """
    objList = [getObject(ID, date, pos) for ID in IDs]
    return ObjectList(objList)