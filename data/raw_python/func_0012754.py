def getobject(bunchdt, key, name):
    """get the object if you have the key and the name
    returns a list of objects, in case you have more than one
    You should not have more than one"""
    # TODO : throw exception if more than one object, or return more objects
    idfobjects = bunchdt[key]
    if idfobjects:
        # second item in list is a unique ID
        unique_id = idfobjects[0].objls[1]
    theobjs = [idfobj for idfobj in idfobjects if
               idfobj[unique_id].upper() == name.upper()]
    try:
        return theobjs[0]
    except IndexError:
        return None