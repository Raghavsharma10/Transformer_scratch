def removeextensibles(bunchdt, data, commdct, key, objname):
    """remove the extensible items in the object"""
    theobject = getobject(bunchdt, key, objname)
    if theobject == None:
        return theobject
    theidd = iddofobject(data, commdct, key)
    extensible_i = [
        i for i in range(len(theidd)) if 'begin-extensible' in theidd[i]]
    try:
        extensible_i = extensible_i[0]
    except IndexError:
        return theobject
    while True:
        try:
            popped = theobject.obj.pop(extensible_i)
        except IndexError:
            break
    return theobject