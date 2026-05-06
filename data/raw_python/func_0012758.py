def getextensibleindex(bunchdt, data, commdct, key, objname):
    """get the index of the first extensible item"""
    theobject = getobject(bunchdt, key, objname)
    if theobject == None:
        return None
    theidd = iddofobject(data, commdct, key)
    extensible_i = [
        i for i in range(len(theidd)) if 'begin-extensible' in theidd[i]]
    try:
        extensible_i = extensible_i[0]
    except IndexError:
        return theobject