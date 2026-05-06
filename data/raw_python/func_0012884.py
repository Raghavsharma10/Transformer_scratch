def splittermixerfieldlists(data, commdct, objkey):
    """docstring for splittermixerfieldlists"""
    objkey = objkey.upper()
    objindex = data.dtls.index(objkey)
    objcomms = commdct[objindex]
    theobjects = data.dt[objkey]
    fieldlists = []
    for theobject in theobjects:
        fieldlist = list(range(1, len(theobject)))
        fieldlists.append(fieldlist)
    return fieldlists