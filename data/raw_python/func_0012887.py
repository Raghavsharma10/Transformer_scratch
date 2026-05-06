def repeatingfields(theidd, commdct, objkey, flds):
    """return a list of repeating fields
    fld is in format 'Component %s Name'
    so flds = [fld % (i, ) for i in range(n)]
    does not work for 'fields as indicated' """
    # TODO : make it work for 'fields as indicated'
    if type(flds) != list:
        flds = [flds] # for backward compatability
    objindex = theidd.dtls.index(objkey)
    objcomm = commdct[objindex]
    allfields = []
    for fld in flds:
        thefields = []
        indx = 1
        for i in range(len(objcomm)):
            try:
                thefield = fld % (indx, )
                if objcomm[i]['field'][0] == thefield:
                    thefields.append(thefield)
                    indx = indx + 1
            except KeyError as err:
                pass
        allfields.append(thefields)
    allfields = list(zip(*allfields))
    return [item for sublist in allfields for item in sublist]