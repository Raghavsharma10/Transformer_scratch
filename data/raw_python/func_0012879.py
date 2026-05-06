def extractfields(data, commdct, objkey, fieldlists):
    """get all the objects of objkey.
    fieldlists will have a fieldlist for each of those objects.
    return the contents of those fields"""
    # TODO : this assumes that the field list identical for
    # each instance of the object. This is not true.
    # So we should have a field list for each instance of the object
    # and map them with a zip
    objindex = data.dtls.index(objkey)
    objcomm = commdct[objindex]
    objfields = []
    # get the field names of that object
    for dct in objcomm[0:]:
        try:
            thefieldcomms = dct['field']
            objfields.append(thefieldcomms[0])
        except KeyError as err:
            objfields.append(None)
    fieldindexes = []
    for fieldlist in fieldlists:
        fieldindex = []
        for item in fieldlist:
            if isinstance(item, int):
                fieldindex.append(item)
            else:
                fieldindex.append(objfields.index(item) + 0)
                # the index starts at 1, not at 0
        fieldindexes.append(fieldindex)
    theobjects = data.dt[objkey]
    fieldcontents = []
    for theobject, fieldindex in zip(theobjects, fieldindexes):
        innerlst = []
        for item in fieldindex:
            try:
                innerlst.append(theobject[item])
            except IndexError as err:
                break
        fieldcontents.append(innerlst)
        # fieldcontents.append([theobject[item] for item in fieldindex])
    return fieldcontents