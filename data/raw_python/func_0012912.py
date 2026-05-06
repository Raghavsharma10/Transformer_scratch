def name2idfobject(idf, groupnamess=None, objkeys=None, **kwargs):
    """return the object, if the Name or some other field is known.
    send filed in **kwargs as Name='a name', Roughness='smooth'
    Returns the first find (field search is unordered)
    objkeys -> if objkeys=['ZONE', 'Material'], search only those
    groupnames -> not yet coded"""
    # TODO : this is a very slow search. revist to speed it up.
    if not objkeys:
        objkeys = idfobjectkeys(idf)
    for objkey in objkeys:
        idfobjs = idf.idfobjects[objkey.upper()]
        for idfobj in idfobjs:
            for key, val in kwargs.items():
                try:
                    if idfobj[key] == val:
                        return idfobj
                except BadEPFieldError as e:
                    continue