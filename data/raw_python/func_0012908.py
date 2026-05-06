def getanymentions(idf, anidfobject):
    """Find out if idjobject is mentioned an any object anywhere"""
    name = anidfobject.obj[1]
    foundobjs = []
    keys = idfobjectkeys(idf)
    idfkeyobjects = [idf.idfobjects[key.upper()] for key in keys]
    for idfobjects in idfkeyobjects:
        for idfobject in idfobjects:
            if name.upper() in [item.upper() 
                                    for item in idfobject.obj 
                                        if isinstance(item, basestring)]:
                foundobjs.append(idfobject)
    return foundobjs