def updateidf(idf, dct):
    """update idf using dct"""
    for key in list(dct.keys()):
        if key.startswith('idf.'):
            idftag, objkey, objname, field = key2elements(key)
            if objname == '':
                try:
                    idfobj = idf.idfobjects[objkey.upper()][0]
                except IndexError as e:
                    idfobj = idf.newidfobject(objkey.upper())
            else:
                idfobj = idf.getobject(objkey.upper(), objname)
                if idfobj == None:
                    idfobj = idf.newidfobject(objkey.upper(), Name=objname)
            idfobj[field] = dct[key]