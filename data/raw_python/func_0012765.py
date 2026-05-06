def getallobjlists(idf, refname):
    """get all object-list fields for refname
    return a list:
    [('OBJKEY', refname, fieldindexlist), ...] where
    fieldindexlist = index of the field where the object-list = refname
    """
    dtls = idf.model.dtls
    objlists = []
    for i, fieldidds in enumerate(idf.idd_info):
        indexlist = []
        for j, fieldidd in enumerate(fieldidds):
            if 'object-list' in fieldidd:
                if fieldidd['object-list'][0].upper() == refname.upper():
                    indexlist.append(j)
        if indexlist != []:
            objkey = dtls[i]
            objlists.append((objkey, refname, indexlist))
    return objlists