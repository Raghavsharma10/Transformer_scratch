def getrefnames(idf, objname):
    """get the reference names for this object"""
    iddinfo = idf.idd_info
    dtls = idf.model.dtls
    index = dtls.index(objname)
    fieldidds = iddinfo[index]
    for fieldidd in fieldidds:
        if 'field' in fieldidd:
            if fieldidd['field'][0].endswith('Name'):
                if 'reference' in fieldidd:
                    return fieldidd['reference']
                else:
                    return []