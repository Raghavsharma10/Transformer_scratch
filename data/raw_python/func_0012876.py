def getobjectref(blocklst, commdct):
    """
    makes a dictionary of object-lists
    each item in the dictionary points to a list of tuples
    the tuple is (objectname,  fieldindex)
    """
    objlst_dct = {}
    for eli in commdct:
        for elj in eli:
            if 'object-list' in elj:
                objlist = elj['object-list'][0]
                objlst_dct[objlist] = []

    for objlist in list(objlst_dct.keys()):
        for i in range(len(commdct)):
            for j in range(len(commdct[i])):
                if 'reference' in commdct[i][j]:
                    for ref in commdct[i][j]['reference']:
                        if ref == objlist:
                            objlst_dct[objlist].append((blocklst[i][0], j))
    return objlst_dct