def idfdiffs(idf1, idf2):
    """return the diffs between the two idfs"""
    # for any object type, it is sorted by name
    thediffs = {}
    keys = idf1.model.dtls # undocumented variable

    for akey in keys:
        idfobjs1 = idf1.idfobjects[akey]
        idfobjs2 = idf2.idfobjects[akey]
        names = set([getobjname(i) for i in idfobjs1] +
                    [getobjname(i) for i in idfobjs2])
        names = sorted(names)
        idfobjs1 = sorted(idfobjs1, key=lambda idfobj: idfobj['obj'])
        idfobjs2 = sorted(idfobjs2, key=lambda idfobj: idfobj['obj'])
        for name in names:
            n_idfobjs1 = [item for item in idfobjs1
                          if getobjname(item) == name]
            n_idfobjs2 = [item for item in idfobjs2
                          if getobjname(item) == name]
            for idfobj1, idfobj2 in zip_longest(n_idfobjs1,
                                                           n_idfobjs2):
                if idfobj1 == None:
                    thediffs[(idfobj2.key.upper(), 
                                getobjname(idfobj2))] = (None, idf2.idfname) #(idf1.idfname, None) -> old
                    break
                if idfobj2 == None:
                    thediffs[(idfobj1.key.upper(), 
                                getobjname(idfobj1))] = (idf1.idfname, None) # (None, idf2.idfname) -> old
                    break
                for i, (f1, f2) in enumerate(zip(idfobj1.obj, idfobj2.obj)):
                    if i == 0:
                        f1, f2 = f1.upper(), f2.upper()
                    if f1 != f2:
                        thediffs[(
                            akey,
                            getobjname(idfobj1),
                            idfobj1.objidd[i]['field'][0])] = (f1, f2)
    return thediffs